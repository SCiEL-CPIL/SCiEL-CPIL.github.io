import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)
    
    # Convert subsets_of_block to sorted lists for fast intersection
    for j in range(n_k):
        subsets_of_block[j].sort()
    
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    M_T = M.T

    rng = np.random.default_rng(42)

    # 2. LP Relaxation for initial weights
    duals = None
    try:
        if time.time() < deadline - 40:
            c_lp = np.ones(n_k)
            b_ub = -np.ones(n_t)
            bounds_lp = [(0, 1)] * n_k
            res = linprog(c_lp, A_ub=-M, b_ub=b_ub, bounds=bounds_lp, method='highs',
                          options={"presolve": True, "time_limit": 15})
            if res.success:
                duals = np.abs(res.ineqlin.marginals)
                if np.sum(duals) == 0:
                    duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # 3. Greedy initialization with LP duals
    def greedy_solve(noise_scale=0.0):
        """Greedy set cover using LP duals as weights."""
        gains = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                gains[j] = np.sum(duals[subsets_of_block[j]])
        
        if noise_scale > 0:
            gains += rng.uniform(0, noise_scale, n_k)
        
        uncovered = np.ones(n_t, dtype=bool)
        selected = np.zeros(n_k, dtype=bool)
        solution = []
        
        # Use heap for efficiency
        heap = [(-gains[j], j) for j in range(n_k) if gains[j] > 0]
        heapq.heapify(heap)
        
        while np.any(uncovered):
            if time.time() > deadline - 3:
                return None
            
            best_j = -1
            while heap:
                neg_g, j = heapq.heappop(heap)
                if not selected[j]:
                    best_j = j
                    break
            
            if best_j == -1:
                # Fallback: find any block covering uncovered subsets
                for j in range(n_k):
                    if selected[j]:
                        continue
                    if any(uncovered[s] for s in subsets_of_block[j]):
                        best_j = j
                        break
                if best_j == -1:
                    return None
            
            solution.append(best_j)
            selected[best_j] = True
            
            subs_b = subsets_of_block[best_j]
            newly = [s for s in subs_b if uncovered[s]]
            if not newly:
                continue
            uncovered[newly] = False
            
            # Update gains for remaining blocks
            delta = np.zeros(n_t)
            delta[newly] = duals[newly]
            dec = M_T.dot(delta)
            
            for j in range(n_k):
                if not selected[j] and dec[j] > 0:
                    gains[j] -= dec[j]
                    if gains[j] > 1e-9:
                        heapq.heappush(heap, (-gains[j], j))
        
        return solution

    # Initial solution
    best_sol = greedy_solve()
    if best_sol is None:
        # Pure greedy fallback
        best_sol = []
        uncovered = np.ones(n_t, dtype=bool)
        selected = np.zeros(n_k, dtype=bool)
        while np.any(uncovered):
            best_j = -1
            best_count = 0
            for j in range(n_k):
                if selected[j]:
                    continue
                count = sum(1 for s in subsets_of_block[j] if uncovered[s])
                if count > best_count:
                    best_count = count
                    best_j = j
            if best_j == -1:
                break
            best_sol.append(best_j)
            selected[best_j] = True
            uncovered[subsets_of_block[best_j]] = False
        if not best_sol:
            return [list(k_subsets[0])]

    best_size = len(best_sol)
    best_sol_set = set(best_sol)

    # Helper: compute coverage count vector
    def compute_cover_count(sol):
        cc = np.zeros(n_t, dtype=np.int32)
        for b in sol:
            cc[subsets_of_block[b]] += 1
        return cc

    # Helper: check if solution is valid
    def is_valid(sol):
        cc = compute_cover_count(sol)
        return not np.any(cc == 0)

    # 4. Iterated Local Search with Simulated Annealing
    def local_search(sol, max_iters=500):
        """Local search: try removals and swaps to minimize size."""
        sol = list(sol)
        sol_set = set(sol)
        cc = compute_cover_count(sol)
        
        # Find blocks with unique subsets
        def get_unique_subs(sol_list, cc):
            unique = {}
            for b in sol_list:
                uniq = [s for s in subsets_of_block[b] if cc[s] == 1]
                if uniq:
                    unique[b] = uniq
            return unique
        
        unique = get_unique_subs(sol, cc)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try removal
            for i, b in enumerate(sol):
                if b in unique:
                    continue
                # Can remove b
                del sol[i]
                sol_set.discard(b)
                cc[subsets_of_block[b]] -= 1
                improved = True
                unique = get_unique_subs(sol, cc)
                break
            
            if improved:
                continue
            
            # Try swap: remove b, add b' that covers b's unique subsets
            for i, b_remove in enumerate(sol):
                if b_remove not in unique:
                    continue
                uniq_set = set(unique[b_remove])
                
                # Find candidate blocks that cover all unique subsets
                best_j = -1
                best_metric = float('inf')
                
                for j in range(n_k):
                    if j in sol_set:
                        continue
                    subs_j = subsets_of_block[j]
                    
                    # Check if j covers all unique subsets of b_remove
                    covers = True
                    for s in uniq_set:
                        if s not in subs_j:
                            covers = False
                            break
                    if not covers:
                        continue
                    
                    # Metric: prefer blocks with high overlap (already covered)
                    # and low number of new subsets
                    overlap = sum(1 for s in subs_j if cc[s] > 0)
                    new_subs = len(subs_j) - overlap
                    metric = new_subs  # minimize new subsets introduced
                    
                    if metric < best_metric:
                        best_metric = metric
                        best_j = j
                
                if best_j != -1:
                    sol[i] = best_j
                    sol_set.discard(b_remove)
                    sol_set.add(best_j)
                    cc[subsets_of_block[b_remove]] -= 1
                    cc[subsets_of_block[best_j]] += 1
                    improved = True
                    unique = get_unique_subs(sol, cc)
                    break
        
        return sol

    # 5. Perturbation: heavy-tailed random flips
    def perturb(sol, intensity):
        """Randomly flip a subset of block selections."""
        sol = list(sol)
        n_flip = max(1, int(intensity * len(sol)))
        
        flipped = set()
        attempts = 0
        while len(flipped) < n_flip and attempts < 20:
            idx = rng.integers(0, len(sol))
            if idx not in flipped:
                flipped.add(idx)
            attempts += 1
        
        for idx in flipped:
            # Replace with random block
            new_j = rng.integers(0, n_k)
            if new_j not in sol:
                sol[idx] = new_j
        
        # Ensure no duplicates
        seen = set()
        unique_sol = []
        for b in sol:
            if b not in seen:
                seen.add(b)
                unique_sol.append(b)
        
        return unique_sol

    # 6. ILS main loop
    current_sol = list(best_sol)
    current_size = best_size
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.1
    cooling = 0.995
    
    T = T_start
    no_improve_count = 0
    
    while time.time() < deadline - 10:
        # Local search
        ls_sol = local_search(current_sol)
        
        if len(ls_sol) < current_size:
            current_sol = ls_sol
            current_size = len(ls_sol)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Update best
        if current_size < best_size:
            best_size = current_size
            best_sol = list(current_sol)
        
        # Check time
        if time.time() > deadline - 10:
            break
        
        # Perturbation with heavy-tailed intensity
        # Use Pareto-like distribution for intensity
        u = rng.random()
        intensity = 0.3 / (1 + u * 2)  # ranges from ~0.1 to 0.3
        
        perturbed = perturb(current_sol, intensity)
        
        # Local search on perturbed
        pert_ls = local_search(perturbed)
        
        # Acceptance criterion (simulated annealing)
        if len(pert_ls) < current_size:
            current_sol = pert_ls
            current_size = len(pert_ls)
            no_improve_count = 0
        else:
            delta = len(pert_ls) - current_size
            if delta > 0:
                accept_prob = np.exp(-delta / max(T, 1e-6))
                if rng.random() < accept_prob:
                    current_sol = pert_ls
                    current_size = len(pert_ls)
        
        # Update best
        if current_size < best_size:
            best_size = current_size
            best_sol = list(current_sol)
        
        # Cool down
        T = max(T * cooling, T_end)
        
        # If stuck for too long, restart from best
        if no_improve_count > 50:
            current_sol = list(best_sol)
            current_size = best_size
            no_improve_count = 0
            T = T_start * 0.5  # Partial restart

    # 7. Final local search on best
    if time.time() < deadline - 3:
        final_ls = local_search(best_sol, max_iters=1000)
        if len(final_ls) < best_size:
            best_sol = final_ls
            best_size = len(best_sol)

    # 8. Final cleanup: remove duplicates
    seen = set()
    unique_sol = []
    for b in best_sol:
        if b not in seen:
            seen.add(b)
            unique_sol.append(b)
    best_sol = unique_sol

    # Verify validity
    cc = compute_cover_count(best_sol)
    if np.any(cc == 0):
        # Add missing blocks
        for s in np.where(cc == 0)[0]:
            for j in blocks_of_subset[s]:
                if j not in best_sol:
                    best_sol.append(j)
                    break
        # Re-deduplicate
        seen = set()
        unique_sol = []
        for b in best_sol:
            if b not in seen:
                seen.add(b)
                unique_sol.append(b)
        best_sol = unique_sol

    return [list(k_subsets[idx]) for idx in best_sol]
