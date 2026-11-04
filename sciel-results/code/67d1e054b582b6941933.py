"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute for fast delta
    # We'll work with p as a numpy array of locations for each facility
    
    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def eigen_init():
        """Initialize using leading eigenvectors of a and b."""
        # Compute leading eigenvectors
        try:
            eigvals_a, eigvecs_a = np.linalg.eigh(a)
            eigvals_b, eigvecs_b = np.linalg.eigh(b)
            
            # Take the top k eigenvectors (largest eigenvalues)
            # For QAP, we want to align the structures
            k = min(10, n)
            
            # Sort by eigenvalue descending
            idx_a = np.argsort(eigvals_a)[::-1][:k]
            idx_b = np.argsort(eigvals_b)[::-1][:k]
            
            U_a = eigvecs_a[:, idx_a]
            U_b = eigvecs_b[:, idx_b]
            
            # Project facilities and locations into latent space
            # Facility i is represented by U_a[i, :]
            # Location l is represented by U_b[l, :]
            
            # We want to assign facility i to location p[i] such that 
            # the alignment is good.
            # A simple heuristic: assign i to the location l that maximizes
            # the dot product of their embeddings, or minimizes distance.
            # But this is O(n^2 * k) which is fine for n=256.
            
            # Actually, let's use the linear assignment on the embedding distance
            # dist(i, l) = || U_a[i] - U_b[l] ||^2
            # = ||U_a[i]||^2 + ||U_b[l]||^2 - 2 U_a[i] @ U_b[l].T
            
            # This is equivalent to maximizing U_a[i] @ U_b[l].T
            # So we can solve the assignment problem on the matrix -U_a @ U_b.T
            
            M = -U_a @ U_b.T
            row_ind, col_ind = linear_sum_assignment(M)
            
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
            
        except Exception:
            # Fallback to random
            p = np.random.permutation(n)
            return p

    # --- Phase 1: Initial Solution via Eigen Init ---
    p = eigen_init()
    
    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 10.0)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Iterated Local Search (ILS) ---
    # Use remaining time for ILS with Hungarian re-optimization
    
    # Allocate time: 
    # - 10s for initial polish (done)
    # - Remaining ~105s for ILS
    
    # Strategy:
    # 1. Start from best solution
    # 2. Perturb (random swaps or Hungarian reassignment)
    # 3. Polish
    # 4. Accept if better
    # 5. Repeat until time runs out
    
    # We'll use a mix of:
    # a) Random perturbations
    # b) Hungarian reassignment on linearized cost
    # c) Tabu-like diversification
    
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 10
    
    # Temperature for simulated annealing-like acceptance
    # We'll use a simple rule: accept if better, or with small probability if worse
    
    while time.time() < deadline - 5.0:
        # Choose perturbation strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Strategy 1: Random swaps
            p = best_p.copy()
            num_swaps = random.randint(3, 15)
            for _ in range(num_swaps):
                i_s = random.randint(0, n - 1)
                j_s = random.randint(0, n - 1)
                if i_s != j_s:
                    p[i_s], p[j_s] = p[j_s], p[i_s]
                    
        elif strategy < 0.6:
            # Strategy 2: Hungarian reassignment
            p = best_p.copy()
            cost_mat = linearized_cost_matrix(p)
            
            # Add noise to escape local minima
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + np.random.randn(n, n) * noise_scale
            
            try:
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                new_p = np.empty(n, dtype=np.int64)
                new_p[row_ind] = col_ind
                p = new_p
            except Exception:
                p = best_p.copy()
                
        else:
            # Strategy 3: Large random perturbation + eigen re-init blend
            p = best_p.copy()
            # Swap a random subset
            num_swaps = random.randint(10, 30)
            for _ in range(num_swaps):
                i_s = random.randint(0, n - 1)
                j_s = random.randint(0, n - 1)
                if i_s != j_s:
                    p[i_s], p[j_s] = p[j_s], p[i_s]
            
            # Blend with eigen init
            p_eig = eigen_init()
            blend_ratio = random.uniform(0.3, 0.7)
            for i in range(n):
                if random.random() < blend_ratio:
                    p[i] = p_eig[i]
            
            # Ensure it's a valid permutation (fix duplicates if any)
            # This is a simple fix: if there are duplicates, replace with missing
            seen = set(p.tolist())
            missing = [x for x in range(n) if x not in seen]
            for i in range(n):
                if p[i] not in seen or seen.count(p[i]) > 1:
                    # This is a rough fix; in practice, the above should be rare
                    # We'll just re-randomize this position
                    p[i] = random.choice(missing)
                    if missing:
                        missing.remove(p[i])
        
        # Polish the perturbed solution
        polish_end = min(deadline - 2.0, time.time() + 8.0)
        p, p_cost = two_opt_polish(p, max_time=polish_end)
        
        # Accept if better
        if p_cost < best_cost:
            best_p = p.copy()
            best_cost = p_cost
            no_improve_rounds = 0
            hungarian_count += 1
        else:
            no_improve_rounds += 1
            
            # If stuck for too long, try a more aggressive restart
            if no_improve_rounds >= max_no_improve:
                # Full restart from eigen init
                p = eigen_init()
                polish_end = min(deadline - 2.0, time.time() + 8.0)
                p, p_cost = two_opt_polish(p, max_time=polish_end)
                
                if p_cost < best_cost:
                    best_p = p.copy()
                    best_cost = p_cost
                    no_improve_rounds = 0
                else:
                    no_improve_rounds = 0  # Reset to try again
    
    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
