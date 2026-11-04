"""QAP tai256c solver: iterated local search with spectral bisection init, Hungarian re-optimization, and tabu diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    # Precompute symmetric A for faster delta calculations
    A_sym = a + a.T

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_local_search(p, cost, max_passes=50):
        """
        Optimized local search using precomputed D matrix.
        D[i, j] = b[p[i], p[j]]
        """
        p = p.copy()
        D = b[np.ix_(p, p)].copy()
        current_cost = cost

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(n - 1):
                if time.time() > deadline:
                    return p, current_cost

                js = np.arange(i + 1, n)

                # Simplified delta: delta(i,j) = sum_k A_sym[i,k]*(D[j,k]-D[i,k]) 
                # + sum_k A_sym[j,k]*(D[i,k]-D[j,k]) - A_sym[i,j]*(D[i,j]+D[j,i]-D[i,i]-D[j,j])
                # = sum_k (A_sym[i,k] - A_sym[j,k])*(D[j,k]-D[i,k]) - A_sym[i,j]*(D[i,j]+D[j,i]-D[i,i]-D[j,j])
                
                # Vectorized: for each j in js, compute delta
                # delta_j = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #         - A_sym[i,j] * (D[i,j] + D[j,i] - D[i,i] - D[j,j])
                
                diff_A = A_sym[i][None, :] - A_sym[js]  # shape (len(js), n)
                diff_D = D[js] - D[i][None, :]  # shape (len(js), n)
                
                S = np.sum(diff_A * diff_D, axis=1)
                
                # Correction: -A_sym[i, js] * (D[i, js] + D[js, i] - D[i, i] - D[js, js])
                correction = -A_sym[i, js] * (D[i, js] + D[js, i] - D[i, i] - np.diag(D[js]))
                
                deltas = S + correction

                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]

                if min_delta < 0:
                    j = js[min_idx]
                    p[i], p[j] = p[j], p[i]

                    # Update D in-place (swap rows and columns)
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()

                    current_cost += int(min_delta)
                    improved = True

        return p, current_cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def multi_kick_diversify(p, cost, n_kicks=3):
        """Apply multiple segment reversals of varying lengths for stronger diversification."""
        best_p = p.copy()
        best_cost = cost
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            
            # Random segment reversal of length 2-15
            seg_len = np.random.randint(2, 16)
            if seg_len >= n:
                seg_len = n - 1
            start = np.random.randint(0, n - seg_len)
            end = start + seg_len
            
            p[start:end+1] = p[start:end+1][::-1]
            new_cost = compute_cost(p)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
            else:
                # Revert
                p[start:end+1] = p[start:end+1][::-1]
        
        return best_p, best_cost

    def fiedler_partition(matrix, size):
        """Find Fiedler vector of a graph Laplacian to bisect nodes."""
        D = np.diag(matrix.sum(axis=1))
        L = D - matrix
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(size)
        
        return fiedler > 0

    def recursive_assign(facilities, locations, depth=0):
        """Recursively bisect and assign."""
        if len(facilities) <= 1:
            if len(facilities) == 1 and len(locations) == 1:
                return {facilities[0]: locations[0]}
            return {}
        
        if len(facilities) <= 2:
            result = {}
            for i, f in enumerate(facilities):
                result[f] = locations[i]
            return result
        
        sub_a = a[np.ix_(facilities, facilities)]
        fac_partition = fiedler_partition(sub_a, len(facilities))
        
        sub_b = b[np.ix_(locations, locations)]
        loc_partition = fiedler_partition(-sub_b, len(locations))
        
        fac_group1 = [facilities[i] for i in range(len(facilities)) if fac_partition[i]]
        fac_group2 = [facilities[i] for i in range(len(facilities)) if not fac_partition[i]]
        loc_group1 = [locations[i] for i in range(len(locations)) if loc_partition[i]]
        loc_group2 = [locations[i] for i in range(len(locations)) if not loc_partition[i]]
        
        if len(fac_group1) > len(loc_group1) and len(fac_group2) > 0:
            diff = len(fac_group1) - len(loc_group1)
            for _ in range(min(diff, len(fac_group1))):
                fac_group2.append(fac_group1.pop())
        elif len(fac_group1) < len(loc_group1) and len(fac_group2) > 0:
            diff = len(loc_group1) - len(fac_group1)
            for _ in range(min(diff, len(fac_group2))):
                fac_group1.append(fac_group2.pop())
        
        if len(fac_group1) == 0:
            fac_group1 = [facilities[0]]
            fac_group2 = facilities[1:]
        if len(fac_group2) == 0:
            fac_group2 = [facilities[-1]]
            fac_group1 = facilities[:-1]
        if len(loc_group1) == 0:
            loc_group1 = [locations[0]]
            loc_group2 = locations[1:]
        if len(loc_group2) == 0:
            loc_group2 = [locations[-1]]
            loc_group1 = locations[:-1]
        
        result = {}
        result.update(recursive_assign(fac_group1, loc_group1, depth + 1))
        result.update(recursive_assign(fac_group2, loc_group2, depth + 1))
        
        return result

    def spectral_bisection_init():
        """Generate initial permutation using spectral bisection approach."""
        all_facilities = list(range(n))
        all_locations = list(range(n))
        
        assignment = recursive_assign(all_facilities, all_locations)
        
        p = np.empty(n, dtype=np.int64)
        for fac, loc in assignment.items():
            p[fac] = loc
        
        assigned_locs = set(assignment.values())
        unassigned_facs = [f for f in range(n) if f not in assignment]
        unassigned_locs = [l for l in range(n) if l not in assigned_locs]
        for f, l in zip(unassigned_facs, unassigned_locs):
            p[f] = l
        
        return p

    def rank_matching_init():
        """Fallback rank-matching initialization."""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply block-reversal kicks with tabu for stronger diversification."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 8
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            
            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            key = (i, j)
            
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            p[i:j+1] = p[i:j+1][::-1]
            new_cost = compute_cost(p)
            
            if not is_tabu or (new_cost < best_cost):
                cost = new_cost
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                p[i:j+1] = p[i:j+1][::-1]
            
            iteration += 1
        
        return best_p, best_cost

    # Initial solution 1: spectral bisection for a globally informed start
    p = spectral_bisection_init()
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = fast_local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Also try rank-matching as a second initial solution
    p2 = rank_matching_init()
    p2, cost2 = fast_local_search(p2.copy(), compute_cost(p2))
    if cost2 < best_cost:
        best_p = p2.copy()
        best_cost = cost2

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    stuck_count = 0
    while time.time() < deadline - 15:
        iteration += 1

        p = best_p.copy()
        
        # Adaptive perturbation: stronger when stuck
        if stuck_count > 5:
            block_len = np.random.randint(5, 15)
        elif stuck_count > 2:
            block_len = np.random.randint(4, 12)
        else:
            block_len = np.random.randint(3, 10)
        
        start = np.random.randint(0, n - block_len)
        p[start:start+block_len] = p[start:start+block_len][::-1]
        
        # Use multi-kick diversification instead of tabu_kick for better exploration
        p, cost = multi_kick_diversify(p, compute_cost(p), n_kicks=2)

        p, cost = fast_local_search(p, cost)

        # Hungarian re-optimization: more frequent when stuck
        h_interval = max(2, 8 - stuck_count // 3)
        if iteration % h_interval == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = fast_local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            stuck_count = 0
        else:
            stuck_count += 1

    # Final high-precision polish: try multiple small perturbations before final LS
    if time.time() < deadline - 5:
        final_p = best_p.copy()
        final_cost = best_cost
        
        # Try a few small random perturbations and take the best after LS
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            cand_p = final_p.copy()
            # Small perturbation: swap 2-3 random pairs
            n_swaps = np.random.randint(1, 4)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                cand_p[i], cand_p[j] = cand_p[j], cand_p[i]
            cand_cost = compute_cost(cand_p)
            cand_p, cand_cost = fast_local_search(cand_p, cand_cost, max_passes=20)
            if cand_cost < final_cost:
                final_p = cand_p.copy()
                final_cost = cand_cost
        
        # Final polish on the best candidate
        final_p, final_cost = fast_local_search(final_p, final_cost, max_passes=30)
        if final_cost < best_cost:
            best_p = final_p.copy()
            best_cost = final_cost
    
    return best_p.tolist()
