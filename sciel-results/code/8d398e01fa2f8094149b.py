"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute for fast cost evaluation
    # cost(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can write this as sum_i (A_i . B_{p[i]} where A_i is row i of a weighted by... 
    # Actually: sum_{i,j} a[i,j] b[p[i],p[j]] = sum_i sum_j a[i,j] b[p[i],p[j]]
    # Let's define for each facility i, the contribution when placed at location k is:
    # sum_j a[i,j] b[k, p[j]] ... this is coupled.
    #
    # Better: use the standard QAP cost computation.
    # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
    # = sum_i (sum_j a[i,j] b[p[i],p[j]])
    # For a given p, we can compute: for each i, let loc_i = p[i]. Then
    # inner_i = sum_j a[i,j] b[loc_i, p[j]]
    # This is O(n^2) per evaluation if done naively, but we can precompute.
    #
    # For swap-based local search, we need efficient delta computation.
    #
    # Let's use the approach: precompute row sums of a, and for each pair of locations
    # we can compute the change when swapping two facilities.
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    # Compute initial cost
    def compute_cost(p_list):
        p_arr = np.array(p_list, dtype=np.int64)
        total = 0
        for i in range(n):
            for j in range(n):
                total += a[i, j] * b[p_arr[i], p_arr[j]]
        return total
    
    # Faster cost computation using numpy
    def compute_cost_np(p_arr):
        # cost = sum_{i,j} a[i,j] b[p[i],p[j]]
        # = sum_i sum_j a[i,j] b[p[i],p[j]]
        # We can compute: for each i, b[p[i], p[0:n]] is a row of b indexed by p
        # Then multiply by a[i,:] and sum
        bp = b[p_arr]  # shape (n, n): bp[i, j] = b[p[i], p[j]]
        return int(np.sum(a * bp))
    
    p_arr = np.array(p, dtype=np.int64)
    best_cost = compute_cost_np(p_arr)
    best_p = p_arr.copy()
    
    # Precompute for efficient swap delta
    # When we swap facilities i and j (i.e., swap p[i] and p[j]), the change in cost is:
    # delta = sum_{k != i, k != j} a[i,k] * (b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k] * (b[p[i],p[k]] - b[p[j],p[k]])
    #       + a[i,j] * (b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i] * (b[p[i],p[j]] - b[p[j],p[i]])
    #       + a[i,i] * (b[p[j],p[j]] - b[p[i],p[i]]) + a[j,j] * (b[p[i],p[i]] - b[p[j],p[j]])
    #
    # More carefully:
    # Let old_i = p[i], old_j = p[j]. After swap: new_i = old_j, new_j = old_i.
    # The terms involving i or j change.
    # Terms with neither i nor j: unchanged.
    # Terms (i,k) for k != i,j: a[i,k] * b[new_i, p[k]] - a[i,k] * b[old_i, p[k]] = a[i,k] * (b[old_j, p[k]] - b[old_i, p[k]])
    # Terms (j,k) for k != i,j: a[j,k] * (b[old_i, p[k]] - b[old_j, p[k]])
    # Terms (i,i): a[i,i] * (b[old_j, old_j] - b[old_i, old_i])
    # Terms (j,j): a[j,j] * (b[old_i, old_i] - b[old_j, old_j])
    # Terms (i,j): a[i,j] * (b[old_j, old_i] - b[old_i, old_j])
    # Terms (j,i): a[j,i] * (b[old_i, old_j] - b[old_j, old_i])
    
    # Precompute: for each location k, b[k, :] is available.
    # For efficiency, precompute for each pair of locations (u, v), the vector b[u, :] - b[v, :]
    # But that's O(n^2 * n) = O(n^3) memory which is 150^3 = 3.375M * 8 bytes = ~27MB, acceptable.
    
    # Actually, let's compute the delta on the fly but vectorized.
    # For a swap of i and j:
    # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[old_j, p[k]] - b[old_i, p[k]])
    #         + a[i,i] * (b[old_j, old_j] - b[old_i, old_i])
    #         + a[j,j] * (b[old_i, old_i] - b[old_j, old_j])
    #         + (a[i,j] - a[j,i]) * (b[old_j, old_i] - b[old_i, old_j])
    
    # Let's define:
    # diff_b = b[old_j] - b[old_i]  (shape n)
    # Then sum_{k != i, j} (a[i,k] - a[j,k]) * diff_b[p[k]]
    # = sum_k (a[i,k] - a[j,k]) * diff_b[p[k]] - (a[i,i] - a[j,i]) * diff_b[old_i] - (a[i,j] - a[j,j]) * diff_b[old_j]
    
    # Let's precompute a_diff[i,j] = a[i,:] - a[j,:] for all i,j? That's O(n^2 * n) = O(n^3).
    # 150^3 = 3.375M elements, 8 bytes each = ~27MB. OK.
    
    a_diff = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            a_diff[i, j] = a[i] - a[j]
    
    # Now local search: try all swaps, accept best improving move (steepest descent)
    # To speed up, we can use a random order or first-improvement with random restarts.
    
    # Let's use a simulated-annealing-like approach or just best-improvement with time limit.
    
    # For n=150, there are C(150,2) = 11175 possible swaps. Each swap delta computation
    # can be done in O(n) if we precompute things well.
    # 11175 * 150 = ~1.68M operations per full pass. That's fast.
    
    # Let's do best-improvement local search with a time limit.
    
    def swap_delta(p_arr, i, j):
        old_i = p_arr[i]
        old_j = p_arr[j]
        if old_i == old_j:
            return 0
        diff_b = b[old_j] - b[old_i]  # shape (n,)
        # sum_k (a[i,k] - a[j,k]) * diff_b[p_arr[k]]
        # = a_diff[i,j] @ diff_b[p_arr]
        # But we need to exclude k=i and k=j from the sum, then add back the special terms.
        # Actually, the formula above already handles it if we just compute the full sum and adjust.
        
        # Full sum: S = sum_k (a[i,k] - a[j,k]) * diff_b[p_arr[k]]
        # This includes k=i and k=j.
        # For k=i: term is (a[i,i] - a[j,i]) * diff_b[old_i] = (a[i,i] - a[j,i]) * (b[old_j, old_i] - b[old_i, old_i])
        # For k=j: term is (a[i,j] - a[j,j]) * diff_b[old_j] = (a[i,j] - a[j,j]) * (b[old_j, old_j] - b[old_i, old_j])
        
        # The actual delta is:
        # sum_{k != i,j} (a[i,k] - a[j,k]) * (b[old_j, p[k]] - b[old_i, p[k]])
        # + a[i,i] * (b[old_j, old_j] - b[old_i, old_i])
        # + a[j,j] * (b[old_i, old_i] - b[old_j, old_j])
        # + (a[i,j] - a[j,i]) * (b[old_j, old_i] - b[old_i, old_j])
        
        # Let me re-derive more carefully.
        # Original cost contribution from terms involving i or j:
        # C_orig = sum_{k} a[i,k] b[old_i, p[k]] + sum_{k} a[j,k] b[old_j, p[k]]
        #         - a[i,i] b[old_i, old_i] - a[j,j] b[old_j, old_j]  (to avoid double counting diagonal? No, we don't subtract)
        # Actually, the full cost is sum_{i',j'} a[i',j'] b[p[i'],p[j']].
        # Terms where i'=i: sum_{j'} a[i,j'] b[old_i, p[j']]
        # Terms where j'=i: sum_{i'} a[i',i] b[p[i'], old_i]  (but these are already included if i' != i)
        # This is getting confusing. Let me just compute the delta directly.
        
        # Delta = C_new - C_orig
        # C_new - C_orig = sum_{i',j'} a[i',j'] (b[p_new[i'], p_new[j']] - b[p[i'], p[j']])
        # Only p_new[i] != p[i] and p_new[j] != p[j], all others same.
        # So the difference is:
        # sum_{j'} a[i,j'] (b[old_j, p_new[j']] - b[old_i, p[j']])
        # + sum_{i'} a[i',j] (b[p_new[i'], old_i] - b[p[i'], old_j])
        # - a[i,j] b[old_i, old_j] - a[j,i] b[old_j, old_i]  (to correct for double counting the (i,j) and (j,i) terms)
        # Wait, when i'=i and j'=j, both sums include the term a[i,j] b[old_j, old_i] in the first and a[i,j] b[old_i, old_j] in the second? No.
        
        # Let me be very explicit.
        # In C_orig, the term (i,j) is a[i,j] b[old_i, old_j].
        # In C_new, the term (i,j) is a[i,j] b[old_j, old_i].
        # In C_orig, the term (j,i) is a[j,i] b[old_j, old_i].
        # In C_new, the term (j,i) is a[j,i] b[old_i, old_j].
        
        # For k not in {i,j}:
        # Term (i,k): a[i,k] b[old_i, p[k]] -> a[i,k] b[old_j, p[k]], change: a[i,k] (b[old_j, p[k]] - b[old_i, p[k]])
        # Term (k,i): a[k,i] b[p[k], old_i] -> a[k,i] b[p[k], old_j], change: a[k,i] (b[p[k], old_j] - b[p[k], old_i])
        # Term (j,k): a[j,k] b[old_j, p[k]] -> a[j,k] b[old_i, p[k]], change: a[j,k] (b[old_i, p[k]] - b[old_j, p[k]])
        # Term (k,j): a[k,j] b[p[k], old_j] -> a[k,j] b[p[k], old_i], change: a[k,j] (b[p[k], old_i] - b[p[k], old_j])
        
        # For k = i:
        # Term (i,i): a[i,i] b[old_i, old_i] -> a[i,i] b[old_j, old_j], change: a[i,i] (b[old_j, old_j] - b[old_i, old_i])
        # For k = j:
        # Term (j,j): a[j,j] b[old_j, old_j] -> a[j,j] b[old_i, old_i], change: a[j,j] (b[old_i, old_i] - b[old_j, old_j])
        
        # Terms (i,j) and (j,i):
        # (i,j): change = a[i,j] (b[old_j, old_i] - b[old_i, old_j])
        # (j,i): change = a[j,i] (b[old_i, old_j] - b[old_j, old_i])
        
        # So total delta:
        # = sum_{k not in {i,j}} [a[i,k] (b[old_j, p[k]] - b[old_i, p[k]]) + a[k,i] (b[p[k], old_j] - b[p[k], old_i])
        #   + a[j,k] (b[old_i, p[k]] - b[old_j, p[k]]) + a[k,j] (b[p[k], old_i] - b[p[k], old_j])]
        # + a[i,i] (b[old_j, old_j] - b[old_i, old_i])
        # + a[j,j] (b[old_i, old_i] - b[old_j, old_j])
        # + (a[i,j] - a[j,i]) (b[old_j, old_i] - b[old_i, old_j])
        
        # Note: b[p[k], old_j] - b[p[k], old_i] = - (b[old_i, p[k]] - b[old_j, p[k]]) if b is symmetric.
        # But b may not be symmetric. Let's keep it general.
        
        # This is getting complex. Let me just compute it directly for each swap in O(n) time.
        
        old_i_loc = p_arr[i]
        old_j_loc = p_arr[j]
        
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p_arr[k]
            delta += a[i, k] * (b[old_j_loc, pk] - b[old_i_loc, pk])
            delta += a[k, i] * (b[pk, old_j_loc] - b[pk, old_i_loc])
            delta += a[j, k] * (b[old_i_loc, pk] - b[old_j_loc, pk])
            delta += a[k, j] * (b[pk, old_i_loc] - b[pk, old_j_loc])
        
        delta += a[i, i] * (b[old_j_loc, old_j_loc] - b[old_i_loc, old_i_loc])
        delta += a[j, j] * (b[old_i_loc, old_i_loc] - b[old_j_loc, old_j_loc])
        delta += (a[i, j] - a[j, i]) * (b[old_j_loc, old_i_loc] - b[old_i_loc, old_j_loc])
        
        return int(delta)
    
    # This O(n) per swap is too slow for 11175 swaps per pass (11175 * 150 = 1.68M, but in Python loop it's slow).
    # Let's vectorize the delta computation.
    
    def swap_delta_vec(p_arr, i, j):
        old_i_loc = p_arr[i]
        old_j_loc = p_arr[j]
        
        # diff_b[k] = b[old_j_loc, k] - b[old_i_loc, k] for the (i,k) and (j,k) terms
        # diff_bt[k] = b[k, old_j_loc] - b[k, old_i_loc] for the (k,i) and (k,j) terms
        diff_b = b[old_j_loc] - b[old_i_loc]  # shape (n,)
        diff_bt = b[:, old_j_loc] - b[:, old_i_loc]  # shape (n,)
        
        # sum_{k != i,j} a[i,k] * diff_b[p_arr[k]] + a[k,i] * diff_bt[p_arr[k]]
        #             + a[j,k] * (-diff_b[p_arr[k]]) + a[k,j] * (-diff_bt[p_arr[k]])
        # = sum_{k != i,j} (a[i,k] - a[j,k]) * diff_b[p_arr[k]] + (a[k,i] - a[k,j]) * diff_bt[p_arr[k]]
        
        # Let S1 = sum_k (a[i,k] - a[j,k]) * diff_b[p_arr[k]]
        #     S2 = sum_k (a[k,i] - a[k,j]) * diff_bt[p_arr[k]]
        # Then delta = S1 + S2 - (terms for k=i and k=j) + special terms
        
        # S1 = a_diff[i,j] @ diff_b[p_arr]  (using precomputed a_diff)
        # S2 = (a[:,i] - a[:,j]) @ diff_bt[p_arr]
        
        S1 = int(np.dot(a_diff[i, j], diff_b[p_arr]))
        S2 = int(np.dot(a[:, i] - a[:, j], diff_bt[p_arr]))
        
        # Subtract k=i and k=j contributions from S1 and S2
        # For k=i: S1 contribution = (a[i,i] - a[j,i]) * diff_b[old_i_loc]
        #          S2 contribution = (a[i,i] - a[i,j]) * diff_bt[old_i_loc]
        # For k=j: S1 contribution = (a[i,j] - a[j,j]) * diff_b[old_j_loc]
        #          S2 contribution = (a[j,i] - a[j,j]) * diff_bt[old_j_loc]
        
        S1_ii = (a[i, i] - a[j, i]) * diff_b[old_i_loc]
        S1_ij = (a[i, j] - a[j, j]) * diff_b[old_j_loc]
        S2_ii = (a[i, i] - a[i, j]) * diff_bt[old_i_loc]
        S2_ij = (a[j, i] - a[j, j]) * diff_bt[old_j_loc]
        
        delta = S1 + S2 - S1_ii - S1_ij - S2_ii - S2_ij
        
        # Add special terms for (i,i), (j,j), (i,j), (j,i)
        delta += a[i, i] * (b[old_j_loc, old_j_loc] - b[old_i_loc, old_i_loc])
        delta += a[j, j] * (b[old_i_loc, old_i_loc] - b[old_j_loc, old_j_loc])
        delta += (a[i, j] - a[j, i]) * (b[old_j_loc, old_i_loc] - b[old_i_loc, old_j_loc])
        
        return int(delta)
    
    # Let's verify this with a simple case... actually, let's just use it and hope.
    # To be safe, let's use the direct computation for the first few swaps and compare.
    
    # Local search: best-improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                d = swap_delta_vec(p_arr, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        if best_i >= 0:
            p_arr[best_i], p_arr[best_j] = p_arr[best_j], p_arr[best_i]
            best_cost += best_delta
            improved = True
    
    # If we have time, try random restarts or 2-opt-like moves
    if time.time() < deadline - 10:
        # Try a few random perturbations and re-optimize
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Random swap
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
            cur_cost = compute_cost_np(p_arr)
            # Local search from here
            improved2 = True
            while improved2 and time.time() < deadline - 5:
                improved2 = False
                best_delta = 0
                best_i, best_j = -1, -1
                for ii in range(n):
                    if time.time() > deadline - 5:
                        break
                    for jj in range(ii + 1, n):
                        d = swap_delta_vec(p_arr, ii, jj)
                        if d < best_delta:
                            best_delta = d
                            best_i, best_j = ii, jj
                if best_i >= 0:
                    p_arr[best_i], p_arr[best_j] = p_arr[best_j], p_arr[best_i]
                    cur_cost += best_delta
                    improved2 = True
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p_arr.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
