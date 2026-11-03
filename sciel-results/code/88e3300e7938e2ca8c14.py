"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.sum(diff_a * diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                
        return p, current_cost

    def hungarian_init(current_p):
        """Build linearized cost matrix and solve with Hungarian algorithm."""
        # Cost of assigning facility i to location j given current assignment
        # c[i,j] = sum_k a[i,k] * b[j, current_p[k]] + sum_k a[k,i] * b[current_p[k], j]
        # This approximates the marginal contribution of placing facility i at location j
        
        # For efficiency, compute:
        # term1[i,j] = sum_k a[i,k] * b[j, current_p[k]]
        # term2[i,j] = sum_k a[k,i] * b[current_p[k], j]
        
        # term1: for each i, we need sum over k of a[i,k] * b[j, current_p[k]]
        # = (a @ b[:, current_p])[i, j] -- wait, let me think more carefully
        # b[:, current_p] is an n x n matrix where column k is b[:, current_p[k]]
        # Actually b[j, current_p[k]] for fixed j,k
        # Let B_perm = b[np.ix_(current_p, current_p)] -- no that's not right either
        
        # Let me define: B_cur = b[:, current_p]  # shape (n, n), B_cur[j, k] = b[j, current_p[k]]
        B_cur = b[:, current_p]  # B_cur[j, k] = b[j, current_p[k]]
        
        # term1[i, j] = sum_k a[i, k] * B_cur[j, k] = (a @ B_cur.T)[i, j]
        term1 = a @ B_cur.T  # term1[i, j] = sum_k a[i,k] * b[j, current_p[k]]
        
        # term2[i, j] = sum_k a[k, i] * b[current_p[k], j]
        # b[current_p[k], j] for fixed k,j
        # Let B_cur2 = b[current_p, :]  # shape (n, n), B_cur2[k, j] = b[current_p[k], j]
        B_cur2 = b[current_p, :]  # B_cur2[k, j] = b[current_p[k], j]
        
        # term2[i, j] = sum_k a[k, i] * B_cur2[k, j] = (a.T @ B_cur2)[i, j]
        term2 = a.T @ B_cur2  # term2[i, j] = sum_k a[k,i] * b[current_p[k], j]
        
        cost_matrix = term1 + term2
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS / SA Loop with Hungarian-based perturbations
    p = best_p.copy()
    current_cost = best_cost
    
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() - start_time < timeout:
        # Check time before expensive operations
        if time.time() - start_time > timeout - 5.0:
            break
            
        # Perturbation strategy: alternate between random swaps and Hungarian-based
        use_hungarian = (no_improve_count > 5) or (np.random.random() < 0.3)
        
        if use_hungarian:
            # Use Hungarian algorithm to generate a new permutation
            new_p = hungarian_init(p)
            
            # Apply some random swaps to diversify
            num_swaps = 3
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                delta = get_delta(new_p, i, j)
                new_p[i], new_p[j] = new_p[j], new_p[i]
                # Recompute cost from scratch since we changed multiple positions
            current_cost = calc_cost(new_p)
            p = new_p
        else:
            # Random swaps perturbation
            num_swaps = 5
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for too long, reset to best and perturb more aggressively
        if no_improve_count >= max_no_improve:
            p = best_p.copy()
            current_cost = best_cost
            no_improve_count = 0
            
            # More aggressive perturbation
            num_swaps = 10
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta

    return best_p.tolist()
# EVOLVE-BLOCK-END
