import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute for faster delta calculation? 
    # Delta for swapping i, j:
    # ext_delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[y, p[k]] - B[x, p[k]])
    # where x = p[i], y = p[j].
    # This is O(n).
    
    def get_cost(p):
        # p is a numpy array of locations for facilities 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # This is trace(a @ b[p][:,p]) ? No.
        # b_perm = b[np.ix_(p, p)]
        # return np.sum(a * b_perm)
        # Using einsum is faster
        return np.einsum('ij,ik,jk->', a, b[np.ix_(p, p)])
        # Wait, b[np.ix_(p,p)] creates a new array. 
        # Better: sum_{i,j} a[i,j] b[p[i], p[j]]
        # This is equivalent to sum_{i} sum_{j} a[i,j] b[p[i], p[j]]
        
    def calc_delta(i, j, p):
        """
        Calculate cost change if we swap facilities at positions i and j.
        p is the current permutation (location of facility k is p[k]).
        """
        x = p[i]
        y = p[j]
        
        # Vectorized calculation of external delta
        # delta_k = (A[i,k] - A[j,k]) * (B[y, p[k]] - B[x, p[k]])
        # Sum over k != i, j
        
        # Create mask for k != i, j
        # To avoid creating large temporary arrays, we can compute sum over all k and subtract i and j terms.
        
        # Term for k: (A[i,k] - A[j,k]) * (B[y, p[k]] - B[x, p[k]])
        # Let's compute the full sum first.
        
        # B_y_p = B[y, p]
        # B_x_p = B[x, p]
        # diff_B = B_y_p - B_x_p
        # diff_A = A[i, :] - A[j, :]
        # total_sum = np.sum(diff_A * diff_B)
        
        # Subtract contributions from k=i and k=j
        # k=i: (A[i,i] - A[j,i]) * (B[y, x] - B[x, x])
        # k=j: (A[i,j] - A[j,j]) * (B[y, y] - B[x, y])
        
        # Internal term change:
        # Original: A[i,j]B[x,y] + A[j,i]B[y,x]
        # New: A[i,j]B[y,x] + A[j,i]B[x,y]
        # Change: A[i,j](B[y,x]-B[x,y]) + A[j,i](B[x,y]-B[y,x])
        
        # If B is symmetric, B[x,y] == B[y,x], so internal change is 0.
        # We assume B is symmetric for QAPLIB.
        
        # Let's use the vectorized full sum and subtract i,j.
        diff_A = a[i] - a[j]
        # B[y, p] and B[x, p]
        # p is an array of indices.
        diff_B = b[y, p] - b[x, p]
        
        total_ext = np.dot(diff_A, diff_B)
        
        # Subtract k=i
        # k=i term in dot product: diff_A[i] * diff_B[i]
        # diff_A[i] = A[i,i] - A[j,i]
        # diff_B[i] = B[y, p[i]] - B[x, p[i]] = B[y, x] - B[x, x]
        sub_i = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        
        # Subtract k=j
        # diff_A[j] = A[i,j] - A[j,j]
        # diff_B[j] = B[y, p[j]] - B[x, p[j]] = B[y, y] - B[x, y]
        sub_j = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        
        ext_delta = total_ext - sub_i - sub_j
        
        # Internal delta
        # If B is symmetric, it's 0. Let's check symmetry once.
        # For safety, compute it.
        int_delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
        
        return ext_delta + int_delta

    def local_search(p, deadline):
        """
        Perform 2-opt local search on permutation p.
        Returns improved permutation and cost.
        """
        p = p.copy()
        improved = True
        last_improvement_time = time.time()
        
        while improved:
            improved = False
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # Find best improving move
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # We can iterate through all pairs. n=100, 4950 pairs.
            # For each pair, calc_delta is O(n) = 100 ops. Total ~500k ops per pass.
            # This is fast.
            
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    delta = calc_delta(i, j, p)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        
            if best_i != -1:
                # Perform swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                improved = True
                last_improvement_time = time.time()
                
        return p

    # Initialize best solution
    # Use a greedy initialization: sort by row sums
    a_sums = np.sum(a, axis=1)
    b_sums = np.sum(b, axis=1)
    
    # Heavy facilities to light locations
    heavy_order = np.argsort(-a_sums, kind='stable')
    light_order = np.argsort(b_sums, kind='stable')
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = light_order
    
    p_best = p_init.copy()
    cost_best = get_cost(p_best)
    
    # Iterated Local Search
    while time.time() < deadline - 10.0:
        # Perturb p_best
        p_curr = p_best.copy()
        
        # Randomly swap 2-5 pairs
        num_swaps = np.random.randint(2, 6)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
            
        # Local search
        p_new = local_search(p_curr, deadline)
        cost_new = get_cost(p_new)
        
        if cost_new < cost_best:
            p_best = p_new
            cost_best = cost_new
            
    return p_best.tolist()
