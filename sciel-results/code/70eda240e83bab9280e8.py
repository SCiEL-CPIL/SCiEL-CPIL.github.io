"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    # 1. Initialization: Rank-based heuristic
    # This is a common strong baseline for QAP.
    # Sort facilities by row sums of A (flow) and locations by row sums of B (distance).
    # Match the most active facility to the most central location.
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # We want to minimize cost. High flow should go to low distance (central).
    # Sort a_sum descending (heavy first), b_sum ascending (central first).
    order_a = np.argsort(-a_sum, kind="stable")
    order_b = np.argsort(b_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    # Facility at rank k in A gets Location at rank k in B
    p[order_a] = order_b

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search: Fast 2-opt swaps
    # We use a vectorized delta calculation to speed up the search significantly.
    # The cost of swapping i and j is:
    # delta = (a[i] - a[j]) @ (B[p[j]] - B[p[i]])
    # where B is the distance matrix and p is the permutation.
    # We precompute B_rows for the current permutation to speed up dot products?
    # Actually, B[p[j]] is just row p[j] of B.
    # So delta = (a[i] - a[j]) @ (b[p[j]] - b[p[i]])
    
    # To make this faster, we can precompute a matrix M where M[i, k] = a[i] @ b[k]?
    # That would be A @ B. Let's call it AB.
    # AB[i, k] = sum_m a[i, m] b[k, m]
    # Then delta = AB[i, p[j]] - AB[i, p[i]] - AB[j, p[j]] + AB[j, p[i]]
    
    AB = a @ b
    
    # Now the delta calculation is just 4 lookups and additions.
    # This is O(1) per pair evaluation.
    # We can vectorize the inner loop over j for a fixed i.
    
    # Let's iterate i from 0 to n-1.
    # For a fixed i, we want to find j > i that minimizes delta.
    # delta(i, j) = AB[i, p[j]] - AB[i, p[i]] - AB[j, p[j]] + AB[j, p[i]]
    # Let C_i = AB[i, :]
    # Let C_j = AB[j, :]
    # delta = C_i[p[j]] - C_i[p[i]] - C_j[p[j]] + C_j[p[i]]
    
    # We can precompute for the current p:
    # Term 1: C_i[p[j]] for all j. This is C_i @ p? No, p is indices.
    # C_i[p] is a vector where element j is C_i[p[j]].
    # Term 2: -C_i[p[i]] is a constant for fixed i.
    # Term 3: -C_j[p[j]] for all j. This is the diagonal of AB @ P?
    # Let D[j] = AB[j, p[j]].
    # Term 4: C_j[p[i]] for all j. This is AB[:, p[i]] (column p[i] of AB).
    
    # So for fixed i:
    # const1 = -AB[i, p[i]]
    # vec1 = AB[i, p]  # vector of length n, element j is AB[i, p[j]]
    # vec2 = -D        # vector of length n, element j is -AB[j, p[j]]
    # vec3 = AB[:, p[i]] # vector of length n, element j is AB[j, p[i]]
    
    # total_delta[j] = vec1[j] + const1 + vec2[j] + vec3[j]
    
    # We can compute this vector for all j in O(n) operations (vectorized).
    # Then find min j > i.
    
    # Precompute D for current p.
    # D[j] = AB[j, p[j]]
    # We can update D incrementally when swapping.
    
    D = np.array([AB[j, p[j]] for j in range(n)])
    
    # Main loop
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Loop over i
        for i in range(n):
            if time.time() > deadline:
                break
                
            # Compute delta vector for all j
            # vec1 = AB[i, p]
            vec1 = AB[i, p]
            const1 = -AB[i, p[i]]
            vec2 = -D
            vec3 = AB[:, p[i]]
            
            deltas = vec1 + const1 + vec2 + vec3
            
            # We only consider j > i to avoid double counting and self-swaps
            # Also j != i
            # Set deltas for j <= i and j == i to inf
            # Note: p[i] is the location of i. Swapping i with i is delta 0.
            # We want min delta < 0 ideally.
            
            # Mask out invalid j
            # j must be > i
            # Also we shouldn't swap i with itself, but j>i handles that.
            
            # To speed up, we can just find the global min and check if it's valid?
            # No, we must respect j > i to ensure we don't revisit (i, j) as (j, i).
            # Actually, if we do full 2-opt, we can just check all j != i.
            # But standard 2-opt usually iterates i and j>i.
            
            # Let's just find the min in j > i.
            if i + 1 < n:
                valid_deltas = deltas[i+1:]
                min_val = np.min(valid_deltas)
                if min_val < best_delta:
                    best_delta = min_val
                    # Find index
                    # argmin gives index in valid_deltas, which is i+1 + idx
                    idx = np.argmin(valid_deltas)
                    best_j = i + 1 + idx
                    best_i = i
        
        if best_i == -1 or best_delta >= 0:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturb
                # Random swap
                i1 = np.random.randint(n)
                i2 = np.random.randint(n)
                if i1 != i2:
                    p[i1], p[i2] = p[i2], p[i1]
                    # Update D
                    D[i1] = AB[i1, p[i1]]
                    D[i2] = AB[i2, p[i2]]
                    current_cost = calc_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    no_improve = 0
                continue
            else:
                # Try to accept a non-improving move? No, let's just stop or restart.
                # For simplicity, let's do a random restart if stuck.
                if no_improve > 100:
                    # Re-initialize
                    order_a = np.argsort(-a_sum + np.random.randn(n)*0.1, kind="stable")
                    order_b = np.argsort(b_sum + np.random.randn(n)*0.1, kind="stable")
                    p = np.empty(n, dtype=np.int64)
                    p[order_a] = order_b
                    D = np.array([AB[j, p[j]] for j in range(n)])
                    current_cost = calc_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    no_improve = 0
                else:
                    # Small perturbation
                    i1 = np.random.randint(n)
                    i2 = np.random.randint(n)
                    if i1 != i2:
                        p[i1], p[i2] = p[i2], p[i1]
                        D[i1] = AB[i1, p[i1]]
                        D[i2] = AB[i2, p[i2]]
                        current_cost = calc_cost(p)
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        no_improve = 0
                continue
        
        # Perform swap
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        
        # Update D
        D[i] = AB[i, p[i]]
        D[j] = AB[j, p[j]]
        
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
