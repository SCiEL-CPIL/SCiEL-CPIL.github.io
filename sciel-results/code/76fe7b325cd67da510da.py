# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Precompute for fast delta calculation
    # We will use a vectorized approach for the 2-swap delta if possible, 
    # but for n=150, a tight Python loop with NumPy dot products is often sufficient 
    # if we minimize overhead. 
    # However, to ensure speed for SA, we need very fast delta evaluation.
    
    # Let's implement a fast 2-swap delta function.
    # Delta(i, j) for swapping p[i] and p[j]
    # Let u = p[i], v = p[j]
    # The change in cost is:
    # 1. Terms involving i and j directly:
    #    Old: a[i,j] b[u,v] + a[j,i] b[v,u]
    #    New: a[i,j] b[v,u] + a[j,i] b[u,v]
    #    Diff1 = a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
    
    # 2. Terms involving i and k (k != i, j):
    #    Old: a[i,k] b[u, p[k]]
    #    New: a[i,k] b[v, p[k]]
    #    Diff2_i = sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    
    # 3. Terms involving k and i (k != i, j):
    #    Old: a[k,i] b[p[k], u]
    #    New: a[k,i] b[p[k], v]
    #    Diff3_i = sum_{k!=i,j} a[k,i] (b[p[k], v] - b[p[k], u])
    
    # 4. Terms involving j and k (k != i, j):
    #    Old: a[j,k] b[v, p[k]]
    #    New: a[j,k] b[u, p[k]]
    #    Diff2_j = sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
    
    # 5. Terms involving k and j (k != i, j):
    #    Old: a[k,j] b[p[k], v]
    #    New: a[k,j] b[p[k], u]
    #    Diff3_j = sum_{k!=i,j} a[k,j] (b[p[k], u] - b[p[k], v])
    
    # We can combine Diff2_i and Diff2_j:
    # Sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
    # And Diff3_i and Diff3_j:
    # Sum_{k!=i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
    
    # To make this fast, we can precompute row/col differences? No, i and j change.
    # But we can vectorize the sum over k.
    
    # Let's implement a helper that calculates delta for a specific pair (i, j) quickly.
    # Since n=150, O(n) per pair is acceptable if the constant factor is low.
    
    def calc_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        if u == v:
            return 0.0
            
        # Diff1
        d1 = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        # Prepare vectors for sums
        # We need sum over k != i, j
        # Let's compute sum over all k and subtract k=i, k=j terms
        
        # Vector A_diff_row = a[i, :] - a[j, :]
        # Vector A_diff_col = a[:, i] - a[:, j]
        
        # Vector B_diff_row = b[v, p] - b[u, p]  (Shape n)
        # Vector B_diff_col = b[p, v] - b[p, u]  (Shape n)
        
        # Sum1 = dot(A_diff_row, B_diff_row)
        # Sum2 = dot(A_diff_col, B_diff_col)
        
        # Subtract k=i and k=j contributions from Sum1 and Sum2
        
        A_diff_row = a[i, :] - a[j, :]
        A_diff_col = a[:, i] - a[:, j]
        
        B_diff_row = b[v, p] - b[u, p]
        B_diff_col = b[p, v] - b[p, u]
        
        Sum1_all = np.dot(A_diff_row, B_diff_row)
        Sum2_all = np.dot(A_diff_col, B_diff_col)
        
        # Subtract k=i
        # A_diff_row[i] = a[i,i] - a[j,i]
        # B_diff_row[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        sub1_i = A_diff_row[i] * B_diff_row[i]
        
        # A_diff_col[i] = a[i,i] - a[j,i]
        # B_diff_col[i] = b[p[i], v] - b[p[i], u] = b[u, v] - b[u, u]
        sub2_i = A_diff_col[i] * B_diff_col[i]
        
        # Subtract k=j
        # A_diff_row[j] = a[i,j] - a[j,j]
        # B_diff_row[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        sub1_j = A_diff_row[j] * B_diff_row[j]
        
        # A_diff_col[j] = a[i,j] - a[j,j]
        # B_diff_col[j] = b[p[j], v] - b[p[j], u] = b[v, v] - b[v, u]
        sub2_j = A_diff_col[j] * B_diff_col[j]
        
        Sum1 = Sum1_all - sub1_i - sub1_j
        Sum2 = Sum2_all - sub2_i - sub2_j
        
        return d1 + Sum1 + Sum2

    # Phase 1: Fast Local Search (Hill Climbing) to polish initial solution
    # Use Best Improvement or First Improvement? First is faster per iteration, Best is better convergence.
    # Given n=150, n^2/2 ~ 11k pairs. 
    # Let's do a few passes of First Improvement to get out of obvious bad spots.
    
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 30.0:
                break
            for j in range(i + 1, n):
                delta = calc_delta(i, j, p)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break # First improvement, restart i loop? Or continue? 
                    # Restarting i loop is standard for steepest descent, but slow.
                    # Let's just continue to j+1 to find more improvements in this pass.
        # If we found improvements, do another pass.
    
    # Phase 2: Simulated Annealing
    # Start temperature high enough to accept many moves, cool down.
    # Current cost is local min from Phase 1.
    
    T = 1000.0 # Initial temperature (arbitrary scale, delta is integer)
    # Max delta can be large. Let's estimate typical delta magnitude.
    # Or just start high.
    cooling_rate = 0.99
    min_T = 1.0
    
    random.seed(42) # Deterministic
    
    while time.time() < deadline - 5.0:
        if T < min_T:
            break
            
        # Pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        delta = calc_delta(i, j, p)
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= cooling_rate
        
        # Optional: Restart if stuck? 
        # For now, simple cooling.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
