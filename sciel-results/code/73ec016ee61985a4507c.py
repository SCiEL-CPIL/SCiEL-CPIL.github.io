"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank matching
    # Heavy flows to short distances
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    # Initial permutation: p[facility] = location
    # heavy_order[i] is the i-th heaviest facility
    # central_order[i] is the i-th most central location
    p = np.zeros(n, dtype=int)
    for i in range(n):
        p[heavy_order[i]] = central_order[i]
        
    # Precompute cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Discrete Polish (Local Search)
    # Tabu Search with 2-swap neighborhood
    
    tabu_size = 30
    tabu_list = {}
    current_step = 0
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_list.clear()
            continue
            
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Precompute Bp = b[:, p] for faster access? 
        # Actually, delta for swapping i and j:
        # Let u = p[i], v = p[j]
        # Change in cost = sum_k a[i,k] (b[v, p[k]] - b[u, p[k]]) + sum_k a[j,k] (b[u, p[k]] - b[v, p[k]])
        # + terms involving i and j themselves are handled by the swap logic.
        # A more efficient delta calculation:
        # Delta = (a[i,j] + a[j,i]) * (b[p[i], p[j]] - b[p[j], p[i]]) + 
        #         sum_{k != i,j} (a[i,k] + a[j,k]) * (b[p[k], p[j]] - b[p[k], p[i]]) ... wait, indices are tricky.
        
        # Let's use the standard efficient delta for QAP swap (i, j):
        # Let u = p[i], v = p[j]
        # For any k not i, j:
        # Old term: a[i,k]b[u, p[k]] + a[k,i]b[p[k], u] + a[j,k]b[v, p[k]] + a[k,j]b[p[k], v]
        # New term: a[i,k]b[v, p[k]] + a[k,i]b[p[k], v] + a[j,k]b[u, p[k]] + a[k,j]b[p[k], u]
        # Diff_k = (a[i,k] + a[k,i]) * (b[v, p[k]] - b[u, p[k]]) + (a[j,k] + a[k,j]) * (b[u, p[k]] - b[v, p[k]])
        #        = (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
        
        # For k=i:
        # Old: a[i,i]b[u,u]
        # New: a[i,i]b[v,v]
        # Diff_i = a[i,i](b[v,v] - b[u,u])
        
        # For k=j:
        # Old: a[j,j]b[v,v]
        # New: a[j,j]b[u,u]
        # Diff_j = a[j,j](b[u,u] - b[v,v])
        
        # Cross term i, j:
        # Old: a[i,j]b[u,v] + a[j,i]b[v,u]
        # New: a[i,j]b[v,u] + a[j,i]b[u,v]
        # Diff_ij = (a[i,j] + a[j,i]) * (b[v,u] - b[u,v])
        
        # Total Delta = Diff_i + Diff_j + Diff_ij + sum_{k != i, j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
        
        # This can be vectorized if we precompute row sums of A and B?
        # Let's try a simpler O(n^2) scan but optimized.
        
        # To speed up, we can precompute:
        # R_a[i] = sum_k a[i,k]
        # R_b[u] = sum_k b[u,k]
        # But the delta depends on specific u, v.
        
        # Let's just do the double loop, it's O(n^2) per step. n=150, so 22500 pairs.
        # If we do 1000 steps, that's 22.5M operations. Python loops are slow.
        # We need to vectorize the delta calculation or use a faster heuristic.
        
        # Alternative: Use a random greedy local search?
        # Or use a limited number of random swaps and keep if better?
        
        # Let's try vectorizing the delta calculation for all j for a fixed i?
        # For fixed i, u=p[i].
        # Delta(i, j) for all j.
        # v = p[j]
        # Term 1: (a[i,j] + a[j,i]) * (b[v,u] - b[u,v])
        # Term 2: a[i,i](b[v,v] - b[u,u]) + a[j,j](b[u,u] - b[v,v])
        # Term 3: sum_{k != i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
        
        # This is still complex to vectorize fully for all pairs (i,j) simultaneously without O(n^3) or large memory.
        
        # Given the time limit, let's stick to a simpler local search:
        # 1-pair swap (2-opt) but evaluated sequentially.
        
        # Actually, for n=150, a simple random swap local search might be faster in Python than a complex vectorized one if not done right.
        # But Tabu is better.
        
        # Let's try to optimize the inner loop.
        # Precompute A_sym = a + a.T
        # Precompute B_sym = b + b.T (if b is symmetric, this is 2b, but let's be general)
        # Actually, the formula above uses a[i,k] + a[k,i].
        
        # Let's use a simpler heuristic: Random 2-swap with immediate acceptance if better.
        # This is fast in Python if we just do it.
        
        # To make it faster, we can use NumPy for the delta calculation of a specific swap.
        
        # Let's try a different approach: 
        # Since n is small, we can afford O(n^2) per step if the constant is low.
        # But Python loops are the bottleneck.
        
        # Let's try to vectorize the search for the best swap for a given i.
        # For a fixed i, we want to find j that minimizes Delta(i,j).
        
        # Let's precompute:
        # A_row_sum[i] = sum_k a[i,k]
        # B_col_sum[u] = sum_k b[k,u]
        
        # The main term is sum_{k} (a[i,k] - a[j,k]) * (b[p[k], v] - b[p[k], u]) ... no, the formula is:
        # sum_{k != i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[v, p[k]] - b[u, p[k]])
        
        # Let S_i_k = a[i,k] + a[k,i]
        # Let S_j_k = a[j,k] + a[k,j]
        
        # Term = sum_{k != i,j} (S_i_k - S_j_k) * (b[v, p[k]] - b[u, p[k]])
        # = sum_{k != i,j} S_i_k * (b[v, p[k]] - b[u, p[k]]) - sum_{k != i,j} S_j_k * (b[v, p[k]] - b[u, p[k]])
        
        # Let V_k = b[v, p[k]] - b[u, p[k]]
        
        # This is getting complicated. Let's just use a simple random walk with restarts.
        
        # Strategy:
        # 1. Start with rank matching.
        # 2. Do 2000 iterations of:
        #    a. Pick random i, j.
        #    b. Calculate cost delta (using a fast vectorized method if possible, or just full recalculation if n is small? No, n=150 is too big for full recalculation).
        #    c. If delta < 0, accept.
        #    d. If no improvement for 100 steps, perturb.
        
        # How to calculate delta fast?
        # We can precompute the current cost matrix contribution.
        # Cost = sum a[i,j] b[p[i], p[j]]
        
        # Let's try to use the fact that swapping i and j only changes terms involving i or j.
        # Changed terms:
        # 1. (i,i), (i,j), (j,i), (j,j)
        # 2. (i,k), (k,i) for k != i,j
        # 3. (j,k), (k,j) for k != i,j
        
        # Let's implement the delta calculation using NumPy operations on rows/cols.
        
        # Precompute:
        # A = a
        # B = b
        # p = current permutation
        # Bp = B[:, p]  # This is the permuted B. Bp[u, v] = B[u, p[v]]? No.
        # If we define B_perm = B[p][:, p], then B_perm[i, j] = B[p[i], p[j]].
        # Cost = sum A * B_perm.
        
        # If we swap p[i] and p[j]:
        # New B_perm' differs from B_perm only in rows i, j and cols i, j.
        
        # Let u = p[i], v = p[j].
        # In B_perm:
        # Row i is B[u, p[0:n]]
        # Row j is B[v, p[0:n]]
        # Col i is B[p[0:n], u]
        # Col j is B[p[0:n], v]
        
        # After swap:
        # Row i becomes B[v, p[0:n]] (old row j)
        # Row j becomes B[u, p[0:n]] (old row i)
        # Col i becomes B[p[0:n], v] (old col j)
        # Col j becomes B[p[0:n], u] (old col i)
        
        # Delta = sum_{k} A[i,k] (B_perm'[i,k] - B_perm[i,k]) + A[j,k] (B_perm'[j,k] - B_perm[j,k])
        #       + sum_{k} A[k,i] (B_perm'[k,i] - B_perm[k,i]) + A[k,j] (B_perm'[k,j] - B_perm[k,j])
        #       - (double counting correction for i,j terms if any? No, the sums cover all k)
        #       Wait, the terms (i,i), (i,j), (j,i), (j,j) are included in both row and col sums?
        #       Row sum for i includes k=i and k=j.
        #       Col sum for i includes k=i and k=j.
        #       So (i,i) is counted in Row i (k=i) and Col i (k=i).
        #       In the full cost sum, a[i,i] b[p[i], p[i]] is one term.
        #       In my delta formula:
        #       Row i term for k=i: A[i,i] (B_perm'[i,i] - B_perm[i,i])
        #       Col i term for k=i: A[i,i] (B_perm'[i,i] - B_perm[i,i])
        #       So it's counted twice. But in the actual cost, it's only one term.
        #       So we need to be careful.
        
        # Let's use the explicit delta formula derived earlier, it's safer.
        # Delta = (a[i,i] + a[j,j]) * (b[v,v] - b[u,u])  # Wait, signs?
        # Let's re-derive carefully.
        
        # Old Cost terms involving i or j:
        # C_old = sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u] + sum_k a[j,k] b[v, p[k]] + sum_k a[k,j] b[p[k], v]
        # Note: terms (i,i), (i,j), (j,i), (j,j) are included in these sums.
        # Specifically:
        # k=i in sum 1: a[i,i] b[u,u]
        # k=j in sum 1: a[i,j] b[u,v]
        # k=i in sum 2: a[i,i] b[u,u]  (Wait, p[i]=u, so b[p[i], u] = b[u,u])
        # k=j in sum 2: a[j,i] b[v,u]
        # k=i in sum 3: a[j,i] b[v,u]
        # k=j in sum 3: a[j,j] b[v,v]
        # k=i in sum 4: a[i,j] b[u,v]
        # k=j in sum 4: a[j,j] b[v,v]
        
        # So C_old = a[i,i]b[u,u] + a[i,j]b[u,v] + a[i,i]b[u,u] + a[j,i]b[v,u] + a[j,i]b[v,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,j]b[v,v]
        #          + sum_{k != i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v])
        #          = 2a[i,i]b[u,u] + 2a[i,j]b[u,v] + 2a[j,i]b[v,u] + 2a[j,j]b[v,v]
        #          + sum_{k != i,j} (...)
        
        # New Cost terms involving i or j (after swap u<->v):
        # p'[i]=v, p'[j]=u
        # C_new = 2a[i,i]b[v,v] + 2a[i,j]b[v,u] + 2a[j,i]b[u,v] + 2a[j,j]b[u,u]
        #         + sum_{k != i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u])
        
        # Delta = C_new - C_old
        # Delta = 2a[i,i](b[v,v]-b[u,u]) + 2a[j,j](b[u,u]-b[v,v])
        #         + 2a[i,j](b[v,u]-b[u,v]) + 2a[j,i](b[u,v]-b[v,u])
        #         + sum_{k != i,j} (a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u]) + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]))
        # Delta = 2(a[i,i]-a[j,j])(b[v,v]-b[u,u]) + 2(a[i,j]-a[j,i])(b[v,u]-b[u,v])
        #         + sum_{k != i,j} (a[i,k]+a[k,i]-a[j,k]-a[k,j]) (b[v,p[k]]-b[u,p[k]])
        
        # This matches my previous derivation.
        
        # To speed up the sum over k, we can precompute:
        # Let A_sym = a + a.T
        # Let B_sym = b + b.T
        # The term is sum_{k != i,j} A_sym[i,k] (b[v,p[k]]-b[u,p[k]]) - A_sym[j,k] (b[v,p[k]]-b[u,p[k]])
        # = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]]-b[u,p[k]])
        
        # Let D_k = b[v,p[k]] - b[u,p[k]]
        # We want sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) D_k
        # = sum_{k} (A_sym[i,k] - A_sym[j,k]) D_k - (A_sym[i,i]-A_sym[j,i])D_i - (A_sym[i,j]-A_sym[j,j])D_j
        
        # D_i = b[v,u] - b[u,u]
        # D_j = b[v,v] - b[u,v]
        
        # So we can compute the sum over all k using dot products.
        # Let V = b[v, p] - b[u, p]  (vector of length n)
        # Let W = A_sym[i] - A_sym[j]  (vector of length n)
        # Sum_all = dot(W, V)
        # Correction = (A_sym[i,i]-A_sym[j,i]) * D_i + (A_sym[i,j]-A_sym[j,j]) * D_j
        # Sum_rest = Sum_all - Correction
        
        # This is O(n) per pair (i,j)!
        # Total O(n^3) per step? No, O(n^2) pairs * O(n) per pair = O(n^3) per step.
        # n=150, n^3 = 3.3M. This is fast in NumPy.
        
        # So we can evaluate all pairs (i,j) in O(n^3) total per step?
        # No, we want to find the best pair.
        # We can vectorize over j for a fixed i?
        # Or just loop i, and for each i, compute delta for all j > i.
        
        # Let's try this.
        
        A_sym = a + a.T
        
        for i in range(n):
            u = p[i]
            # Compute V for all j? No, V depends on v=p[j].
            # We need to find j that minimizes Delta.
            
            # For fixed i, u is fixed.
            # Delta(i,j) = 2(a[i,i]-a[j,j])(b[v,v]-b[u,u]) + 2(a[i,j]-a[j,i])(b[v,u]-b[u,v])
            #              + sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]]-b[u,p[k]])
            
            # This is hard to vectorize over j because v=p[j] varies.
            
            # Let's just do the O(n^2) loop but with the O(n) dot product for the sum part.
            # This will be slow in Python.
            
            # Alternative: Just use random swaps.
            # Pick 2 random indices, calculate delta, accept if better.
            # This is O(n) per swap.
            # We can do many swaps.
            
            pass # Skip this complex loop, use random swaps below.
            
        # Let's implement random swaps with delta calculation.
        
        # Precompute A_sym
        if not hasattr(solve, 'A_sym'):
            solve.A_sym = a + a.T
        A_sym = solve.A_sym
        
        # Random Swap Local Search
        while time.time() < deadline:
            # Pick random i, j
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            u = p[i]
            v = p[j]
            
            # Calculate Delta
            # Term 1: 2(a[i,i]-a[j,j])(b[v,v]-b[u,u])
            t1 = 2 * (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
            
            # Term 2: 2(a[i,j]-a[j,i])(b[v,u]-b[u,v])
            t2 = 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
            
            # Term 3: sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]]-b[u,p[k]])
            # Vectorized:
            # V = b[v, p] - b[u, p]
            # W = A_sym[i] - A_sym[j]
            # Sum_all = dot(W, V)
            # D_i = b[v,u] - b[u,u]
            # D_j = b[v,v] - b[u,v]
            # Correction = (A_sym[i,i]-A_sym[j,i]) * D_i + (A_sym[i,j]-A_sym[j,j]) * D_j
            # Sum_rest = Sum_all - Correction
            
            V = b[v, p] - b[u, p]
            W = A_sym[i] - A_sym[j]
            Sum_all = np.dot(W, V)
            
            D_i = b[v,u] - b[u,u]
            D_j = b[v,v] - b[u,v]
            
            Correction = (A_sym[i,i] - A_sym[j,i]) * D_i + (A_sym[i,j] - A_sym[j,j]) * D_j
            Sum_rest = Sum_all - Correction
            
            Delta = t1 + t2 + Sum_rest
            
            if Delta < 0:
                # Accept
                p[i], p[j] = p[j], p[i]
                current_cost += int(Delta)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            if no_improve_count > 500:
                # Perturb
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve_count = 0
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
