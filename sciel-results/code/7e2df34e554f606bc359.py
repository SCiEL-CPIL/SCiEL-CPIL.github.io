import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Convert to float for faster vectorized ops, 
    # but keep in mind exact integer requirement for final score? 
    # The evaluator computes the score. We just need to return p.
    # Using float64 for internal delta calculations is standard and safe for QAPLIB sizes.
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # 1. Initialization: Rank Matching
    # Heavy flows (high row sum in A) should go to short distances (low row sum in B)
    # Wait, if A[i,k] is flow, high flow means we want B[p[i], p[k]] to be small.
    # So facility i with high total flow should be placed at a location with low total distance?
    # Actually, the interaction is pairwise.
    # Heuristic: Sort facilities by row sum of A descending.
    # Sort locations by row sum of B ascending.
    # Assign facility with highest flow to location with lowest distance sum.
    
    a_sums = A.sum(axis=1)
    b_sums = B.sum(axis=1)
    
    # Sort facilities by flow intensity (desc)
    fac_order = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance intensity (asc)
    loc_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    for i_idx, f_idx in enumerate(fac_order):
        p[f_idx] = loc_order[i_idx]
        
    # Calculate initial cost
    # Cost = sum_{i,j} A[i,j] B[p[i], p[j]]
    # This is trace(A @ B_perm) where B_perm[i,j] = B[p[i], p[j]]
    # Or simply:
    B_perm = B[np.ix_(p, p)]
    current_cost = np.sum(A * B_perm)
    
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Local Search: 2-opt (Swap)
    # Precompute row differences? No, just compute on the fly.
    # To speed up, we can try to find the best swap in each iteration.
    # For n=150, 11k pairs.
    
    while time.time() < deadline:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Iterate over all pairs i < j
        # Vectorized approach for finding best swap is complex due to dependency on p.
        # We can batch evaluate swaps.
        
        # Let's try a simple loop first, optimized with NumPy dot products.
        # For each i, we can compute the delta for swapping i with all j > i?
        
        # Actually, a full scan of all pairs might be slow if done naively in Python loops.
        # Let's vectorize the evaluation of deltas for a fixed i against all j.
        
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            ui = p[i]
            # We want to find j that minimizes delta.
            # Delta(i, j) = sum_{k != i, j} (A[i,k] - A[j,k]) * (B[p[j], p[k]] - B[p[i], p[k]])
            # This depends on j in two ways: A[j,k] and B[p[j], p[k]].
            
            # Let's just do a simple random restart of 2-opt if stuck?
            # Or just scan all pairs.
            
            # To make this fast, we can pre-compute some terms?
            # Let V_j = B[p[j], :]
            # Let W_i = A[i, :]
            # Delta is roughly (W_i - W_j) . (V_j - V_i)
            
            # Let's try evaluating all pairs in a batch using broadcasting?
            # n=150. 150x150 matrix of deltas.
            # Memory: 150*150*8 bytes = 180KB. Very small.
            # Time: O(n^3) if done naively?
            # Delta[i,j] calculation involves a dot product of length n.
            # So total ops n^3 = 150^3 = 3.3 million.
            # In NumPy, this is extremely fast (< 1ms).
            
            # Let's compute the delta matrix for all pairs (i,j) in one go.
            
            # P is the permutation.
            # A is n x n.
            # B_perm is n x n.
            
            # We need Delta[i,j] for all i,j.
            # Formula: Delta(i,j) = sum_k (A[i,k] - A[j,k]) * (B_perm[j,k] - B_perm[i,k])
            # Note: B_perm[j,k] is B[p[j], p[k]].
            # Let M1[i,j,k] = A[i,k] - A[j,k]
            # Let M2[i,j,k] = B_perm[j,k] - B_perm[i,k]
            # This is too much memory (150^3).
            
            # Alternative:
            # Delta(i,j) = A[i,:] . (B_perm[j,:] - B_perm[i,:]) - A[j,:] . (B_perm[j,:] - B_perm[i,:])
            # = A[i,:] . B_perm[j,:] - A[i,:] . B_perm[i,:] - A[j,:] . B_perm[j,:] + A[j,:] . B_perm[i,:]
            
            # Let C = A @ B_perm.T
            # C[i,j] = sum_k A[i,k] B_perm[j,k] = sum_k A[i,k] B[p[j], p[k]]
            
            # Then:
            # A[i,:] . B_perm[j,:] = C[i,j]
            # A[i,:] . B_perm[i,:] = C[i,i]
            # A[j,:] . B_perm[j,:] = C[j,j]
            # A[j,:] . B_perm[i,:] = C[j,i]
            
            # So Delta(i,j) = C[i,j] - C[i,i] - C[j,j] + C[j,i]
            
            # This is O(1) per pair after computing C!
            # Computing C = A @ B_perm.T is O(n^3) matrix multiplication.
            # For n=150, this is very fast (~1ms).
            
            C = A @ B_perm.T
            
            # Compute Delta matrix
            # Delta[i,j] = C[i,j] - C[i,i] - C[j,j] + C[j,i]
            # We want min Delta for i != j.
            
            # diag_C = np.diag(C)
            # Delta = C - diag_C[:, None] - diag_C[None, :] + C.T
            
            diag_C = np.diag(C)
            Delta = C - diag_C[:, None] - diag_C[None, :] + C.T
            
            # Exclude i=j (diagonal is 0 anyway if formula holds? 
            # Delta(i,i) = C[i,i] - C[i,i] - C[i,i] + C[i,i] = 0. Correct.)
            
            # Find min value in Delta
            # We only care about i < j to avoid duplicates, but min is symmetric.
            min_delta = np.min(Delta)
            
            if min_delta < 1e-9: # Strictly improving
                # Find indices
                idx = np.unravel_index(np.argmin(Delta), Delta.shape)
                i, j = idx
                # Ensure i != j
                if i != j:
                    # Apply swap
                    p[i], p[j] = p[j], p[i]
                    B_perm = B[np.ix_(p, p)]
                    current_cost += min_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                else:
                    break
            else:
                # No improving swap found
                break
                
    return best_p.tolist()
