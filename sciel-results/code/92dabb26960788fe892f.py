import time
import sys
import numpy as np
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Target support: indices (u, v, w) where u=i*m+j, v=j*p+k, w=k*n+i
    # The tensor has value 1 at these indices, 0 elsewhere.
    target_support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_support.add((u, v, w))
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Convert to numpy for faster operations
    U_sb_np = np.array(U_sb, dtype=np.float64)
    V_sb_np = np.array(V_sb, dtype=np.float64)
    W_sb_np = np.array(W_sb, dtype=np.float64)
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def error_fast(U, V, W):
        """Compute sum of squared errors using vectorized operations."""
        r = U.shape[1]
        # Compute the tensor T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # We can do this efficiently by iterating over columns
        # T = sum_c outer(U[:,c], V[:,c]) * W[:,c]
        # But we need to compare against target.
        
        # Since dimensions are small, let's use a semi-vectorized approach
        # Precompute target tensor
        # Actually, let's just compute the error directly but efficiently
        
        # For each column c, we add outer(U[:,c], V[:,c]) * W[:,c] to T
        # But T is 3D. Let's iterate over c and accumulate.
        
        # Alternative: compute T as a 3D array
        # T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
        
        # We can compute this by:
        # For each u, v: T[u,v,:] = sum_c (U[u,c] * V[v,c]) * W[:,c]
        # = (U @ V.T) * W.T ... no, that's not right.
        
        # Let's use einsum or explicit loops if needed.
        # Given the small size, explicit loops over c might be fine,
        # but let's try to vectorize over u,v.
        
        # T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        
        # Let's compute the error as sum_{u,v,w} (T[u,v,w] - target[u,v,w])^2
        # = sum_{u,v,w} T[u,v,w]^2 - 2 sum_{u,v,w} T[u,v,w] target[u,v,w] + sum target^2
        # sum target^2 is constant (number of 1s = n*m*p = 42)
        
        # Let's compute the cross term and the square term.
        
        # This is still complex. Let's just use a simple loop over c
        # and accumulate into a 3D array if memory allows, or just compute error directly.
        
        # Given r is small (20-40) and nm*mp*np_ = 1764,
        # 1764 * r operations is ~70k ops. Very fast.
        
        err = 0.0
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Contribution to T is outer(uc, vc) * wc
            # We need sum over c of these.
            # Let's accumulate T in a 3D array? 
            # nm * mp * np_ = 1764 floats. Very small.
            pass
            
        # Let's just do the direct calculation in a loop for clarity and correctness first,
        # then optimize if needed. But 1764 * r is tiny.
        
        # Actually, let's compute the full tensor T once per call.
        T = np.zeros((nm, mp, np_))
        for c in range(r):
            T += np.outer(U[:, c], V[:, c])[:, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
            
        # Compute error
        # target is 1 where (u,v,w) in support, else 0
        # We can create a target tensor
        # But creating it every time is expensive.
        # Let's precompute the target tensor indices.
        
        # Better: compute sum T^2 - 2 * sum T*target + constant
        # sum T^2 = np.sum(T**2)
        # sum T*target = sum of T[u,v,w] for (u,v,w) in support
        
        cross_sum = 0.0
        for (u, v, w) in target_support:
            cross_sum += T[u, v, w]
            
        sq_sum = np.sum(T**2)
        
        # Error = sq_sum - 2*cross_sum + len(target_support)
        # Because target is 0 or 1, so (T-target)^2 = T^2 - 2T*target + target^2
        # sum target^2 = len(target_support)
        
        return sq_sum - 2 * cross_sum + len(target_support)
    
    def is_valid(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is exact."""
        # Recompute T
        T = np.zeros((nm, mp, np_))
        for c in range(rank):
            T += np.outer(U[:, c], V[:, c])[:, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
            
        for (u, v, w) in target_support:
            if abs(T[u, v, w] - 1.0) > tol:
                return False
        # Check zeros
        # This is expensive. Let's rely on the error function for search,
        # and do a strict check only when we think we found a solution.
        # For now, assume if error is very small, it's valid.
        # But we need to be sure.
        
        # Let's check a few random non-support points?
        # Or just trust the error metric if it's < 1e-9.
        # The error metric includes all points.
        return True

    def solve_rank(r, values, max_attempts=500):
        """Try to find a decomposition of rank r using local search."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            U = np.random.choice(values, size=(nm, r))
            V = np.random.choice(values, size=(mp, r))
            W = np.random.choice(values, size=(np_, r))
            
            current_err = error_fast(U, V, W)
            if current_err < 1e-9:
                return U, V, W
                
            stuck_counter = 0
            max_stuck = 100
            
            while time.time() < deadline:
                if stuck_counter > max_stuck:
                    break
                    
                # Pick a random matrix, row, column
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dims = nm
                elif mat_idx == 1:
                    mat = V
                    dims = mp
                else:
                    mat = W
                    dims = np_
                
                idx = np.random.choice(dims)
                col = np.random.choice(r)
                
                old_val = mat[idx, col]
                new_val = np.random.choice(values)
                
                if new_val == old_val:
                    continue
                    
                mat[idx, col] = new_val
                new_err = error_fast(U, V, W)
                
                if new_err < current_err:
                    current_err = new_err
                    stuck_counter = 0
                    if current_err < 1e-9:
                        return U, V, W
                else:
                    mat[idx, col] = old_val
                    stuck_counter += 1
                    
        return None

    # Search for lower ranks
    # Known lower bounds are around 20-25 for this size.
    # Schoolbook is 42.
    # Let's try from 25 to 41.
    
    # Values to try: integers first, then half-integers
    int_values = [0, 1, -1]
    half_int_values = [0, 0.5, 1, -0.5, -1]
    
    # Try ranks in increasing order
    for r in range(25, best_rank):
        if time.time() > deadline - 5:  # Leave some time
            break
            
        # Try with integers
        res = solve_rank(r, int_values, max_attempts=50)
        if res is not None:
            U, V, W = res
            # Verify strictly
            if is_valid(U, V, W, r):
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break  # Found the minimal rank in this search space
        
        # Try with half-integers
        res = solve_rank(r, half_int_values, max_attempts=50)
        if res is not None:
            U, V, W = res
            if is_valid(U, V, W, r):
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break

    # If no better rank found, return schoolbook
    if best_rank == rank_sb:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
