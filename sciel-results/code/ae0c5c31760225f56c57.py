import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of the matrices in the decomposition
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # The target tensor M has non-zero entries at (i*m+j, j*p+k, k*n+i)
    # We represent the tensor as a flat array or a 3D array
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # Precompute the indices of the non-zero entries in the target tensor
    # and their values (which are 1)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    def check_solution(U, V, W):
        # U: (dim_U, R), V: (dim_V, R), W: (dim_W, R)
        # Check if sum_r U[:,r] V[:,r] W[:,r] matches the target tensor
        # For efficiency, we only check the non-zero entries of the target tensor
        # and ensure that other entries are 0? 
        # Actually, the tensor multiplication is defined by the bilinear form.
        # The condition is that for ALL a,b,c, sum_r U[a,r]V[b,r]W[c,r] == M[a,b,c]
        # M[a,b,c] is 1 if (a,b,c) is in target_indices, else 0.
        
        R = U.shape[1]
        # Compute the full tensor product sum
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow if R is large, but R <= 45, dims are small.
        T = np.zeros((dim_U, dim_V, dim_W))
        for r in range(R):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        # Check target entries
        for a, b, c in target_indices:
            if T[a, b, c] != 1:
                return False
        
        # Check non-target entries are 0
        # We can do this by checking the whole tensor, but we can optimize
        # by noting that most entries are 0.
        # However, we must ensure no "cross-talk".
        # A faster way: 
        # The set of non-zero entries in T must be exactly the target_indices.
        # Let's check if any entry outside target_indices is non-zero.
        # Create a mask
        mask = np.zeros((dim_U, dim_V, dim_W), dtype=bool)
        for a, b, c in target_indices:
            mask[a, b, c] = True
            
        if np.any(T[~mask] != 0):
            return False
            
        return True

    def try_find_decomposition(rank, time_limit=10.0):
        start_time = time.time()
        R = rank
        # Initialize U, V, W with random small integers
        # We can bias towards 0, 1, -1
        U = np.random.randint(-1, 2, size=(dim_U, R))
        V = np.random.randint(-1, 2, size=(dim_V, R))
        W = np.random.randint(-1, 2, size=(dim_W, R))
        
        # Local search
        # We want to minimize the error.
        # Error = sum of (T[a,b,c] - M[a,b,c])^2
        
        def get_error(U, V, W):
            T = np.zeros((dim_U, dim_V, dim_W))
            for r in range(R):
                T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            # Calculate error on all entries
            # M is 1 at target_indices, 0 elsewhere
            # We can compute error as sum(T^2) - 2*sum(T at target) + num_target
            # But we also need T[~target] to be 0.
            # So error = sum(T^2) - 2 * (sum of T at target indices) + len(target_indices)
            # Wait, if T at target is 1, error is 0. If T is 0, error is 1.
            # If T is 2, error is 1.
            # This quadratic form works if we assume T is close to 0 or 1.
            
            # Faster error calculation:
            # err = 0
            # for a,b,c in target_indices:
            #    err += (T[a,b,c] - 1)**2
            # for all other a,b,c:
            #    err += T[a,b,c]**2
            
            # Let's just compute the full T and check.
            # To speed up, we can only update the error for changed entries.
            # But for simplicity, let's compute full T.
            # dim_U=9, dim_V=15, dim_W=15. Total 2025 entries.
            # R=45. 45 * 2025 ops is ~90k ops. Very fast.
            
            mask = np.zeros((dim_U, dim_V, dim_W), dtype=bool)
            for a, b, c in target_indices:
                mask[a, b, c] = True
                
            # T is already computed
            err = 0
            # Sum of squares of all T
            err += np.sum(T**2)
            # Subtract 2 * sum of T at target indices
            err -= 2 * np.sum(T[mask])
            # Add number of target indices (since M=1 there, (T-1)^2 = T^2 - 2T + 1)
            err += len(target_indices)
            
            return err

        current_error = get_error(U, V, W)
        
        while time.time() - start_time < time_limit:
            # Pick a random matrix and a random element
            mat_idx = random.randint(0, 2) # 0:U, 1:V, 2:W
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            r_idx = random.randint(0, R - 1)
            a_idx = random.randint(0, mat.shape[0] - 1)
            
            old_val = mat[a_idx, r_idx]
            
            # Try changing the value by +/- 1
            new_val = old_val + random.choice([-1, 1])
            
            mat[a_idx, r_idx] = new_val
            
            new_error = get_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    return U, V, W
            else:
                mat[a_idx, r_idx] = old_val
                
        return None

    # Try ranks from 44 down to 40
    # If we find a solution, return it
    # If not, return the schoolbook solution (rank 45)
    
    for rank in range(44, 39, -1):
        # Try a few times for each rank
        for attempt in range(3):
            res = try_find_decomposition(rank, time_limit=10.0)
            if res is not None:
                U, V, W = res
                # Convert to lists of lists as required
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
