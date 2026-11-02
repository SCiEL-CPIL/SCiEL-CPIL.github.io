import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target tensor: T[i, j, k] = 1 for all i, j, k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Compute contraction
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor form: sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1
        # Let's compute the resulting tensor T'
        # T'[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape to 3D for easier indexing
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Sum over c
        # T'_i_j_k = sum_c U_i_j_c * V_j_k_c * W_k_i_c
        # We can do this via einsum or loops
        # T' = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Check if T' is all ones
        # Due to floating point, we check closeness to 1
        # But we need exact integers.
        # So we check if rounded T' is all ones and original T' is close to 1.
        
        T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Check if it's close to 1
        if not np.allclose(T_prime, 1.0, atol=1e-5):
            return False
        
        # Check if it's exactly 1 when rounded
        if not np.allclose(np.round(T_prime), 1.0, atol=1e-9):
            return False
            
        return True

    def search_rank(r):
        # Try to find a solution of rank r
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random integers in {-1, 0, 1} or small range
            # Random integers often work better for tensor rank than reals
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            # Local search: try to flip signs or add small perturbations?
            # Better: Gradient descent on reals, then round.
            
            U_real = U.astype(float)
            V_real = V.astype(float)
            W_real = W.astype(float)
            
            # Scale to avoid overflow/underflow
            # Normalize norms?
            
            lr = 0.1
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                # Compute T'
                U_3d = U_real.reshape(n, m, r)
                V_3d = V_real.reshape(m, p, r)
                W_3d = W_real.reshape(p, n, r)
                
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                
                # Error
                error = np.sum((T_prime - 1.0)**2)
                
                if error < 1e-6:
                    # Check exactness
                    if verify(U_real, V_real, W_real):
                        return U_real, V_real, W_real
                    # Try rounding
                    U_int = np.round(U_real)
                    V_int = np.round(V_real)
                    W_int = np.round(W_real)
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    
                # Gradient
                # dT/dc_U[i,j,c] = V[j,k,c] * W[k,i,c] (summed over k? No, T is indexed by i,j,k)
                # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # dT[i,j,k]/dU[i',j',c'] = delta[i,i'] delta[j,j'] V[j,k,c'] W[k,i,c']
                
                # This is complex to compute analytically fast.
                # Use numerical gradient? Too slow.
                
                # Alternative: Simple local search on integers.
                # Break out and do integer local search.
                break
            
            # Integer local search
            U_int = np.round(U_real).astype(int)
            V_int = np.round(V_real).astype(int)
            W_int = np.round(W_real).astype(int)
            
            current_error = np.sum((np.einsum('ijc,jkc,kic->ijk', 
                U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r)) - 1.0)**2)
            
            for _ in range(2000):
                if time.time() > deadline:
                    return None
                
                # Pick a random element to perturb
                mat = random.choice(['U', 'V', 'W'])
                idx = random.choice(range(mat.__len__() if False else (n*m if mat=='U' else m*p if mat=='V' else p*n)))
                col = random.randrange(r)
                
                old_val = (U_int if mat=='U' else V_int if mat=='V' else W_int)[idx, col]
                
                # Try -1, 0, 1, 2, -2
                for delta in [-2, -1, 0, 1, 2]:
                    if delta == 0: continue
                    new_val = old_val + delta
                    if abs(new_val) > 5: continue
                    
                    # Update
                    if mat == 'U':
                        U_int[idx, col] = new_val
                    elif mat == 'V':
                        V_int[idx, col] = new_val
                    else:
                        W_int[idx, col] = new_val
                    
                    # Compute error
                    T_prime = np.einsum('ijc,jkc,kic->ijk', 
                        U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r))
                    new_error = np.sum((T_prime - 1.0)**2)
                    
                    if new_error < current_error:
                        current_error = new_error
                        if new_error < 1e-9:
                            if verify(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                    else:
                        # Revert
                        if mat == 'U':
                            U_int[idx, col] = old_val
                        elif mat == 'V':
                            V_int[idx, col] = old_val
                        else:
                            W_int[idx, col] = old_val

        return None

    # Try ranks from 9 to 12
    for r in range(9, 13):
        if time.time() > deadline:
            break
        result = search_rank(r)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-END
