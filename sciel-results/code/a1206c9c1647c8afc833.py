import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # Actually, the tensor is T[i, j, k] = 1 if it's a term in the product.
    # The product is C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T[i, j, k] = 1 for all i, j, k.
    # Wait, the tensor is defined by the trilinear form.
    # The entries of the tensor are 1 for all (i, j, k).
    
    # We want to find U, V, W such that sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1 for all i, j, k.
    
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Check schoolbook
    # It is always valid.
    
    # Try to find lower rank
    # We will try ranks from rank_schoolbook - 1 down to rank_schoolbook - 10
    # For each rank, we try a few random initializations and ALS
    
    def verify(U, V, W):
        # Check if sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] == 1
        # U shape: (n*m, r)
        # V shape: (m*p, r)
        # W shape: (p*n, r)
        # We need to check for all i, j, k
        # This is equivalent to checking if the tensor product equals the target tensor.
        # We can compute the tensor product for each c and sum.
        # But it's slow.
        # Instead, we can check a few random indices or all indices if small.
        # n=3, m=5, p=7 -> 105 indices.
        # We can compute the full tensor product.
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can use einsum.
        # U: (i, j, c) -> (n, m, r)
        # V: (j, k, c) -> (m, p, r)
        # W: (k, i, c) -> (p, n, r)
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # This is not a standard einsum because of the index permutations.
        # U[i, j, c] V[j, k, c] W[k, i, c]
        # Let's reshape U to (n, m, r), V to (m, p, r), W to (p, n, r).
        # Then T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # This is a contraction over c, but i, j, k are free.
        # We can compute this by looping over c? No, r is up to 105.
        # Looping over c is 105 * 105 = 11025 ops. Very fast.
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            # U[:, :, c] is (n, m)
            # V[:, :, c] is (m, p)
            # W[:, :, c] is (p, n)
            # We need to add U[i, j] * V[j, k] * W[k, i] to T[i, j, k]
            # This is an outer product of U[:, :, c], V[:, :, c], W[:, :, c] with specific index mapping.
            # T[i, j, k] += U[i, j] * V[j, k] * W[k, i]
            # We can do this with einsum: 'ij, jk, ki -> ijk'
            T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
            
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-9)

    def als_decompose(target_tensor, n, m, p, rank, max_iters=50, tol=1e-6):
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Reshape for convenience
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            # Update U
            # For each i, j, c, U[i, j, c] should be such that sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c]) = 1
            # Let M_U[i, j, c] = sum_k V[j, k, c] * W[k, i, c]
            # Then we need sum_c U[i, j, c] * M_U[i, j, c] = 1
            # This is a linear system for each (i, j) if we consider c as the variable?
            # No, U[i, j, :] is a vector of length rank.
            # We want U[i, j, :] such that U[i, j, :] dot M_U[i, j, :] = 1.
            # This is underdetermined. We can use least squares.
            # But we want to minimize the residual.
            # Actually, we can update U by solving a least squares problem for each (i, j).
            # Or, we can use the standard ALS update:
            # U[i, j, :] = (M_U[i, j, :]^T M_U[i, j, :])^{-1} M_U[i, j, :]^T * 1
            # This is the minimum norm solution.
            
            # Compute M_U
            # M_U[i, j, c] = sum_k V[j, k, c] W[k, i, c]
            # M_U = einsum('jkc, kic -> ijc', V_r, W_r)
            M_U = np.einsum('jkc,kic->ijc', V_r, W_r)
            
            # Update U
            # For each i, j, solve U[i, j, :] dot M_U[i, j, :] = 1
            # This is a linear equation with rank variables.
            # We can use np.linalg.lstsq
            # But we can do it vectorized.
            # U[i, j, :] = M_U[i, j, :] / (M_U[i, j, :] dot M_U[i, j, :])
            # This is the minimum norm solution to x dot m = 1.
            # Let's check: (m / |m|^2) dot m = 1. Correct.
            
            norm_sq = np.sum(M_U ** 2, axis=2, keepdims=True)
            norm_sq[norm_sq < 1e-10] = 1e-10 # Avoid division by zero
            U_r = M_U / norm_sq
            
            # Update V
            # M_V[j, k, c] = sum_i U[i, j, c] W[k, i, c]
            M_V = np.einsum('ijc,kic->jkc', U_r, W_r)
            norm_sq = np.sum(M_V ** 2, axis=2, keepdims=True)
            norm_sq[norm_sq < 1e-10] = 1e-10
            V_r = M_V / norm_sq
            
            # Update W
            # M_W[k, i, c] = sum_j U[i, j, c] V[j, k, c]
            M_W = np.einsum('ijc,jkc->kic', U_r, V_r)
            norm_sq = np.sum(M_W ** 2, axis=2, keepdims=True)
            norm_sq[norm_sq < 1e-10] = 1e-10
            W_r = M_W / norm_sq
            
            # Compute error
            # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = np.sum((T - 1.0) ** 2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U_r.reshape(-1, rank), V_r.reshape(-1, rank), W_r.reshape(-1, rank)

    # Try to find a decomposition with rank < 105
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Try ranks 104 down to 100
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U, V, W = als_decompose(None, n, m, p, r, max_iters=100)
            
            # Check if the solution is close to integers
            # We can try to round and verify
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            if verify(U_int, V_int, W_int):
                best_rank = r
                best_U = U_int
                best_V = V_int
                best_W = W_int
                found = True
                break
        
        if found:
            # If we found a lower rank, we can try to go even lower
            continue
        else:
            # If we didn't find one for this rank, maybe lower ranks are harder?
            # Or maybe we just need more attempts.
            # For now, break if we fail for a rank?
            # No, let's try the next lower rank.
            pass

    # If we found a better rank, return it
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
