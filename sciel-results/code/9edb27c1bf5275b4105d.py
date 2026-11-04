import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def verify_and_snap(U, V, W, tol=1e-9):
        # Snap to half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        
        # Verify exact decomposition with snapped values
        # T[i,j,k] should be 1
        # Sum_c U_s[im+j, c] V_s[jp+k, c] W_s[kn+i, c]
        
        # Efficient check using tensor contraction
        # We can reshape U, V, W to 3D tensors?
        # U is (n*m, R) -> (n, m, R)
        # V is (m*p, R) -> (m, p, R)
        # W is (p*n, R) -> (p, n, R)
        
        U_3d = U_s.reshape(n, m, -1)
        V_3d = V_s.reshape(m, p, -1)
        W_3d = W_s.reshape(p, n, -1)
        
        # Contract over R
        # Result[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        # Note: W index is (p, n), so W_3d[k, i, c] corresponds to W[k*n+i, c]
        
        # einsum: 'ijc,jkc,kic->ijk'
        try:
            reconstructed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        except:
            return None, None, None, False

        if np.allclose(reconstructed, 1.0, atol=tol):
            return U_s, V_s, W_s, True
        return None, None, None, False

    def optimize_rank(rank, max_iter=2000, lr=0.01):
        # Initialize
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Precompute indices for gradient
        # For each c, U[:, c] is a vector of length n*m.
        # Error E = sum_{i,j,k} (1 - sum_c U[im+j,c] V[jp+k,c] W[kn+i,c])^2
        
        # Gradient calculation:
        # Let S_c[i, j, k] = V[j*p+k, c] * W[k*n+i, c]
        # Then term for U[im+j, c] in the sum is S_c[i, j, k].
        # dE/dU[im+j, c] = -2 * sum_{i,j,k: im+j fixed} err[i,j,k] * S_c[i,j,k]
        # Since im+j fixes i and j, we sum over k.
        # dU[im+j, c] = -2 * sum_k err[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        
        # Similarly for V and W.
        
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        beta = 0.9
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Compute error and gradients
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # Reconstruct
            recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = 1.0 - recon
            
            # Gradients
            # dU[im+j, c] = -2 * sum_k err[i,j,k] * V_3d[j,k,c] * W_3d[k,i,c]
            # einsum: 'ijk,jkc,kic->ijc' -> then reshape to (n*m, rank)
            grad_U_3d = -2 * np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
            grad_U = grad_U_3d.reshape(n * m, rank)
            
            # dV[jp+k, c] = -2 * sum_i err[i,j,k] * U_3d[i,j,c] * W_3d[k,i,c]
            # einsum: 'ijk,ijc,kic->jkc'
            grad_V_3d = -2 * np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
            grad_V = grad_V_3d.reshape(m * p, rank)
            
            # dW[kn+i, c] = -2 * sum_j err[i,j,k] * U_3d[i,j,c] * V_3d[j,k,c]
            # einsum: 'ijk,ijc,jkc->kic'
            grad_W_3d = -2 * np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
            grad_W = grad_W_3d.reshape(p * n, rank)
            
            # Update with momentum
            momentum_U = beta * momentum_U + (1 - beta) * grad_U
            momentum_V = beta * momentum_V + (1 - beta) * grad_V
            momentum_W = beta * momentum_W + (1 - beta) * grad_W
            
            U -= lr * momentum_U
            V -= lr * momentum_V
            W -= lr * momentum_W
            
            # Periodic check for snapping
            if it % 100 == 0:
                U_s, V_s, W_s, success = verify_and_snap(U, V, W, tol=1e-8)
                if success:
                    return U_s, V_s, W_s
        
        # Final check
        U_s, V_s, W_s, success = verify_and_snap(U, V, W, tol=1e-8)
        if success:
            return U_s, V_s, W_s
        return None

    # Try ranks from low to high
    # Known rank for 3x4x6 is likely around 50-60.
    # Schoolbook is 72.
    # Let's try a few ranks.
    ranks_to_try = [54, 56, 58, 60, 62, 64, 66, 68, 70, 71]
    
    # Shuffle to add randomness in case some ranks are harder
    random.shuffle(ranks_to_try)
    
    best_result = None
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try a few restarts for this rank
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            res = optimize_rank(r, max_iter=1000, lr=0.05)
            if res:
                U, V, W = res
                # Return immediately if found
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we found a continuous solution but not snapped, we might keep it?
        # No, must be half-integers.
        
    return get_schoolbook()
# EVOLVE-BLOCK-END
