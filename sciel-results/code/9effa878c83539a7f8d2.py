"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can do this efficiently using einsum
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Reshape U, V, W to make indexing easier
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is a 3-index sum
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += U_r[:,:,c][:, :, None] * V_r[:,:,c][None, :, :] * W_r[:,:,c][None, None, :]
        
        return np.allclose(T, 1.0, atol=tol)

    def try_rank(r, max_iters=5000, seed=42):
        """
        Try to find a rank-r decomposition using structured ansatz.
        Uses block-diagonal structure with correction terms.
        """
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values scaled appropriately
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # Use gradient descent to minimize the error
        # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # dE/dU[i*m+j,c] = sum_{i',j',k': i'=i,j'=j} 2*(T-1)*V[j'*p+k',c]*W[k'*n+i',c]
            # Simplify: for fixed (i,j), dE/dU[i*m+j,c] = 2*(T[i,j,:]-1) @ (V[j,:,:] * W[:,i,:])
            
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            # Compute T
            T = np.zeros((n, m, p))
            for c in range(r):
                T += U_r[:,:,c][:, :, None] * V_r[:,:,c][None, :, :] * W_r[:,:,c][None, None, :]
            
            err = np.sum((T - 1.0)**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                return best_U, best_V, best_W
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            dU_r = dU.reshape(n, m, r)
            dV_r = dV.reshape(m, p, r)
            dW_r = dW.reshape(p, n, r)
            
            # dE/dU[i,j,c] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
            # dE/dV[j,k,c] = 2 * sum_i (T[i,j,k] - 1) * U[i,j,c] * W[k,i,c]
            # dE/dW[k,i,c] = 2 * sum_j (T[i,j,k] - 1) * U[i,j,c] * V[j,k,c]
            
            for c in range(r):
                # For dU
                # dU_r[i,j,c] = 2 * sum_k (T[i,j,k]-1) * V_r[j,k,c] * W_r[k,i,c]
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for k in range(p):
                            s += (T[i,j,k] - 1.0) * V_r[j,k,c] * W_r[k,i,c]
                        dU_r[i,j,c] = 2.0 * s
                
                # For dV
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for i in range(n):
                            s += (T[i,j,k] - 1.0) * U_r[i,j,c] * W_r[k,i,c]
                        dV_r[j,k,c] = 2.0 * s
                
                # For dW
                for k in range(p):
                    for i in range(n):
                        s = 0.0
                        for j in range(m):
                            s += (T[i,j,k] - 1.0) * U_r[i,j,c] * V_r[j,k,c]
                        dW_r[k,i,c] = 2.0 * s
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            if it % 100 == 99:
                lr *= 0.9
            
            # Early termination if not improving
            if it % 100 == 0 and err > best_err * 1.1 and it > 500:
                break
        
        # Check if we found a valid solution
        if best_err < 1e-6:
            # Snap to nearest half-integers and verify
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            
            if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), tol=1e-6):
                return U_snap, V_snap, W_snap
            
            # Try with tighter tolerance
            if verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-6):
                # Round to reasonable precision
                U_final = np.round(best_U, 8)
                V_final = np.round(best_V, 8)
                W_final = np.round(best_W, 8)
                if verify(U_final.tolist(), V_final.tolist(), W_final.tolist(), tol=1e-6):
                    return U_final, V_final, W_final
        
        return None

    # Try ranks from 24 upward
    for r in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in [42, 123, 777, 999, 2024]:
            if time.time() > deadline:
                break
            
            result = try_rank(r, max_iters=3000, seed=seed)
            if result is not None:
                U, V, W = result
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Final verification
                if verify(U_list, V_list, W_list, tol=1e-6):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END

