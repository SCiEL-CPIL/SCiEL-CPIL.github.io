"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s to be safe under 150s limit
    
    def rank54_decomp():
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P2
                U[1*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                W[k1*n + 1, col] -= 1
                col += 1
                # P3
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] -= 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P4
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] -= 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                # P5
                U[0*m + j0, col] += 1
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] -= 1
                W[k1*n + 0, col] += 1
                col += 1
                # P6
                U[1*m + j0, col] += 1
                U[0*m + j0, col] -= 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P7
                U[0*m + j1, col] += 1
                U[1*m + j1, col] -= 1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Vectorized verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def compute_residual(U, V, W):
        """Compute sum of squared residuals using vectorized ops."""
        rank = U.shape[1]
        resid = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    resid += (val - 1.0)**2
        return resid
    
    def gradient_step(U, V, W, lr):
        """Compute and apply gradient step for tensor decomposition."""
        rank = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0
                    # dU[i*m+j, t] += residual * V[j*p+k, t] * W[k*n+i, t]
                    dU[i*m+j, :] += residual * V[j*p+k, :] * W[k*n+i, :]
                    # dV[j*p+k, t] += residual * U[i*m+j, t] * W[k*n+i, t]
                    dV[j*p+k, :] += residual * U[i*m+j, :] * W[k*n+i, :]
                    # dW[k*n+i, t] += residual * U[i*m+j, t] * V[j*p+k, t]
                    dW[k*n+i, :] += residual * U[i*m+j, :] * V[j*p+k, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        return U, V, W
    
    # Get base rank-54 solution
    best_result = rank54_decomp()
    U_base = np.array(best_result["U"], dtype=float)
    V_base = np.array(best_result["V"], dtype=float)
    W_base = np.array(best_result["W"], dtype=float)
    
    # Try to find lower rank solutions
    found_lower = False
    for target_rank in [53, 52]:
        if time.time() > deadline - 40:
            break
        found = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            try:
                rng = np.random.RandomState(42 + attempt * 7)
                # Initialize with rank54 truncated and perturbed
                U = U_base[:, :target_rank].copy()
                V = V_base[:, :target_rank].copy()
                W = W_base[:, :target_rank].copy()
                
                # Add random perturbation
                U += rng.randn(n*m, target_rank) * 0.05
                V += rng.randn(m*p, target_rank) * 0.05
                W += rng.randn(p*n, target_rank) * 0.05
                
                # Iterative refinement with adaptive learning rate
                lr = 0.005
                best_resid = compute_residual(U, V, W)
                for it in range(500):
                    if time.time() > deadline - 15:
                        break
                    U, V, W = gradient_step(U, V, W, lr)
                    resid = compute_residual(U, V, W)
                    if resid < 1e-8:
                        found = True
                        break
                    if resid < best_resid:
                        best_resid = resid
                        # Occasionally reduce learning rate
                        if it % 50 == 0 and it > 0:
                            lr *= 0.9
                    else:
                        # Revert if worse
                        lr *= 0.5
                if found:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found_lower = True
                    break
            except Exception:
                continue
        if found_lower:
            break
    
    # Final verification of the best result
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        # Fall back to base if verification fails
        best_result = rank54_decomp()
    
    return best_result
# EVOLVE-BLOCK-END
