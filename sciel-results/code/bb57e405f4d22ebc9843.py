"""Tensor decomposition search for 2x3 x 3x3 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_sb
    best_solution = (U_sb, V_sb, W_sb)
    
    # Precompute index mappings for speed
    # For each (i,j,k): u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    def compute_residual(U, V, W):
        """Compute residual tensor R[i,j,k] = sum_c U[u,c]*V[v,c]*W[w,c] - 1"""
        R = np.ones((n, m, p)) * -1.0  # start at -1, add the sum
        for (ui, vi, wi) in indices:
            # R[i,j,k] += dot(U[ui,:], V[vi,:]*W[wi,:])
            # We need to map back to (i,j,k)
            pass
        # Better: build R directly
        R = np.zeros((n, m, p))
        for idx, (ui, vi, wi) in enumerate(indices):
            i = idx // (m * p)
            j = (idx // p) % m
            k = idx % p
            R[i, j, k] = np.dot(U[ui, :], V[vi, :] * W[wi, :])
        R -= 1.0
        return R
    
    def try_rank(rank, max_iters=3000, lr_init=0.005, n_restarts=30):
        """Try to find decomposition of given rank via gradient descent."""
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(restart * 7919 + rank * 13)
            U = np.random.randn(n * m, rank) * 0.3
            V = np.random.randn(m * p, rank) * 0.3
            W = np.random.randn(p * n, rank) * 0.3
            
            lr = lr_init
            for step in range(max_iters):
                if time.time() > deadline - 20:
                    return None
                
                R = compute_residual(U, V, W)
                err = np.max(np.abs(R))
                
                if err < 1e-7:
                    # Snap to half-integers
                    U_s = np.round(U * 2) / 2
                    V_s = np.round(V * 2) / 2
                    W_s = np.round(W * 2) / 2
                    R_s = compute_residual(U_s, V_s, W_s)
                    if np.max(np.abs(R_s)) < 1e-9:
                        return (U_s, V_s, W_s)
                    # Also try integer snap
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    R_i = compute_residual(U_i, V_i, W_i)
                    if np.max(np.abs(R_i)) < 1e-9:
                        return (U_i, V_i, W_i)
                    break
                
                # Compute gradients (vectorized per component)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for idx, (ui, vi, wi) in enumerate(indices):
                    r = R.flat[idx]
                    if abs(r) < 1e-10:
                        continue
                    # dU[ui,c] += r * V[vi,c] * W[wi,c]
                    dU[ui, :] += r * (V[vi, :] * W[wi, :])
                    # dV[vi,c] += r * U[ui,c] * W[wi,c]
                    dV[vi, :] += r * (U[ui, :] * W[wi, :])
                    # dW[wi,c] += r * U[ui,c] * V[vi,c]
                    dW[wi, :] += r * (U[ui, :] * V[vi, :])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if step % 200 == 0 and step > 0:
                    lr *= 0.9
                    if lr < 1e-5:
                        lr = 1e-5
                
                # Occasionally try to snap and check
                if step % 500 == 0:
                    U_s = np.round(U * 2) / 2
                    V_s = np.round(V * 2) / 2
                    W_s = np.round(W * 2) / 2
                    R_s = compute_residual(U_s, V_s, W_s)
                    if np.max(np.abs(R_s)) < 1e-9:
                        return (U_s, V_s, W_s)
        
        return None
    
    # Try ranks from low to high
    for rank in range(9, 18):
        if time.time() > deadline - 30:
            break
        
        sol = try_rank(rank)
        if sol is not None:
            best_rank = rank
            best_solution = sol
            # Don't break - try to find even lower rank
            # Actually we're going from low to high, so first found is best
            break
    
    # Convert to list format
    U, V, W = best_solution
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                v = round(val * 2) / 2.0
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
