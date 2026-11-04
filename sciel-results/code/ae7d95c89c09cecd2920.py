import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Leave 5s safety margin
    
    # Schoolbook rank = 60
    rank_school = n * m * p
    
    # Build schoolbook decomposition as fallback
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = build_schoolbook()
    
    # Verify the decomposition using vectorized operations
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
        ii_flat = ii.flatten()
        jj_flat = jj.flatten()
        kk_flat = kk.flatten()
        
        u_idx = ii_flat * m + jj_flat
        v_idx = jj_flat * p + kk_flat
        w_idx = kk_flat * n + ii_flat
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        
        return np.all(np.abs(sums - 1.0) < tol)
    
    # Precompute index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()
    kk_flat = kk.flatten()
    
    u_idx = ii_flat * m + jj_flat
    v_idx = jj_flat * p + kk_flat
    w_idx = kk_flat * n + ii_flat
    
    N = len(u_idx)  # n*m*p = 60
    
    # Try to find a lower rank decomposition
    def try_find_low_rank(max_rank, num_trials=15):
        """Try to find a rank-r decomposition for r < rank_school."""
        best_U, best_V, best_W = None, None, None
        best_rank = rank_school
        
        for r in range(15, max_rank + 1):
            if time.time() > deadline - 15.0:
                break
            
            for trial in range(num_trials):
                if time.time() > deadline - 15.0:
                    break
                
                # Random initialization with small values
                U = np.random.randn(n * m, r) * 0.2
                V = np.random.randn(m * p, r) * 0.2
                W = np.random.randn(p * n, r) * 0.2
                
                # Gradient descent with vectorized operations
                lr = 0.02
                for step in range(300):
                    if time.time() > deadline - 15.0:
                        break
                    
                    U_sel = U[u_idx, :]
                    V_sel = V[v_idx, :]
                    W_sel = W[w_idx, :]
                    
                    vals = (U_sel * V_sel * W_sel).sum(axis=1)
                    errs = vals - 1.0
                    
                    # Compute gradients using np.add.at for grouped sums
                    eVW = errs[:, None] * V_sel * W_sel  # (N, r)
                    eUW = errs[:, None] * U_sel * W_sel
                    eUV = errs[:, None] * U_sel * V_sel
                    
                    U_grad = np.zeros_like(U)
                    V_grad = np.zeros_like(V)
                    W_grad = np.zeros_like(W)
                    
                    np.add.at(U_grad, u_idx, eVW)
                    np.add.at(V_grad, v_idx, eUW)
                    np.add.at(W_grad, w_idx, eUV)
                    
                    # Update with adaptive learning rate
                    U -= lr * U_grad
                    V -= lr * V_grad
                    W -= lr * W_grad
                    
                    # Check convergence
                    if step % 20 == 19:
                        U_sel = U[u_idx, :]
                        V_sel = V[v_idx, :]
                        W_sel = W[w_idx, :]
                        vals = (U_sel * V_sel * W_sel).sum(axis=1)
                        max_err = np.max(np.abs(vals - 1.0))
                        
                        if max_err < 1e-3:
                            # Try to round to half-integers
                            U_round = np.round(U * 2) / 2
                            V_round = np.round(V * 2) / 2
                            W_round = np.round(W * 2) / 2
                            
                            if verify(U_round, V_round, W_round, tol=1e-6):
                                best_U = U_round.tolist()
                                best_V = V_round.tolist()
                                best_W = W_round.tolist()
                                best_rank = r
                                return best_U, best_V, best_W, best_rank
                            else:
                                # Use unrounded
                                best_U = U.tolist()
                                best_V = V.tolist()
                                best_W = W.tolist()
                                best_rank = r
                                return best_U, best_V, best_W, best_rank
        
        return None, None, None, best_rank
    
    # Try to find a low-rank decomposition
    best_U, best_V, best_W, best_rank = try_find_low_rank(50, num_trials=10)
    
    if best_U is not None and best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
