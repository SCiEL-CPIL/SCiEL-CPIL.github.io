"""Matrix multiplication 4x4x6 tensor decomposition search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    # Schoolbook decomposition
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def compute_tensor_error(U, V, W, rank):
        """Compute sum of squared errors for the tensor decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (1.0 - s) ** 2
        return err
    
    def try_optimize(rank, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition via gradient-based optimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        err = compute_tensor_error(U, V, W, rank)
        best_err = err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Learning rate schedule
        lr = 0.01
        patience = 0
        max_patience = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients via finite differences or analytic
            # For efficiency, compute gradients analytically
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        grad_factor = -2.0 * (1.0 - s)
                        for c in range(rank):
                            dU[i*m+j, c] += grad_factor * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += grad_factor * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += grad_factor * U[i*m+j, c] * V[j*p+k, c]
            
            # Normalize gradients to prevent explosion
            dU_norm = np.sqrt(np.mean(dU**2))
            dV_norm = np.sqrt(np.mean(dV**2))
            dW_norm = np.sqrt(np.mean(dW**2))
            max_norm = max(dU_norm, dV_norm, dW_norm, 1e-10)
            dU /= max_norm
            dV /= max_norm
            dW /= max_norm
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            new_err = compute_tensor_error(U, V, W, rank)
            
            if new_err < err:
                err = new_err
                patience = 0
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    # Adaptive learning rate
                    lr = min(lr * 1.05, 0.1)
                if err < 1e-8:
                    break
            else:
                # Revert
                U += lr * dU
                V += lr * dV
                W += lr * dW
                patience += 1
                lr = max(lr * 0.95, 0.001)
                if patience > max_patience:
                    break
        
        if best_err < 1e-6:
            U_list = [[round(x, 10) for x in row] for row in best_U.tolist()]
            V_list = [[round(x, 10) for x in row] for row in best_V.tolist()]
            W_list = [[round(x, 10) for x in row] for row in best_W.tolist()]
            return U_list, V_list, W_list, best_err
        return None, None, None, best_err
    
    def try_local_search(rank, max_iters=3000, seed=42):
        """Local search with single-element perturbations."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        err = compute_tensor_error(U, V, W, rank)
        best_err = err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        step_size = 0.05
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            mat = rng.randint(3)
            if mat == 0:
                r_idx = rng.randint(n * m)
                c_idx = rng.randint(rank)
                old_val = U[r_idx, c_idx]
                U[r_idx, c_idx] = old_val + rng.randn() * step_size
            elif mat == 1:
                r_idx = rng.randint(m * p)
                c_idx = rng.randint(rank)
                old_val = V[r_idx, c_idx]
                V[r_idx, c_idx] = old_val + rng.randn() * step_size
            else:
                r_idx = rng.randint(p * n)
                c_idx = rng.randint(rank)
                old_val = W[r_idx, c_idx]
                W[r_idx, c_idx] = old_val + rng.randn() * step_size
            
            new_err = compute_tensor_error(U, V, W, rank)
            if new_err < err:
                err = new_err
                step_size = min(step_size * 1.1, 0.2)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                if err < 1e-8:
                    break
            else:
                if mat == 0:
                    U[r_idx, c_idx] = old_val
                elif mat == 1:
                    V[r_idx, c_idx] = old_val
                else:
                    W[r_idx, c_idx] = old_val
                step_size = max(step_size * 0.9, 0.01)
        
        if best_err < 1e-6:
            U_list = [[round(x, 10) for x in row] for row in best_U.tolist()]
            V_list = [[round(x, 10) for x in row] for row in best_V.tolist()]
            W_list = [[round(x, 10) for x in row] for row in best_W.tolist()]
            return U_list, V_list, W_list, best_err
        return None, None, None, best_err
    
    # Try ranks from 95 down to 70
    ranks_to_try = list(range(95, 69, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple seeds for gradient-based optimization
        for seed in [r, r + 100, r + 200]:
            if time.time() > deadline - 10:
                break
            U_list, V_list, W_list, err = try_optimize(r, max_iters=3000, seed=seed)
            if U_list is not None:
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
        
        # Try local search as backup
        if time.time() > deadline - 10:
            break
        U_list, V_list, W_list, err = try_local_search(r, max_iters=2000, seed=r)
        if U_list is not None:
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
