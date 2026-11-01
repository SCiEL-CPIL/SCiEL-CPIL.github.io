import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 150.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 60
    
    def compute_residual_sq(U, V, W):
        """Compute sum of squared residuals."""
        total = 0.0
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = V[k*p:(k+1)*p, :]
                w_val = W[k*n + i]
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                total += np.sum((mat - np.eye(m))**2)
        return total
    
    def check_exact(U, V, W):
        """Check if entries are half-integers and identity holds exactly."""
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not np.allclose(U2, np.round(U2), atol=1e-8):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-8):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-8):
            return False
        U_int = np.round(U2).astype(int)
        V_int = np.round(V2).astype(int)
        W_int = np.round(W2).astype(int)
        for i in range(n):
            for k in range(p):
                U_slice = U_int[i*m:(i+1)*m, :]
                V_slice = V_int[k*p:(k+1)*p, :]
                w_val = W_int[k*n + i]
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.allclose(mat, 4 * np.eye(m), atol=1e-8):
                    return False
        return True
    
    def try_rank(rank):
        """Try to find a valid decomposition at given rank."""
        nonlocal best_U, best_V, best_W, best_rank
        if rank >= best_rank:
            return False
        
        # Multiple strategies
        strategies = ['random_integer', 'random_half', 'structured']
        
        for strategy in strategies:
            if time.time() - start_time > DEADLINE - 5:
                return False
            
            max_iters = 500
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE - 5:
                    return False
                
                if strategy == 'random_integer':
                    U = np.random.randint(-3, 4, (nm, rank)) * 0.5
                    V = np.random.randint(-3, 4, (mp, rank)) * 0.5
                    W = np.random.randint(-3, 4, (pn, rank)) * 0.5
                elif strategy == 'random_half':
                    U = np.random.randn(nm, rank) * 0.5
                    V = np.random.randn(mp, rank) * 0.5
                    W = np.random.randn(pn, rank) * 0.5
                else:  # structured
                    U = np.zeros((nm, rank))
                    V = np.zeros((mp, rank))
                    W = np.zeros((pn, rank))
                    # Initialize with sparse random pattern
                    for r in range(rank):
                        # Pick random i, j for U
                        idx = np.random.randint(0, nm)
                        U[idx, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                        idx2 = np.random.randint(0, mp)
                        V[idx2, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                        idx3 = np.random.randint(0, pn)
                        W[idx3, r] = random.choice([-1, 1]) * random.choice([0.5, 1.0])
                
                # Local search: perturb random entries
                for step in range(200):
                    if time.time() - start_time > DEADLINE - 5:
                        return False
                    
                    # Pick random entry to modify
                    mat = random.choice([U, V, W])
                    row = np.random.randint(0, mat.shape[0])
                    col = np.random.randint(0, mat.shape[1])
                    
                    old_val = mat[row, col]
                    # Perturb by small amount
                    delta = random.choice([-1.0, -0.5, 0.5, 1.0])
                    mat[row, col] = old_val + delta
                    
                    # Round to nearest half-integer
                    mat[row, col] = round(mat[row, col] * 2) / 2.0
                    
                    # Check if improved
                    res = compute_residual_sq(U, V, W)
                    if res < 1e-6:
                        if check_exact(U, V, W):
                            best_rank = rank
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            return True
                    elif res > 1.0:
                        # Revert if too bad
                        mat[row, col] = old_val
                
                # Periodically try to snap to half-integers and check
                if it % 50 == 0:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if check_exact(U_rounded, V_rounded, W_rounded):
                        best_rank = rank
                        best_U = U_rounded.copy()
                        best_V = V_rounded.copy()
                        best_W = W_rounded.copy()
                        return True
        
        return False
    
    # Try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = 30, but also consider other bounds
    # For matrix multiplication, M(n,m,p) >= max(n*m, m*p, p*n) is not always tight
    # Use 30 as starting point
    min_rank = max(nm, mp, pn)
    
    for rank in range(min_rank, best_rank):
        if time.time() - start_time > DEADLINE - 5:
            break
        if try_rank(rank):
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
