"""Matrix multiplication 3x4x4 tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n=3, m=4, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=4):
    """Faster verification using tensor contraction."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid_set:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-8:
                        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 48
    
    # Schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0]*best_rank for _ in range(nm)]
    best_V = [[0]*best_rank for _ in range(mp)]
    best_W = [[0]*best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Precompute target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    def compute_residual_fast(U_arr, V_arr, W_arr, r):
        """Compute residual using efficient tensor operations."""
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        residual = 0.0
        # Valid entries should be 1
        for (a, b, c) in target_indices:
            residual += (T_comp[a, b, c] - 1.0) ** 2
        
        # Forbidden entries should be 0
        total_sq = np.sum(T_comp ** 2)
        valid_sq = 0.0
        for (a, b, c) in target_indices:
            valid_sq += T_comp[a, b, c] ** 2
        residual += (total_sq - valid_sq)
        
        return residual
    
    def gradient_step(U, V, W, r, lr):
        """One gradient descent step with efficient computation."""
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        
        T_target = np.zeros((nm, mp, pn))
        for (a, b, c) in target_indices:
            T_target[a, b, c] = 1.0
        
        diff = T_comp - T_target
        
        # dU[a, r'] = 2 * sum_{b,c} diff[a,b,c] * V[b,r'] * W[c,r']
        dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        
        # dV[b, r'] = 2 * sum_{a,c} diff[a,b,c] * U[a,r'] * W[c,r']
        dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        
        # dW[c, r'] = 2 * sum_{a,b} diff[a,b,c] * U[a,r'] * V[b,r']
        dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, np.sum(diff ** 2)
    
    def structured_init(r, seed):
        """Initialize exploiting the 3x4x4 row structure."""
        rng = random.Random(seed)
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Block structure: 4 columns per row, then shared columns
        for i in range(n):
            base_col = i * 4
            for j in range(m):
                for k in range(p):
                    if base_col + j < r:
                        col_idx = base_col + j
                        U[i*m + j, col_idx] = rng.gauss(0, 0.5)
                        V[j*p + k, col_idx] = rng.gauss(0, 0.5)
                        W[k*n + i, col_idx] = rng.gauss(0, 0.5)
        
        # Shared random columns
        for col in range(n*4, r):
            for a in range(nm):
                if rng.random() < 0.3:
                    U[a, col] = rng.gauss(0, 0.3)
            for b in range(mp):
                if rng.random() < 0.3:
                    V[b, col] = rng.gauss(0, 0.3)
            for c in range(pn):
                if rng.random() < 0.3:
                    W[c, col] = rng.gauss(0, 0.3)
        
        return U, V, W
    
    def random_init(r, seed):
        """Simple random initialization."""
        rng = random.Random(seed)
        U = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(nm)])
        V = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(mp)])
        W = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(pn)])
        return U, V, W
    
    def optimize(U, V, W, r, max_iter=2000, initial_lr=0.01):
        """Optimize with adaptive learning rate."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        lr = initial_lr
        best_res = np.inf
        stall_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            U, V, W, res = gradient_step(U, V, W, r, lr)
            
            if res < best_res:
                best_res = res
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 50:
                    lr *= 0.9
                    stall_count = 0
            
            if res < 1e-12:
                break
        
        return U, V, W, best_res
    
    def polish(U, V, W, r, max_iter=1000):
        """High-precision polish with very small learning rate."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        lr = 1e-4
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            U, V, W, res = gradient_step(U, V, W, r, lr)
            
            if res < 1e-15:
                break
        
        return U, V, W
    
    best_result = None
    
    for r in range(24, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_attempts = 5 if r <= 32 else 3
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            seed = 1000 + r * 100 + attempt
            
            if attempt < 2:
                U, V, W = structured_init(r, seed)
            else:
                U, V, W = random_init(r, seed)
            
            U, V, W, res = optimize(U, V, W, r, max_iter=1500, initial_lr=0.01)
            
            if time.time() > deadline - 3:
                break
            
            if res < 1e-8:
                U, V, W = polish(U, V, W, r, max_iter=500)
                
                if _verify_fast(U, V, W):
                    best_result = (r, U.tolist(), V.tolist(), W.tolist())
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        r, U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": best_U, "V": best_V, "W": best_W}
