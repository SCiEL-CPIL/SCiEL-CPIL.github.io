import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Leave 10s margin

    # Precompute the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W, R):
        """Verify exact decomposition with integer/half-integer entries."""
        for arr in [U, V, W]:
            for val in arr:
                if not (val * 2 == int(val * 2)):
                    return False
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        if not np.allclose(T_rec, T, atol=1e-10):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(R):
                        s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if s != 1:
                        return False
        return True

    # Build schoolbook as fallback
    best_rank = n * m * p
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school, best_rank)

    # Try a structured approach: exploit the block structure
    # For each j, we have a 2x6x12 sub-tensor
    # The sub-tensor for fixed j has entries at [i, k, k*2+i] for i in [0,2), k in [0,6)
    
    # Try to find a decomposition where the rank is reduced by sharing terms across j values
    
    # Approach: Use the fact that the tensor can be viewed as a sum of m=3 "blocks"
    # Each block j has a specific structure
    
    # Let's try to find a decomposition with rank R where R < 36
    # We'll use a heuristic search
    
    def search_decomposition(R):
        """Search for a rank-R decomposition using local search."""
        if time.time() - start_time > timeout:
            return None
        
        import random
        random.seed(42)
        
        # Start from the schoolbook decomposition and try to merge terms
        # or find a better structure
        
        # Try different initializations
        for init_type in ['random', 'structured']:
            if time.time() - start_time > timeout:
                break
            
            if init_type == 'random':
                U = np.zeros((n * m, R), dtype=np.int64)
                V = np.zeros((m * p, R), dtype=np.int64)
                W = np.zeros((p * n, R), dtype=np.int64)
                
                for r in range(R):
                    # Random sparse initialization
                    for _ in range(3):
                        U[random.randint(0, n * m - 1), r] = random.choice([-1, 1])
                    for _ in range(3):
                        V[random.randint(0, m * p - 1), r] = random.choice([-1, 1])
                    for _ in range(3):
                        W[random.randint(0, p * n - 1), r] = random.choice([-1, 1])
            else:
                # Structured initialization based on the tensor pattern
                U = np.zeros((n * m, R), dtype=np.int64)
                V = np.zeros((m * p, R), dtype=np.int64)
                W = np.zeros((p * n, R), dtype=np.int64)
                
                # Try to assign terms based on the (i, j, k) structure
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    if r < n * m * p:
                        U[i * m + j, r] = 1
                        V[j * p + k, r] = 1
                        W[k * n + i, r] = 1
            
            # Local search: try to improve the decomposition
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Compute current error
                T_rec = np.einsum('ar,br,cr->abc', U.astype(np.float64), V.astype(np.float64), W.astype(np.float64))
                
                target_positions = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
                target_set = set(target_positions)
                
                error = 0
                for a, b, c in target_positions:
                    if T_rec[a, b, c] != 1:
                        error += 1
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            if (a, b, c) not in target_set and T_rec[a, b, c] != 0:
                                error += 1
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if error == 0:
                        U_list = [[int(U[a, r]) for r in range(R)] for a in range(n * m)]
                        V_list = [[int(V[b, r]) for r in range(R)] for b in range(m * p)]
                        W_list = [[int(W[c, r]) for r in range(R)] for c in range(p * n)]
                        return (U_list, V_list, W_list, R)
                
                # Try a random perturbation
                if random.random() < 0.3:
                    r = random.randint(0, R - 1)
                    U[random.randint(0, n * m - 1), r] = random.choice([-1, 0, 1])
                elif random.random() < 0.3:
                    r = random.randint(0, R - 1)
                    V[random.randint(0, m * p - 1), r] = random.choice([-1, 0, 1])
                else:
                    r = random.randint(0, R - 1)
                    W[random.randint(0, p * n - 1), r] = random.choice([-1, 0, 1])
        
        return None
    
    # Try ranks from 20 to 35
    for R in range(20, 36):
        if time.time() - start_time > timeout:
            break
        
        result = search_decomposition(R)
        if result is not None:
            U, V, W, rank = result
            if verify(U, V, W, rank):
                best_decomp = (U, V, W, rank)
                break
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
