import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Precompute index mappings
    # u = i*m + j, v = j*p + k, w = k*n + i
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            eqs = [(j * p + k, k * n + i) for k in range(p)]
            U_indices.append((i, j, u, eqs))
            
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = [(i * m + j, k * n + i) for i in range(n)]
            V_indices.append((j, k, v, eqs))
            
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = [(i * m + j, j * p + k) for j in range(m)]
            W_indices.append((k, i, w, eqs))
    
    def verify_solution(U, V, W):
        """Verify exact decomposition with half-integer entries"""
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True
    
    def als_decompose(R, max_iters=500, tol=1e-12, init_type='random', seed=42):
        """ALS decomposition with various initialization strategies"""
        np.random.seed(seed)
        
        if init_type == 'structured':
            # Winograd-inspired structured initialization
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Pattern 1: Diagonal-like assignments
            for r in range(R):
                # Cycle through positions with structured offsets
                i_idx = r % n
                j_idx = (r // n) % m
                k_idx = (r // (n * m)) % p
                
                U[i_idx * m + j_idx, r] = np.random.choice([-1.0, 1.0]) * 0.5
                V[j_idx * p + k_idx, r] = np.random.choice([-1.0, 1.0]) * 0.5
                W[k_idx * n + i_idx, r] = np.random.choice([-1.0, 1.0]) * 0.5
                
                # Add cross terms for better coverage
                if r < R // 2:
                    i2 = (i_idx + 1) % n
                    j2 = (j_idx + 1) % m
                    k2 = (k_idx + 1) % p
                    U[i2 * m + j2, r] += np.random.choice([-0.5, 0.5])
                    V[j2 * p + k2, r] += np.random.choice([-0.5, 0.5])
                    W[k2 * n + i2, r] += np.random.choice([-0.5, 0.5])
                    
        elif init_type == 'sparse':
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            for r in range(R):
                for _ in range(3):  # More non-zeros per column
                    U[np.random.randint(n * m), r] = np.random.choice([-1, 1]) * 0.5
                for _ in range(3):
                    V[np.random.randint(m * p), r] = np.random.choice([-1, 1]) * 0.5
                for _ in range(3):
                    W[np.random.randint(p * n), r] = np.random.choice([-1, 1]) * 0.5
                    
        elif init_type == 'gaussian':
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
        else:  # random
            U = np.random.randint(-2, 3, (n * m, R)) * 0.5
            V = np.random.randint(-2, 3, (m * p, R)) * 0.5
            W = np.random.randint(-2, 3, (p * n, R)) * 0.5
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.array([V[v, :] * W[w, :] for v, w in eqs])
                if B.shape[0] >= R:
                    U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=1e-12)
                    U[u, :] = U_u
                else:
                    U_u, _, _, _ = np.linalg.lstsq(B.T, np.ones(len(eqs)), rcond=1e-12)
                    U[u, :] = U_u
                    
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.array([U[u, :] * W[w, :] for u, w in eqs])
                if C.shape[0] >= R:
                    V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=1e-12)
                    V[v, :] = V_v
                else:
                    V_v, _, _, _ = np.linalg.lstsq(C.T, np.ones(len(eqs)), rcond=1e-12)
                    V[v, :] = V_v
                    
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.array([U[u, :] * V[v, :] for u, v in eqs])
                if D.shape[0] >= R:
                    W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=1e-12)
                    W[w, :] = W_w
                else:
                    W_w, _, _, _ = np.linalg.lstsq(D.T, np.ones(len(eqs)), rcond=1e-12)
                    W[w, :] = W_w
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
        
        # Polish pass with tighter tolerance
        U_polish = U.copy()
        V_polish = V.copy()
        W_polish = W.copy()
        
        for it in range(200):
            if time.time() > deadline - 5:
                break
                
            for i, j, u, eqs in U_indices:
                B = np.array([V_polish[v, :] * W_polish[w, :] for v, w in eqs])
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=1e-14)
                U_polish[u, :] = U_u
                
            for j, k, v, eqs in V_indices:
                C = np.array([U_polish[u, :] * W_polish[w, :] for u, w in eqs])
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=1e-14)
                V_polish[v, :] = V_v
                
            for k, i, w, eqs in W_indices:
                D = np.array([U_polish[u, :] * V_polish[v, :] for u, v in eqs])
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=1e-14)
                W_polish[w, :] = W_w
        
        U_round = np.round(U_polish * 2) / 2
        V_round = np.round(V_polish * 2) / 2
        W_round = np.round(W_polish * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None
    
    def try_direct_search(R, max_attempts=200):
        """Direct combinatorial search for small ranks"""
        if R > 20:  # Too large for direct search
            return None
            
        half_int_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(attempt * 7 + R * 13)
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Random sparse initialization with half-integers
            for r in range(R):
                # Each column has 2-4 non-zero entries per matrix
                num_u = np.random.randint(2, 5)
                num_v = np.random.randint(2, 5)
                num_w = np.random.randint(2, 5)
                
                u_positions = np.random.choice(n * m, size=num_u, replace=False)
                v_positions = np.random.choice(m * p, size=num_v, replace=False)
                w_positions = np.random.choice(p * n, size=num_w, replace=False)
                
                for pos in u_positions:
                    U[pos, r] = np.random.choice(half_int_values)
                for pos in v_positions:
                    V[pos, r] = np.random.choice(half_int_values)
                for pos in w_positions:
                    W[pos, r] = np.random.choice(half_int_values)
            
            if verify_solution(U, V, W):
                return U, V, W
                
        return None
    
    # Try ranks from low to high
    for R in [14, 15, 16, 17]:
        if time.time() > deadline - 20:
            break
            
        # Try direct search first for small ranks
        if R <= 16:
            result = try_direct_search(R, max_attempts=100)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try ALS with various initializations
        init_types = ['structured', 'sparse', 'gaussian', 'random']
        max_restarts = 30 if R <= 15 else 20
        
        for init_type in init_types:
            if time.time() > deadline - 10:
                break
                
            for restart in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                    
                seed = restart * 1000 + R * 10 + init_types.index(init_type)
                res = als_decompose(R, max_iters=300, init_type=init_type, seed=seed)
                if res is not None:
                    U, V, W = res
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
