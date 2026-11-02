"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Try to find a decomposition with rank < 24
    # Known upper bound is 20 (14 for 2x3x3 + 6 for 2x3x1)
    # Try 20, 21, 22, 23
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        result = try_decompose(n, m, p, r, time_limit=10.0)
        if result:
            return result
            
    # Fallback to schoolbook rank 24
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def try_decompose(n, m, p, r, time_limit=10.0):
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eq = len(triples)
    
    # Precompute indices
    I_idx = [i * m + j for i, j, k in triples]
    J_idx = [j * p + k for i, j, k in triples]
    K_idx = [k * n + i for i, j, k in triples]
    
    np.random.seed(int(time.time() * 1000) % 10000)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # ALS Optimization
        for step in range(50):
            # Solve for U
            # M_U[e, I*e_c] = V[J_e, c] * W[K_e, c]
            M_U = np.zeros((num_eq, n * m * r))
            for e in range(num_eq):
                for c in range(r):
                    M_U[e, I_idx[e] * r + c] = V[J_idx[e], c] * W[K_idx[e], c]
            
            U_flat, _, _, _ = np.linalg.lstsq(M_U, np.ones(num_eq), rcond=None)
            U = U_flat.reshape(n * m, r)
            
            # Solve for V
            M_V = np.zeros((num_eq, m * p * r))
            for e in range(num_eq):
                for c in range(r):
                    M_V[e, J_idx[e] * r + c] = U[I_idx[e], c] * W[K_idx[e], c]
            
            V_flat, _, _, _ = np.linalg.lstsq(M_V, np.ones(num_eq), rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Solve for W
            M_W = np.zeros((num_eq, p * n * r))
            for e in range(num_eq):
                for c in range(r):
                    M_W[e, K_idx[e] * r + c] = U[I_idx[e], c] * V[J_idx[e], c]
            
            W_flat, _, _, _ = np.linalg.lstsq(M_W, np.ones(num_eq), rcond=None)
            W = W_flat.reshape(p * n, r)
            
            # Check error
            error = 0.0
            for e in range(num_eq):
                val = 0.0
                for c in range(r):
                    val += U[I_idx[e], c] * V[J_idx[e], c] * W[K_idx[e], c]
                error += (val - 1.0) ** 2
            
            if error < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                if verify(U_r, V_r, W_r, triples, I_idx, J_idx, K_idx):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
    return None

def verify(U, V, W, triples, I_idx, J_idx, K_idx):
    r = U.shape[1]
    for e, (i, j, k) in enumerate(triples):
        val = 0.0
        for c in range(r):
            val += U[I_idx[e], c] * V[J_idx[e], c] * W[K_idx[e], c]
        if abs(val - 1.0) > 1e-9:
            return False
    return True
# EVOLVE-BLOCK-END
