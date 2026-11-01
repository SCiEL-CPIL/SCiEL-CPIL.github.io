import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (U_index, coeff)
    # v_coeffs: list of (V_index_offset, coeff)
    # w_coeffs: list of (W_index_offset, coeff)
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        # P1 = (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # P2 = A00 * (B01 - B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # P3 = A11 * (B10 - B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        # P4 = (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    # Fill in 4 Strassen blocks for columns 0-7
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff

    # Schoolbook terms for column 8 (2x2x1 block)
    # V indices: V[j*p + k], so for k=8: V[8] (j=0), V[17] (j=1)
    # W indices: W[k*n + i], so for k=8: W[16] (i=0), W[17] (i=1)
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ],  [ (16, 1) ]),   # A00 * B0 -> C00
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),   # A01 * B1 -> C00
        ([ (2, 1) ], [ (8, 1) ],  [ (17, 1) ]),   # A10 * B0 -> C10
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),   # A11 * B1 -> C10
    ]

    col_start = 4 * 7  # column 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff

        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff

        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff

    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)

def _try_randomized_search(n, m, p, target_rank, max_time=10.0):
    """
    Use randomized gradient descent to find a decomposition of target_rank.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Target Tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Flattened target for easier gradient computation? 
    # Let's stick to 3D for clarity, but flatten for error calc.
    T_flat = T.flatten()
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try a few random initializations
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with small random values
        # Scaling helps with gradient flow
        scale = 0.1
        U = np.random.randn(nU, target_rank) * scale
        V = np.random.randn(nV, target_rank) * scale
        W = np.random.randn(nW, target_rank) * scale
        
        # Simple SGD
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        # Adaptive learning rate or fixed? Let's try a simple decay.
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            # Compute T_recon
            # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # We can compute this efficiently.
            # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # To save time, we might approximate or just do it directly.
            # 4x18x18 tensor is small.
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            Error = T_recon - T
            err_val = np.sum(Error**2)
            
            if err_val < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
            
            if err_val < best_error:
                best_error = err_val
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # Compute Gradients
            # dE/dU[i,r] = 2 * sum_{j,k} Error[i,j,k] * V[j,r] * W[k,r]
            # dE/dV[j,r] = 2 * sum_{i,k} Error[i,j,k] * U[i,r] * W[k,r]
            # dE/dW[k,r] = 2 * sum_{i,j} Error[i,j,k] * U[i,r] * V[j,r]
            
            # Using einsum for gradients
            # gU[i,r] = 2 * einsum('ijk,jr,kr->ir', Error, V, W)
            # gV[j,r] = 2 * einsum('ijk,ir,kr->jr', Error, U, W)
            # gW[k,r] = 2 * einsum('ijk,ir,jr->kr', Error, U, V)
            
            gU = 2 * np.einsum('ijk,jr,kr->ir', Error, V, W)
            gV = 2 * np.einsum('ijk,ir,kr->jr', Error, U, W)
            gW = 2 * np.einsum('ijk,ir,jr->kr', Error, U, V)
            
            # Momentum Update
            vU = momentum * vU + lr * gU
            vV = momentum * vV + lr * gV
            vW = momentum * vW + lr * gW
            
            U -= vU
            V -= vV
            W -= vW
            
            # Decay learning rate
            lr *= 0.999
            
        # End steps
        
        # Check if we found a solution
        if best_error < 1e-10:
             return best_U.tolist(), best_V.tolist(), best_W.tolist()
             
    # If we didn't find exact, check best
    if best_error < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
    return None

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME

    # First, ensure we have a valid rank-32 solution as fallback
    U32, V32, W32 = _build_rank_32(n, m, p)
    if not _validate(U32, V32, W32, n, m, p):
        # Fallback to schoolbook rank 36 if rank 32 construction failed (shouldn't happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find rank 31 via randomized search
    # Allocate ~30s for this
    time_budget = min(30.0, _TIMEOUT - elapsed - 10.0)
    if time_budget > 5.0:
        res = _try_randomized_search(n, m, p, 31, max_time=time_budget)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try to find rank 30 via randomized search if time permits
    elapsed = time.time() - _START_TIME
    time_budget = min(20.0, _TIMEOUT - elapsed - 10.0)
    if time_budget > 5.0:
        res = _try_randomized_search(n, m, p, 30, max_time=time_budget)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Try to find rank 29 via randomized search if time permits
    elapsed = time.time() - _START_TIME
    time_budget = min(15.0, _TIMEOUT - elapsed - 10.0)
    if time_budget > 5.0:
        res = _try_randomized_search(n, m, p, 29, max_time=time_budget)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank-32
    return {"U": U32, "V": V32, "W": W32}
