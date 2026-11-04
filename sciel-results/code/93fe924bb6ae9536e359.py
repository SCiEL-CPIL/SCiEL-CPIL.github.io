import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    # Precompute tensor indices
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i * m + j
                J_idx = j * p + k
                K_idx = k * n + i
                T_indices.append((I_idx, J_idx, K_idx))
    
    def verify_solution(U, V, W, R):
        """Verify exact decomposition with half-integer entries."""
        U_int = np.round(np.array(U) * 2).astype(int)
        V_int = np.round(np.array(V) * 2).astype(int)
        W_int = np.round(np.array(W) * 2).astype(int)
        
        # Check: sum_r U_int[I,r] * V_int[J,r] * W_int[K,r] == 8 for non-zeros, 0 elsewhere
        T_approx = np.einsum('ir, jr, kr -> ijk', U_int, V_int, W_int)
        
        # Build expected tensor (scaled by 8)
        T_full = np.zeros((dim_U, dim_V, dim_W), dtype=int)
        for (I_idx, J_idx, K_idx) in T_indices:
            T_full[I_idx, J_idx, K_idx] = 8
            
        return np.array_equal(T_approx, T_full)
    
    def schoolbook_decomposition():
        """Return the rank-48 schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    def try_als_with_integer_constraint(R, max_iters, time_limit, seed=42):
        """
        ALS with periodic rounding to half-integers and verification.
        Uses structured initialization based on schoolbook.
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Initialize with schoolbook-like structure plus small perturbations
        # Start from a rank-R sub-selection of schoolbook columns
        U0, V0, W0 = schoolbook_decomposition()
        U0 = np.array(U0, dtype=float)
        V0 = np.array(V0, dtype=float)
        W0 = np.array(W0, dtype=float)
        
        if R <= n * m * p:
            # Take first R columns of schoolbook
            U = U0[:, :R].copy()
            V = V0[:, :R].copy()
            W = W0[:, :R].copy()
        else:
            U = np.zeros((dim_U, R))
            V = np.zeros((dim_V, R))
            W = np.zeros((dim_W, R))
            for c in range(min(R, n*m*p)):
                i = c // (m*p)
                j = (c // p) % m
                k = c % p
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
        
        # Add small random perturbations to break symmetry
        U += np.random.randn(dim_U, R) * 0.1
        V += np.random.randn(dim_V, R) * 0.1
        W += np.random.randn(dim_W, R) * 0.1
        
        # Precompute T for fast evaluation
        T = np.zeros((dim_U, dim_V, dim_W))
        for (I_idx, J_idx, K_idx) in T_indices:
            T[I_idx, J_idx, K_idx] = 1.0
        
        for it in range(max_iters):
            if time.time() > time_limit:
                return None
            
            # Update U
            for i in range(dim_U):
                A_U = np.einsum('jr, kr -> jkr', V, W).reshape(-1, R)
                b_U = T[i, :, :].flatten()
                try:
                    U[i, :], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                except:
                    return None
            
            # Update V
            for j in range(dim_V):
                A_V = np.einsum('ir, kr -> ikr', U, W).reshape(-1, R)
                b_V = T[:, j, :].flatten()
                try:
                    V[j, :], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                except:
                    return None
            
            # Update W
            for k in range(dim_W):
                A_W = np.einsum('ir, jr -> ijr', U, V).reshape(-1, R)
                b_W = T[:, :, k].flatten()
                try:
                    W[k, :], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                except:
                    return None
            
            # Periodically check for integer/half-integer solution
            if it % 3 == 0:
                U_round = np.round(U * 2) / 2.0
                V_round = np.round(V * 2) / 2.0
                W_round = np.round(W * 2) / 2.0
                
                if verify_solution(U_round, V_round, W_round, R):
                    return U_round, V_round, W_round
            
            # Also try pure integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify_solution(U_int, V_int, W_int, R):
                return U_int, V_int, W_int
        
        return None
    
    def try_direct_integer_search(R, time_limit, seed=42):
        """
        Direct search for integer solutions using structured perturbation.
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Start from schoolbook and try to merge columns
        U0, V0, W0 = schoolbook_decomposition()
        U0 = np.array(U0, dtype=float)
        V0 = np.array(V0, dtype=float)
        W0 = np.array(W0, dtype=float)
        
        # Try different random initializations
        for attempt in range(50):
            if time.time() > time_limit:
                return None
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, (dim_U, R)).astype(float)
            V = np.random.randint(-2, 3, (dim_V, R)).astype(float)
            W = np.random.randint(-2, 3, (dim_W, R)).astype(float)
            
            # Run a few ALS iterations
            T = np.zeros((dim_U, dim_V, dim_W))
            for (I_idx, J_idx, K_idx) in T_indices:
                T[I_idx, J_idx, K_idx] = 1.0
            
            for it in range(20):
                if time.time() > time_limit:
                    return None
                
                # Update U
                for i in range(dim_U):
                    A_U = np.einsum('jr, kr -> jkr', V, W).reshape(-1, R)
                    b_U = T[i, :, :].flatten()
                    try:
                        U[i, :], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                    except:
                        break
                
                # Update V
                for j in range(dim_V):
                    A_V = np.einsum('ir, kr -> ikr', U, W).reshape(-1, R)
                    b_V = T[:, j, :].flatten()
                    try:
                        V[j, :], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                    except:
                        break
                
                # Update W
                for k in range(dim_W):
                    A_W = np.einsum('ir, jr -> ijr', U, V).reshape(-1, R)
                    b_W = T[:, :, k].flatten()
                    try:
                        W[k, :], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                    except:
                        break
                
                # Check for integer solution
                U_round = np.round(U * 2) / 2.0
                V_round = np.round(V * 2) / 2.0
                W_round = np.round(W * 2) / 2.0
                if verify_solution(U_round, V_round, W_round, R):
                    return U_round, V_round, W_round
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify_solution(U_int, V_int, W_int, R):
                    return U_int, V_int, W_int
        
        return None
    
    # Main search
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try rank 47 first (best case for improvement)
    for R in [47, 46, 45]:
        if time.time() > deadline:
            break
        
        # Try ALS with structured initialization
        res = try_als_with_integer_constraint(R, max_iters=30, time_limit=deadline, seed=42)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try direct integer search
        res = try_direct_integer_search(R, time_limit=deadline, seed=42)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 48
    U, V, W = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
