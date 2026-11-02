"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0
                
    def verify_decomposition(U, V, W, rank):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err
        
    def try_peeling(rank, time_limit=8):
        """Try to find a rank-`rank` decomposition using peeling algorithm."""
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = rng.randn(dim_A, rank)
            V = rng.randn(dim_B, rank)
            W = rng.randn(dim_C, rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            prev_err = verify_decomposition(U, V, W, rank)
            
            # Alternating Least Squares with peeling
            for step in range(150):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                M = np.einsum('jr,kr->jkr', V, W).reshape(dim_B * dim_C, rank)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                N = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, rank)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(N, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                M = np.einsum('ir,jr->ijr', U, V).reshape(dim_A * dim_B, rank)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
                err = verify_decomposition(U, V, W, rank)
                if err < 1e-8:
                    return U, V, W
                if err > prev_err * 1.01:
                    break
                prev_err = err
                
        return None

    def try_rank13_compression(time_limit=30):
        """
        Try to find a rank-13 decomposition by exploiting linear dependencies
        in the known rank-14 Strassen decomposition.
        
        Strategy: Use the known rank-14 solution as a basis, then try to express
        one of the rank-1 components as a linear combination of the others,
        effectively reducing the rank by 1.
        """
        start_time = time.time()
        
        # First, get the known rank-14 decomposition
        rank14 = _get_rank14_decomposition()
        if rank14 is None:
            return None
        
        U14, V14, W14 = rank14['U'], rank14['V'], rank14['W']
        U14 = np.array(U14, dtype=float)
        V14 = np.array(V14, dtype=float)
        W14 = np.array(W14, dtype=float)
        
        # For each pair of rank components, try to see if one can be expressed
        # as a linear combination of the others
        # We'll try to eliminate component r by expressing it in terms of others
        
        for r_eliminate in range(14):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Remove component r_eliminate
            mask = [i for i in range(14) if i != r_eliminate]
            U_rem = U14[:, mask]
            V_rem = V14[:, mask]
            W_rem = W14[:, mask]
            
            # Try to find coefficients alpha, beta, gamma such that:
            # U14[:, r_eliminate] = sum_s alpha_s * U14[:, s]
            # V14[:, r_eliminate] = sum_s beta_s * V14[:, s]  
            # W14[:, r_eliminate] = sum_s gamma_s * W14[:, s]
            # But this won't work directly because the tensor structure requires
            # U, V, W to be consistent.
            
            # Instead, we'll use a different approach: 
            # Try to find a rank-13 decomposition by solving a constrained optimization
            
            # Build the residual tensor that we need to explain
            # If we have a rank-13 decomposition (U13, V13, W13), then:
            # T = sum_{r=1}^{13} U13[:,r] ⊗ V13[:,r] ⊗ W13[:,r]
            
            # We can try to initialize from the rank-14 solution by projecting
            # out one component
            
            # Alternative approach: Use the fact that the 2x2x4 tensor has
            # specific structure. Let's try a direct search with the constraint
            # that we're looking for rank 13.
            
            # Use ALS starting from a modified rank-14 initialization
            rng = np.random.RandomState(42 + r_eliminate)
            
            # Create rank-13 initialization by removing one component and 
            # redistributing its contribution
            U13 = U14[:, mask].copy()
            V13 = V14[:, mask].copy()
            W13 = W14[:, mask].copy()
            
            # Add small random perturbation to help escape local minima
            U13 += 0.01 * rng.randn(dim_A, 13)
            V13 += 0.01 * rng.randn(dim_B, 13)
            W13 += 0.01 * rng.randn(dim_C, 13)
            
            # Run ALS for rank 13
            prev_err = verify_decomposition(U13, V13, W13, 13)
            
            for step in range(200):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                # Update U13
                M = np.einsum('jr,kr->jkr', V13, W13).reshape(dim_B * dim_C, 13)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                try:
                    U13_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U13 = U13_new.T
                except:
                    pass
                    
                # Update V13
                N = np.einsum('ir,kr->ikr', U13, W13).reshape(dim_A * dim_C, 13)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V13_new, _, _, _ = np.linalg.lstsq(N, T_flat_V.T, rcond=None)
                    V13 = V13_new.T
                except:
                    pass
                    
                # Update W13
                M = np.einsum('ir,jr->ijr', U13, V13).reshape(dim_A * dim_B, 13)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W13_new, _, _, _ = np.linalg.lstsq(M, T_flat_W.T, rcond=None)
                    W13 = W13_new.T
                except:
                    pass
                    
                err = verify_decomposition(U13, V13, W13, 13)
                if err < 1e-8:
                    # Check if entries are close to integers/half-integers
                    is_valid = True
                    for mat in [U13, V13, W13]:
                        for i in range(mat.shape[0]):
                            for j in range(mat.shape[1]):
                                val = mat[i, j]
                                k = round(val * 2)
                                if abs(val - k * 0.5) > 1e-4:
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        U13_list = [[round(val * 2) * 0.5 for val in row] for row in U13.tolist()]
                        V13_list = [[round(val * 2) * 0.5 for val in row] for row in V13.tolist()]
                        W13_list = [[round(val * 2) * 0.5 for val in row] for row in W13.tolist()]
                        
                        U13_np = np.array(U13_list)
                        V13_np = np.array(V13_list)
                        W13_np = np.array(W13_list)
                        
                        if verify_decomposition(U13_np, V13_np, W13_np, 13) < 1e-6:
                            return {"U": U13_list, "V": V13_list, "W": W13_list}
                
                if err > prev_err * 1.01:
                    break
                prev_err = err
        
        return None

    def _get_rank14_decomposition():
        """Return the known rank-14 Strassen decomposition."""
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for B1 (cols 0,1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Strassen for B2 (cols 2,3)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return {"U": U, "V": V, "W": W}

    # Track overall deadline
    overall_start = time.time()
    overall_deadline = 140  # Safety margin before 150s limit
    
    # First, try to find a rank-13 decomposition using the compression heuristic
    if time.time() - overall_start < overall_deadline - 20:
        res13 = try_rank13_compression(time_limit=min(30, overall_deadline - (time.time() - overall_start) - 10))
        if res13 is not None:
            return res13
    
    # Try ranks from 11 to 14 with remaining time
    remaining_time = overall_deadline - (time.time() - overall_start)
    if remaining_time > 10:
        for rank in [11, 12, 13, 14]:
            if time.time() - overall_start > overall_deadline - 5:
                break
            time_for_rank = min(10, remaining_time - 5)
            res = try_peeling(rank, time_limit=time_for_rank)
            if res is not None:
                U, V, W = res
                U_list = [[U[i, r] for r in range(rank)] for i in range(dim_A)]
                V_list = [[V[i, r] for r in range(rank)] for i in range(dim_B)]
                W_list = [[W[i, r] for r in range(rank)] for i in range(dim_C)]
                
                # Check if entries are close to integers/half-integers
                is_valid = True
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for val in row:
                            k = round(val * 2)
                            if abs(val - k * 0.5) > 1e-4:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    U_list = [[round(val * 2) * 0.5 for val in row] for row in U_list]
                    V_list = [[round(val * 2) * 0.5 for val in row] for row in V_list]
                    W_list = [[round(val * 2) * 0.5 for val in row] for row in W_list]
                    U_np = np.array(U_list)
                    V_np = np.array(V_list)
                    W_np = np.array(W_list)
                    if verify_decomposition(U_np, V_np, W_np, rank) < 1e-6:
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to known rank-14 decomposition
    return _get_rank14_decomposition()
# EVOLVE-BLOCK-END
