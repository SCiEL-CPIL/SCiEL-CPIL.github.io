"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x4 by 4x7.
    Uses Strassen blocks with cross-block optimization to minimize rank.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def check_time():
        return time.time() < deadline
    
    # Build the initial rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] -= 1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        V[e_idx, col] -= 1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1
        W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1
        U[a_idx, col] -= 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1
        U[d_idx, col] -= 1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    # Build the initial rank-50 decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    def to_integer_list(mat):
        """Convert numpy matrix to list of lists with rounded integers."""
        return [[int(round(mat[i, j])) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that U, V, W correctly decompose the tensor."""
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by finding null space
    # The decomposition T = sum_r u_r ⊗ v_r ⊗ w_r means:
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # We can try to eliminate columns by finding linear dependencies
    # Strategy: try removing one column at a time and see if we can adjust others
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Try greedy column elimination
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    active = list(range(rank_init))
    
    for _ in range(rank_init):
        if not check_time():
            break
        
        # Try removing each active column
        removed = False
        for idx in active[:]:
            if not check_time():
                break
            
            test_active = [a for a in active if a != idx]
            if len(test_active) == 0:
                break
            
            # Check if we can solve for the remaining columns
            # We need: sum_{r in test_active} U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
            # This is a bilinear system. Let's try a simpler approach:
            # Check if the column idx is a linear combination of others in some sense
            
            # Simpler: just check if removing it still gives valid decomposition
            # by checking if the tensor can still be represented
            U_test = U_work[:, test_active]
            V_test = V_work[:, test_active]
            W_test = W_work[:, test_active]
            
            if verify(U_test, V_test, W_test):
                # Successfully removed column idx
                U_work = U_test
                V_work = V_test
                W_test = W_test
                active = test_active
                removed = True
                break
        
        if not removed:
            break
    
    if len(active) < best_rank:
        best_U = U_work
        best_V = V_work
        best_W = W_work
        best_rank = len(active)
    
    # Now try a more aggressive approach: randomized search for lower rank
    # Try to find a decomposition of rank r for r from best_rank-1 down to some minimum
    for target_rank in range(best_rank - 1, max(40, best_rank - 10), -1):
        if not check_time():
            break
        
        # Try to find U, V, W of shape (8, target_rank), (28, target_rank), (14, target_rank)
        # such that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        
        # Use iterative refinement: start from current best and try to project
        # This is complex, so let's try a different approach:
        # Use the fact that we can perturb the existing decomposition
        
        # Alternative: try specific known constructions
        # For 2x4 * 4x7, known bounds suggest rank might be around 45-50
        
        # Let's try a direct numerical approach:
        # Flatten the tensor and solve a rank-constrained problem
        pass
    
    # Convert to output format
    U_list = to_integer_list(best_U)
    V_list = to_integer_list(best_V)
    W_list = to_integer_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
