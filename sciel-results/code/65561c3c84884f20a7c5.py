import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for safety
    
    # Construct rank-71 solution using Strassen + schoolbook
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Try to find rank-70 solution using structured search
    if time.time() - start_time < time_limit - 20:
        res_70 = try_find_rank_70_structured(n, m, p, start_time, time_limit)
        if res_70 is not None:
            return res_70
    
    return {"U": U, "V": V, "W": W}

def try_find_rank_70_structured(n, m, p, start_time, time_limit):
    """
    Try to find a rank-70 solution by modifying the rank-71 solution.
    Strategy: The rank-71 solution has 4 Strassen blocks (28 rank) + 43 schoolbook terms.
    We need to eliminate one rank unit. 
    
    Approach: Try to merge two schoolbook terms into a single rank-1 term,
    or find a linear dependency in the current decomposition.
    """
    rank = 70
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Start from the rank-71 solution and try to find a rank reduction
    # by checking if any column is a linear combination of others
    
    # Build the rank-71 solution as numpy arrays
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = len(blocks) * 7
    total_terms = n * m * p
    strassen_terms_count = len(blocks) * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank_71 = total_strassen_rank + schoolbook_terms_count
    
    U71 = np.zeros((n_m, total_rank_71))
    V71 = np.zeros((m_p, total_rank_71))
    W71 = np.zeros((p_n, total_rank_71))
    
    col = 0
    
    def add_strassen_block_np(i0, j0, k0, col_offset):
        c = col_offset + 0
        U71[i0*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 1
        U71[(i0+1)*m + j0, c] += 1
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[j0*p + k0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += -1
        
        c = col_offset + 2
        U71[i0*m + j0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += -1
        W71[(k0+1)*n + i0, c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 3
        U71[(i0+1)*m + (j0+1), c] += 1
        V71[(j0+1)*p + k0, c] += 1
        V71[j0*p + k0, c] += -1
        W71[k0*n + i0, c] += 1
        W71[k0*n + (i0+1), c] += 1
        
        c = col_offset + 4
        U71[i0*m + j0, c] += 1
        U71[i0*m + (j0+1), c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += -1
        W71[(k0+1)*n + i0, c] += 1
        
        c = col_offset + 5
        U71[(i0+1)*m + j0, c] += 1
        U71[i0*m + j0, c] += -1
        V71[j0*p + k0, c] += 1
        V71[j0*p + (k0+1), c] += 1
        W71[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 6
        U71[i0*m + (j0+1), c] += 1
        U71[(i0+1)*m + (j0+1), c] += -1
        V71[(j0+1)*p + k0, c] += 1
        V71[(j0+1)*p + (k0+1), c] += 1
        W71[k0*n + i0, c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block_np(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U71[i*m+j, col] += 1
                    V71[j*p+k, col] += 1
                    W71[k*n+i, col] += 1
                    col += 1
    
    # Now try to find a rank-70 solution by random perturbation and rounding
    # Use alternating minimization with structured initialization
    
    for trial in range(20):
        if time.time() - start_time > time_limit - 10:
            break
        
        np.random.seed(42 + trial)
        
        # Initialize with small perturbation around the rank-71 solution's structure
        # Try different strategies for each trial
        
        if trial < 5:
            # Strategy 1: Random initialization
            U = np.random.randn(n_m, rank) * 0.3
            V = np.random.randn(m_p, rank) * 0.3
            W = np.random.randn(p_n, rank) * 0.3
        elif trial < 10:
            # Strategy 2: Start from rank-71, drop one column, perturb
            # Drop a random schoolbook column
            schoolbook_start = total_strassen_rank
            drop_col = schoolbook_start + np.random.randint(schoolbook_terms_count)
            keep_cols = [c for c in range(total_rank_71) if c != drop_col]
            U = U71[:, keep_cols].copy()
            V = V71[:, keep_cols].copy()
            W = W71[:, keep_cols].copy()
            # Add small random perturbation
            U += np.random.randn(n_m, rank) * 0.01
            V += np.random.randn(m_p, rank) * 0.01
            W += np.random.randn(p_n, rank) * 0.01
        else:
            # Strategy 3: Use the rank-71 solution, try to merge two nearby schoolbook terms
            U = U71[:, :total_rank_71-1].copy()  # Drop last column
            V = V71[:, :total_rank_71-1].copy()
            W = W71[:, :total_rank_71-1].copy()
            # Perturb
            U += np.random.randn(n_m, rank) * 0.05
            V += np.random.randn(m_p, rank) * 0.05
            W += np.random.randn(p_n, rank) * 0.05
        
        # Alternating minimization
        for iteration in range(30):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(i,i)*delta(k,k)
                    # Wait, the tensor is T_ijk where C_ik = sum_j A_ij * B_jk * T_ijk
                    # So T_ijk should be 1 if this is a valid multiplication term
                    # Actually for matrix multiplication, T_ijk = 1 for all valid (i,j,k)
                    # No wait, the tensor decomposition is for the multiplication map
                    # T: A ⊗ B -> C
                    # T_{i,j,k} is the coefficient of A_{i,j} * B_{j,k} in C_{i,k}
                    # For standard matrix multiplication, T_{i,j,k} = 1 for all i,j,k
                    # where the term A_{i,j} * B_{j,k} contributes to C_{i,k}
                    
                    # The constraint is:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                    
                    # For fixed (i,j), this gives p*n equations:
                    # For each (k, i'): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] = delta(i, i')
                    # Wait no, the output index is (i, k) where i is row of C, k is col of C
                    # C_{i,k} = sum_j A_{i,j} B_{j,k}
                    # So the tensor T has T_{i,j,k} = 1 if the term exists
                    # The decomposition: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
                    # Wait, the W index is k*n+i which is (k, i) in row-major
                    # So W[k*n+i, c] corresponds to output position (i, k)
                    # The constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                    
                    # For fixed (i,j), we have p*n equations (one for each (k, i')):
                    # Actually no, for fixed (i,j), the output is C_{i,k} for k=0..p-1
                    # And the equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each k
                    # But W[k*n+i, c] depends on both k and i
                    # So for fixed (i,j), we have p equations (one for each k):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    
                    # This is a system of p equations in rank unknowns
                    # But it's nonlinear in U because... wait no, it's linear in U for fixed V,W
                    # A_{k,c} = V[j*p+k, c] * W[k*n+i, c]
                    # b_k = 1
                    # A * U[i*m+j, :] = b
                    
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k), we have n*m equations (one for each (i, i')):
                    # Actually for fixed (j,k), the equation is:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each i
                    # So n equations:
                    # A_{i,c} = U[i*m+j, c] * W[k*n+i, c]
                    # b_i = 1
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i), we have m equations (one for each j):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each j
                    # A_{j,c} = U[i*m+j, c] * V[j*p+k, c]
                    # b_j = 1
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    try:
                        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-8:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                error_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            error_r += (val - 1.0) ** 2
                
                if error_r < 1e-6:
                    # Check that all entries are half-integers
                    if (np.allclose(U_r * 2, np.round(U_r * 2)) and
                        np.allclose(V_r * 2, np.round(V_r * 2)) and
                        np.allclose(W_r * 2, np.round(W_r * 2))):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
    
    return None
