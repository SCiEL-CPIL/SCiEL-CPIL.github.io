"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Strategy: Apply Strassen (rank 7) to the first 3x3 block of B, then extend
    to the remaining 3x4 sub-block using shared row structure of A.
    Target rank: 24 (Strassen 7 + 4 blocks * 4 extra = 7 + 16 = 23, aim for ~24)
    """
    start_time = time.time()
    deadline = 135.0  # Leave safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # ============================================================
    # STRASSEN DECOMPOSITION FOR 3x3 * 3x3
    # ============================================================
    # Strassen's algorithm for 3x3 matrix multiplication uses rank 7
    # (instead of 9 schoolbook). The 7 rank-1 terms are:
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B12
    # M3 = A11 (B21 - B22)
    # M4 = A22 (B12 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 3x3, we need to extend this. Let's use the standard 3x3 Strassen
    # which actually requires more terms. Let me use a different approach.
    #
    # Actually, for 3x3*3x3, the known minimal rank is 19 (not 7).
    # Strassen gives rank 7 for 2x2*2x2. For 3x3, we can use a block approach.
    #
    # Let me use a simpler approach: schoolbook for the full problem but
    # try to find a low-rank decomposition via optimization.
    # ============================================================
    
    # Schoolbook decomposition as baseline (rank 63)
    rank_schoolbook = n * m * p  # 63
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i, c] * V_mat[j, c] * W_mat[k, c] for c in range(U_mat.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def convert_to_interface(U_mat, V_mat, W_mat):
        """Convert from (n x r), (m x r), (p x r) to interface format."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        U_list = [[0] * r for _ in range(n * m)]
        V_list = [[0] * r for _ in range(m * p)]
        W_list = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U_list[i * m + j][c] = U_mat[i, c]
        
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V_list[j * p + k][c] = V_mat[j, c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W_list[k * n + i][c] = W_mat[k, c]
        
        return U_list, V_list, W_list

    def round_to_half_integer(mat):
        """Round matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def als_solve(U, V, W, r, iterations=50):
        """
        Alternating Least Squares to minimize sum (1 - sum_c U[i,c]V[j,c]W[k,c])^2
        """
        n, m, p = DIMENSIONS
        for it in range(iterations):
            # Solve for U
            A_U = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    A_U[j * p + k, :] = V[j, :] * W[k, :]
            
            for i in range(n):
                U[i, :] = np.linalg.lstsq(A_U, np.ones(m * p), rcond=None)[0]
            
            # Solve for V
            A_V = np.zeros((n * p, r))
            for i in range(n):
                for k in range(p):
                    A_V[i * p + k, :] = U[i, :] * W[k, :]
            
            for j in range(m):
                V[j, :] = np.linalg.lstsq(A_V, np.ones(n * p), rcond=None)[0]
                
            # Solve for W
            A_W = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    A_W[i * m + j, :] = U[i, :] * V[j, :]
            
            for k in range(p):
                W[k, :] = np.linalg.lstsq(A_W, np.ones(n * m), rcond=None)[0]
                
        return U, V, W

    def compute_error(U_sol, V_sol, W_sol):
        """Compute the sum of squared errors."""
        n, m, p = DIMENSIONS
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_sol[i], V_sol[j] * W_sol[k])
                    err += (val - 1.0) ** 2
        return err

    # ============================================================
    # CONSTRUCTIVE APPROACH: Block decomposition
    # ============================================================
    # Split B into B[:, 0:3] (3x3) and B[:, 3:7] (3x4)
    # 
    # For the 3x3 * 3x3 part, use a rank-19 decomposition (known optimal)
    # For the 3x3 * 3x4 part, use schoolbook (rank 12) but try to share
    # some factors with the first part.
    #
    # Actually, let's try a direct optimization approach with structured
    # initializations based on the block structure.
    
    np.random.seed(42)
    
    # Try various ranks with structured and random initializations
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 45, 50, 55, 60]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random initializations
        n_init = 5
        for init in range(n_init):
            if time.time() > deadline - 8:
                break
            
            if init == 0:
                # Structured initialization: schoolbook-like but with random mixing
                U_init = np.zeros((n, r))
                V_init = np.zeros((m, r))
                W_init = np.zeros((p, r))
                
                # First min(r, 9) terms: schoolbook for first 3x3 block
                for c in range(min(r, 9)):
                    i = c // 3
                    j = c % 3
                    U_init[i, c] = 1.0
                    V_init[j, c] = 1.0
                    W_init[c % 3, c] = 1.0  # k = c % 3 for first 3 columns
                
                # Remaining terms: random
                if r > 9:
                    U_init[:, 9:] = np.random.randn(n, r - 9) * 0.3
                    V_init[:, 9:] = np.random.randn(m, r - 9) * 0.3
                    W_init[:, 9:] = np.random.randn(p, r - 9) * 0.3
            else:
                # Pure random initialization
                U_init = np.random.randn(n, r) * 0.5 + 0.5
                V_init = np.random.randn(m, r) * 0.5 + 0.5
                W_init = np.random.randn(p, r) * 0.5 + 0.5
            
            U_sol, V_sol, W_sol = als_solve(U_init, V_init, W_init, r, iterations=30)
            
            # Check error
            err = compute_error(U_sol, V_sol, W_sol)
            
            if err < 1e-6:
                # Try rounding to half-integers
                U_rounded = round_to_half_integer(U_sol)
                V_rounded = round_to_half_integer(V_sol)
                W_rounded = round_to_half_integer(W_sol)
                
                # Verify rounded solution
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    U_list, V_list, W_list = convert_to_interface(U_rounded, V_rounded, W_rounded)
                    best_rank = r
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    found = True
                    break
            elif err < 1e-3:
                # Close but not exact - try more iterations
                U_sol2, V_sol2, W_sol2 = als_solve(U_sol, V_sol, W_sol, r, iterations=50)
                err2 = compute_error(U_sol2, V_sol2, W_sol2)
                if err2 < 1e-6:
                    U_rounded = round_to_half_integer(U_sol2)
                    V_rounded = round_to_half_integer(V_sol2)
                    W_rounded = round_to_half_integer(W_sol2)
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        U_list, V_list, W_list = convert_to_interface(U_rounded, V_rounded, W_rounded)
                        best_rank = r
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        found = True
                        break
        
        if found:
            break
    
    # If no better solution found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
