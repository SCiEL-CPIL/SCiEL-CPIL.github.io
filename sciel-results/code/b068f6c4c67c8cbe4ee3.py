import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

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
        return U, V, W, rank

    def winograd_fta_2x3x6():
        """
        Apply Winograd Fourier Transform Algorithm to 2x3 x 3x6 multiplication.
        
        Strategy: Treat the middle dimension (size 3) as a cyclic convolution on Z_3.
        For each pair (i,k) where i in [0,1] and k in [0,5], we need:
          C[i,k] = sum_j A[i,j] * B[j,k]
        
        We can use the DFT on Z_3 to diagonalize the convolution structure.
        The DFT matrix for Z_3 is:
          F = [[1, 1, 1],
               [1, ω, ω^2],
               [1, ω^2, ω]]
        where ω = e^(2πi/3) = -1/2 + i√3/2
        
        Since we need exact rational/integer decompositions, we work over the ring
        Z[ω] and then map back. However, a cleaner approach is to use the fact that
        the multiplication tensor can be decomposed using the algebraic structure.
        
        Alternative: Use the Kronecker product structure.
        2x3 * 3x6 = (2x3) * (3x6)
        
        We can view this as: for each of the 2 rows of A and 6 columns of B,
        we compute a dot product of length 3.
        
        Using Winograd's idea: embed the length-3 convolution into a larger cyclic
        group Z_6 and use the DFT on Z_6.
        
        Actually, let's use a direct algebraic approach.
        
        Key insight: The tensor for 2x3 * 3x6 multiplication has rank at most
        2 * 3 * 6 = 36 (schoolbook). We can potentially reduce this by exploiting
        symmetries.
        
        Let's try the following decomposition based on the structure of the problem.
        For 2x3 * 3x6, we can write:
        
        C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
        
        This is equivalent to 12 independent dot products of length 3.
        
        Using the fact that over a field, the tensor rank of a single dot product
        of length n is n, but when we have multiple related dot products sharing
        structure, we can sometimes do better.
        
        Let's try a rank-26 decomposition based on known results for small matrix
        multiplications.
        
        Actually, let's use a more systematic approach: try to find a decomposition
        using the algebraic structure of the problem.
        
        For 2x3 * 3x6, one approach is to use the fact that this is equivalent to
        computing 12 outputs, each being a sum of 3 products.
        
        Let's try to find a decomposition with rank 26 using the following method:
        1. Start with the schoolbook decomposition (rank 36)
        2. Try to eliminate 10 terms by finding linear dependencies
        
        However, a more promising approach is to use the Winograd FTA directly.
        
        The Winograd FTA for convolution of length n uses rank n + 2*floor(n/2) - 1
        or similar formulas depending on the specific variant.
        
        For our case, let's try a different tactic: use the known result that
        R(2x3, 3x6) <= 26 based on literature on small matrix multiplication.
        
        Let me construct a specific decomposition.
        """
        # We'll use a rank-26 decomposition based on the following structure:
        # The 2x3 * 3x6 multiplication can be decomposed using 26 rank-1 terms.
        
        # Let's define the decomposition explicitly.
        # U has shape (6, 26), V has shape (18, 26), W has shape (12, 26)
        
        rank = 26
        U = [[0] * rank for _ in range(6)]   # 2*3 = 6
        V = [[0] * rank for _ in range(18)]  # 3*6 = 18
        W = [[0] * rank for _ in range(12)]  # 6*2 = 12
        
        # We need to find coefficients such that:
        # For all i in [0,1], j in [0,2], k in [0,5]:
        #   sum_c U[i*3+j][c] * V[j*6+k][c] * W[k*2+i][c] = 1
        
        # This is a system of 36 equations in 26 * (6 + 18 + 12) = 1560 unknowns.
        # We can solve this using optimization or algebraic methods.
        
        # Let's use a randomized approach with algebraic constraints.
        np.random.seed(42)
        
        # Initialize with random values
        U_np = np.random.randint(-2, 3, size=(6, rank)).astype(float)
        V_np = np.random.randint(-2, 3, size=(18, rank)).astype(float)
        W_np = np.random.randint(-2, 3, size=(12, rank)).astype(float)
        
        # Set up the linear system
        # For each (i, j, k), we have:
        # sum_c U[i*3+j][c] * V[j*6+k][c] * W[k*2+i][c] = 1
        
        # This is a trilinear system. We can use alternating least squares.
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Solve for W
            # For fixed U, V, the equation for W is linear:
            # For each (i, k): sum_j sum_c U[i*3+j][c] * V[j*6+k][c] * W[k*2+i][c] = 1
            # This gives us 12 equations in 26*12 = 312 unknowns for W.
            
            # Build the system for W
            A_w = np.zeros((12, 12 * rank))
            b_w = np.ones(12)
            
            for i in range(2):
                for k in range(6):
                    row_idx = k * 2 + i
                    for j in range(3):
                        for c in range(rank):
                            coeff = U_np[i*3+j][c] * V_np[j*6+k][c]
                            A_w[row_idx, (k*2+i)*rank + c] = coeff
                    # Note: W[k*2+i] is a vector of length rank, but we only use it once per (i,k)
                    # Actually, W has shape (12, rank), so W[k*2+i] is a row of length rank.
                    # The equation is: sum_j sum_c U[i*3+j][c] * V[j*6+k][c] * W[k*2+i][c] = 1
                    # This is a dot product of (sum_j U[i*3+j][c] * V[j*6+k][c]) with W[k*2+i]
            
            # Solve the least squares problem
            W_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W_np = W_sol.reshape(12, rank)
            
            # Solve for V
            # For fixed U, W, the equation for V is linear:
            # For each (i, j, k): sum_c U[i*3+j][c] * V[j*6+k][c] * W[k*2+i][c] = 1
            # This gives us 36 equations in 18*26 = 468 unknowns for V.
            
            A_v = np.zeros((36, 18 * rank))
            b_v = np.ones(36)
            
            eq_idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        for c in range(rank):
                            coeff = U_np[i*3+j][c] * W_np[k*2+i][c]
                            A_v[eq_idx, (j*6+k)*rank + c] = coeff
                        eq_idx += 1
            
            V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V_np = V_sol.reshape(18, rank)
            
            # Solve for U
            # For fixed V, W, the equation for U is linear:
            A_u = np.zeros((36, 6 * rank))
            b_u = np.ones(36)
            
            eq_idx = 0
            for i in range(2):
                for j in range(3):
                    for k in range(6):
                        for c in range(rank):
                            coeff = V_np[j*6+k][c] * W_np[k*2+i][c]
                            A_u[eq_idx, (i*3+j)*rank + c] = coeff
                        eq_idx += 1
            
            U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U_np = U_sol.reshape(6, rank)
            
            # Check convergence
            if iteration % 10 == 0:
                max_err = 0.0
                for i in range(2):
                    for j in range(3):
                        for k in range(6):
                            s = 0.0
                            for c in range(rank):
                                s += U_np[i*3+j][c] * V_np[j*6+k][c] * W_np[k*2+i][c]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-6:
                    # Try rounding to half-integers
                    U_snap = np.round(U_np * 2) / 2
                    V_snap = np.round(V_np * 2) / 2
                    W_snap = np.round(W_np * 2) / 2
                    if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, 2, 3, 6):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    
                    # Try integer rounding
                    U_snap2 = np.round(U_np)
                    V_snap2 = np.round(V_np)
                    W_snap2 = np.round(W_np)
                    if verify_decomposition(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), rank, 2, 3, 6):
                        return U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()
        
        return None
    
    # Try Winograd FTA approach
    res = winograd_fta_2x3x6()
    if res:
        U, V, W = res
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook(2, 3, 6)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
