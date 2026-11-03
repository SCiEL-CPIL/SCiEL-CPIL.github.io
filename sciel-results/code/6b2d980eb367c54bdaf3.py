"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 135.0  # 15 seconds margin before 150s limit
    
    # Schoolbook decomposition (rank 90)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try Strassen block decomposition
    # The 3x5x6 tensor can be decomposed into blocks
    # We'll try to use rank-7 Strassen for 2x2x2 blocks where possible
    
    # For 3x5x6, we can partition:
    # A is 3x5, B is 5x6, C is 3x6
    # We can try to use Strassen on overlapping 2x2x2 blocks
    
    # Let's try a direct search approach for lower rank
    # Known lower bounds: For 3x5x6, the minimal rank is at most 84 (via Strassen blocks)
    # Let's try to construct a rank-84 decomposition using Strassen
    
    # Build Strassen 2x2x2 decomposition
    # Standard Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's try a hybrid approach: use known good decompositions
    # For now, return the schoolbook if search doesn't find better in time
    
    # Try random search for lower rank
    def verify_decomposition(U, V, W, rank, tol=1e-10):
        """Verify that the decomposition is exact."""
        # Tensor T[i,j,k] = 1 for all i,j,k
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a rank-84 decomposition using block Strassen
    # Partition A (3x5) into blocks, B (5x6) into blocks
    
    # Let's try: A = [A_top (2x5); A_bottom (1x5)]
    # B = [B_left (5x4); B_right (5x2)]
    # C = [C_top_left (2x4); C_top_right (2x2); C_bottom_left (1x4); C_bottom_right (1x2)]
    
    # For the 2x2x2 corner blocks, use Strassen
    # This is complex, so let's try a different approach
    
    # Try randomized search for rank 84
    if time.time() < deadline - 30:
        # Use a structured approach: known decompositions
        # The minimal rank for 3x5x6 is believed to be around 80-84
        
        # Let's construct a rank-84 decomposition manually using block structure
        # A (3x5), B (5x6), C (3x6)
        
        # Partition indices:
        # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
        
        # We'll use the following block decomposition:
        # Split j into {0,1} and {2,3,4}
        # Split k into {0,1} and {2,3,4,5}
        # Split i into {0,1} and {2}
        
        # This gives us blocks that can use Strassen for the 2x2 parts
        
        # For simplicity, let's use a known construction:
        # Total rank = 2*2*2 (Strassen: 7) * 3 blocks + remaining schoolbook terms
        
        # Actually, let's just try to verify if rank 84 is achievable
        # by constructing it explicitly
        
        # Construction:
        # Use 6 Strassen 2x2x2 blocks (each rank 7) = 42
        # Remaining terms: 90 - 6*8 = 42 (since each 2x2x2 block has 8 schoolbook terms, replaced by 7)
        # Total = 42 + 42 = 84
        
        # Let's build this:
        # The 2x2x2 blocks cover:
        # i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
        # i in {0,1}, j in {0,1}, k in {2,3} -> Strassen  
        # i in {0,1}, j in {0,1}, k in {4,5} -> Strassen
        # i in {0,1}, j in {2,3}, k in {0,1} -> Strassen
        # i in {0,1}, j in {2,3}, k in {2,3} -> Strassen
        # i in {0,1}, j in {2,3}, k in {4,5} -> Strassen
        
        # That's 6 blocks of 2x2x2 = 6*8 = 48 terms replaced by 6*7 = 42 terms
        # Remaining: i=2 with all j,k = 1*5*6 = 30 terms
        # Also: i in {0,1}, j=4, all k = 2*1*6 = 12 terms
        # Total = 42 + 30 + 12 = 84. 
        
        # Let's build this decomposition
        
        U = [[0]*84 for _ in range(15)]  # 3*5 = 15 rows
        V = [[0]*84 for _ in range(30)]  # 5*6 = 30 rows
        W = [[0]*84 for _ in range(18)]  # 6*3 = 18 rows
        
        col = 0
        
        # Strassen 2x2x2 block helper
        def add_strassen_block(i0, j0, k0):
            """Add a Strassen 2x2x2 block starting at (i0, j0, k0). Returns new column index."""
            nonlocal col
            # 7 terms
            # Term 0: (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1]) -> W[k0,n+i0] and W[k0+1,n+i0+1]
            # This is getting complex. Let me use a simpler verification-based approach.
            
            # Actually, let me just hardcode the known good rank-84 result
            # or fall back to schoolbook
            return col
        
        # Given complexity, let's try a numerical search approach
        # Use gradient-free optimization
        
        # Initialize with random values
        import random
        random.seed(42)
        np.random.seed(42)
        
        def objective(params):
            """Objective function for tensor decomposition."""
            rank = len(params) // (15 + 30 + 18)
            U_flat = params[:15*rank]
            V_flat = params[15*rank:15*rank+30*rank]
            W_flat = params[15*rank+30*rank:]
            
            U_mat = np.array(U_flat).reshape(15, rank)
            V_mat = np.array(V_flat).reshape(30, rank)
            W_mat = np.array(W_flat).reshape(18, rank)
            
            # Compute tensor values
            error = 0.0
            for i in range(3):
                for j in range(5):
                    for k in range(6):
                        val = np.sum(U_mat[i*5+j] * V_mat[j*6+k] * W_mat[k*3+i])
                        error += (val - 1.0) ** 2
            return error
        
        # Try rank 84 with random initialization
        if time.time() < deadline - 20:
            rank_try = 84
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                    
                # Random initialization with small values
                params = np.random.randn((15+30+18)*rank_try) * 0.1
                result = minimize(objective, params, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15})
                
                if result.fun < 1e-10:
                    U_flat = result.x[:15*rank_try]
                    V_flat = result.x[15*rank_try:15*rank_try+30*rank_try]
                    W_flat = result.x[15*rank_try+30*rank_try:]
                    
                    U_mat = np.array(U_flat).reshape(15, rank_try)
                    V_mat = np.array(V_flat).reshape(30, rank_try)
                    W_mat = np.array(W_flat).reshape(18, rank_try)
                    
                    # Verify
                    if verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), rank_try):
                        best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                        best_rank = rank_try
                        break
        
        # If we found a better rank, return it
        if best_rank < rank_school:
            return best_result
    
    return best_result
# EVOLVE-BLOCK-END
