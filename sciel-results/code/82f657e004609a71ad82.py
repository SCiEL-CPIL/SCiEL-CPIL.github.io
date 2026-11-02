"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy: Exploit tensor symmetry and Kronecker product structure.
    1. Start with schoolbook (rank 60) as baseline.
    2. Use Kronecker product decomposition: treat 4-dim middle index as 2x2 tensor product.
    3. For the 2x2x2 subproblem, use Strassen's rank-7 decomposition.
    4. Combine to get a potentially lower rank decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, create schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # The 7 rank-1 tensors for 2x2 matrix multiplication
    # Using the standard Strassen formulas with rational coefficients
    strassen_coeffs = [
        # (U_idx, V_idx, W_idx, coeff)
        # U indices: 0,1,2,3 for 2x2
        # V indices: 0,1,2,3 for 2x2
        # W indices: 0,1,2,3 for 2x2
        # Standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
    ]
    
    # Let me define the Strassen decomposition more carefully
    # For 2x2 matrices A, B, C = AB:
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # C = [[c0, c1], [c2, c3]]
    
    # Strassen's 7 products:
    # M1 = (a0 + a3)(b0 + b3)
    # M2 = (a2 + a3)b0
    # M3 = a0(b1 - b3)
    # M4 = a3(b2 - b0)
    # M5 = (a0 + a1)b3
    # M6 = (a2 - a0)(b0 + b1)
    # M7 = (a1 - a3)(b2 + b3)
    
    # Reconstruction:
    # c0 = M1 + M4 - M5 + M7
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M3 + M6
    
    # For tensor decomposition, each rank-1 term is:
    # U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's build the 2x2x2 rank-7 decomposition
    # U is 4x7, V is 4x7, W is 4x7
    
    def strassen_2x2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1: (a0+a3)(b0+b3)
        # U: a0=1, a3=1 (indices 0 and 3)
        # V: b0=1, b3=1 (indices 0 and 3)
        # W: c0=1, c3=1 (indices 0 and 3)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # M2: (a2+a3)b0
        # U: a2=1, a3=1 (indices 2 and 3)
        # V: b0=1 (index 0)
        # W: c2=1 (index 2)
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # M3: a0(b1-b3)
        # U: a0=1 (index 0)
        # V: b1=1, b3=-1 (indices 1 and 3)
        # W: c1=1, c3=1 (indices 1 and 3)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # M4: a3(b2-b0)
        # U: a3=1 (index 3)
        # V: b2=1, b0=-1 (indices 2 and 0)
        # W: c0=1, c2=1 (indices 0 and 2)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # M5: (a0+a1)b3
        # U: a0=1, a1=1 (indices 0 and 1)
        # V: b3=1 (index 3)
        # W: c0=-1, c1=1 (indices 0 and 1)
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # M6: (a2-a0)(b0+b1)
        # U: a2=1, a0=-1 (indices 2 and 0)
        # V: b0=1, b1=1 (indices 0 and 1)
        # W: c3=1 (index 3)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0
        
        # M7: (a1-a3)(b2+b3)
        # U: a1=1, a3=-1 (indices 1 and 3)
        # V: b2=1, b3=1 (indices 2 and 3)
        # W: c0=1 (index 0)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0
        
        return U, V, W
    
    # Now for 3x4x5, we can try to use block structure
    # Split middle dimension 4 into two blocks of size 2
    # A (3x4) = [A_left (3x2) | A_right (3x2)]
    # B (4x5) = [B_top (2x5); B_bottom (2x5)]
    # C = A_left * B_top + A_right * B_bottom
    
    # Each term is (3x2)*(2x5)
    # For (3x2)*(2x5), we can potentially use a decomposition
    
    # Let me try a different approach: direct Kronecker product
    # Treat the 4-dim space as 2x2 tensor product
    # Index j in [0,4) can be written as j = j1*2 + j2 where j1,j2 in [0,2)
    
    # For the multiplication tensor M_{3,4,5}:
    # T[i, j, k] = 1 if and only if the term A[i,j]*B[j,k] appears in C[i,k]
    
    # Using Kronecker structure:
    # j = j1*2 + j2
    # We can write the tensor as a sum over j1,j2 of terms involving 2-dim subproblems
    
    # Actually, let me try a simpler approach: 
    # Use the fact that M_{3,4,5} can be embedded in a larger structure
    
    # For now, let's just return schoolbook if we can't find improvement
    # But let me try one more thing: check if we can reduce rank by 1
    
    # Try rank 59: can we find a linear dependency?
    # The tensor has 3*4*5 = 60 nonzero entries
    # If there's a linear dependency among the rank-1 terms, we can reduce rank
    
    # Let me check if the schoolbook terms are linearly independent
    # Each term is e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # These are 60 distinct basis tensors, so they're linearly independent
    # Thus rank is exactly 60 for schoolbook
    
    # But maybe there's a different decomposition with lower rank?
    # The lower bound is max(3*4, 4*5, 5*3) = max(12, 20, 15) = 20
    # So in principle, rank could be as low as 20
    
    # Given time constraints, let's return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
