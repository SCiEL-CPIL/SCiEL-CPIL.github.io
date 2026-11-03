"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        rank = schoolbook_rank
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
    
    def build_strassen_hybrid():
        """
        Embed Strassen's rank-7 decomposition for the 3x3x3 core (indices i,j,k in {0,1,2}),
        then handle boundary cases with schoolbook corrections.
        
        Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        Tensor: T[i,j,k] = 1 for all valid i,j,k
        
        Strassen's 7 products for 3x3 matrices:
        M1 = (A11+A22)(B11+B22)
        M2 = (A22)(B11-B22)  -- wait, let me use standard Strassen for 3x3
        
        Actually, for a 3x3 matrix multiplication, Strassen's algorithm gives:
        C11 = M1 + M2 + M4 - M6
        C12 = M3 + M5
        C13 = M3 + M5 + M7  -- hmm, this isn't quite right for 3x3
        
        Let me use the standard Strassen decomposition for 3x3:
        Partition A, B, C into 2x2 blocks of 1.5x1.5? No, that's not integer.
        
        Better approach: Use the known rank-7 decomposition for 3x3x3 tensor.
        The 3x3x3 matrix multiplication tensor has rank 7 (Strassen).
        
        The 7 terms are:
        m1 = (a11+a22)(b11+b22)
        m2 = (a22)(b11-b22)  -- actually let me look this up properly
        
        Standard Strassen for 3x3:
        We partition into 2x2 blocks where each "block" is a single element (since 3 is odd,
        we can't do clean 2x2 partitioning). 
        
        Alternative: Use the fact that the 3x3x3 tensor has a known rank-7 decomposition.
        The decomposition uses 7 rank-1 tensors. Let me construct them explicitly.
        
        For 3x3x3 matrix mult, the tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}.
        A rank-1 term is u_i * v_j * w_k.
        
        Strassen's 7 products for multiplying two 3x3 matrices:
        Let A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
        Let B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
        
        The 7 products (one standard formulation):
        P1 = (a11+a22+a33)(b11+b22+b33)
        P2 = (a11+a22)(b11+b22)
        P3 = (a22+a33)(b22+b33)
        P4 = (a33)(b33)
        P5 = (a11+a12+a13)(b11+b21+b31)
        P6 = (a21+a22+a23)(b12+b22+b32)
        P7 = (a13+a23+a33)(b13+b23+b33)
        
        Hmm, this is getting complicated. Let me use a cleaner approach.
        
        Actually, the simplest correct approach: 
        - Use Strassen's rank-7 for the 3x3x3 sub-problem (i,j,k in {0,1,2})
        - Use schoolbook for all cases where i=3 or j=3 or k=3
        
        The 3x3x3 sub-tensor: T[i,j,k]=1 for i,j,k in {0,1,2}
        Strassen's decomposition gives 7 rank-1 terms that sum to 1 for all i,j,k in {0,1,2}.
        
        For the boundary cases, we need schoolbook terms.
        Cases to handle with schoolbook:
        - i=3, j in {0,1,2,3}, k in {0,1,2,3,4}: 4*5 = 20 terms
        - i in {0,1,2}, j=3, k in {0,1,2,3,4}: 3*5 = 15 terms  
        - i in {0,1,2}, j in {0,1,2}, k=4: 3*3 = 9 terms
        Total boundary schoolbook: 20 + 15 + 9 = 44 terms
        
        But wait, some of these overlap with the Strassen core. The Strassen core handles
        i in {0,1,2}, j in {0,1,2}, k in {0,1,2}. So boundary cases are:
        - i=3, any j, any k: 4*5=20
        - i in {0,1,2}, j=3, any k: 3*5=15
        - i in {0,1,2}, j in {0,1,2}, k=4: 3*3=9
        Total: 44 schoolbook terms + 7 Strassen terms = 51 total rank
        
        This is better than 80!
        
        Now I need to explicitly write out Strassen's 7 rank-1 decompositions for 3x3x3.
        
        The 3x3x3 matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k in {0,1,2}
        We need 7 vectors u_c, v_c, w_c (each of length 3) such that:
        sum_{c=1}^{7} u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k in {0,1,2}
        
        Standard Strassen decomposition for 3x3:
        Using the 2x2 Strassen extended to 3x3 by embedding:
        
        Actually, let me just use the explicit known decomposition. For the 3x3x3 tensor,
        one rank-7 decomposition is:
        
        Term 1: u=[1,0,0], v=[1,0,0], w=[1,0,0] -> contributes 1 only at (0,0,0)
        That's not Strassen.
        
        Let me think differently. The matrix multiplication tensor for 3x3 has the property:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        Strassen's algorithm for 3x3 uses 7 multiplications. The 7 products are:
        M1 = (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
        M2 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        M3 = (A[1,1]+A[2,2]) * (B[1,1]+B[2,2])
        M4 = A[2,2] * B[2,2]
        M5 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0])
        M6 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[2,1])
        M7 = (A[0,2]+A[1,2]+A[2,2]) * (B[0,2]+B[1,2]+B[2,2])
        
        And the reconstruction:
        C[0,0] = M1 - M2 + M4  -- not quite right
        
        This is getting messy. Let me use a verified explicit decomposition.
        
        For the 3x3x3 tensor, a known rank-7 decomposition uses these 7 rank-1 terms:
        (u_c[i], v_c[j], w_c[k]) where:
        
        c=1: u=[1,1,1], v=[1,1,1], w=[1,1,1]
        c=2: u=[1,1,0], v=[1,1,0], w=[1,1,0]
        c=3: u=[0,1,1], v=[0,1,1], w=[0,1,1]
        c=4: u=[0,0,1], v=[0,0,1], w=[0,0,1]
        c=5: u=[1,0,0], v=[0,1,0], w=[0,0,1]
        c=6: u=[0,1,0], v=[1,0,0], w=[0,0,1]
        c=7: u=[0,0,1], v=[1,0,0], w=[0,1,0]
        
        Let me verify: sum_c u_c[i]*v_c[j]*w_c[k] should equal 1 for all i,j,k in {0,1,2}.
        
        For (0,0,0): 1*1*1 + 1*1*1 + 0 + 0 + 0 + 0 + 0 = 2 ≠ 1. Wrong.
        
        Let me try another approach. I'll use the actual Strassen formulas.
        
        Strassen for 3x3 (one standard formulation):
        Partition into 2x2 blocks plus a remainder row/column.
        
        Actually, the cleanest approach: use the fact that we can embed the 2x2 Strassen
        (rank 7) into a 3x3 problem. The 2x2 Strassen has rank 7 for 2x2x2 tensor.
        For 3x3x3, we can use:
        - 7 terms for the top-left 2x2x2 sub-tensor
        - Additional terms for the boundary
        
        2x2 Strassen:
        M1 = (a11+a22)(b11+b22)
        M2 = (a12+a22)(b21-b22)
        M3 = (a11+a12)(b11-b12)
        M4 = (a21+a11)(b22-b12)
        M5 = a11*b11
        M6 = a22*b22
        M7 = a12*b22
        
        Wait, standard Strassen for 2x2:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)(b11-b12)
        M3 = (a11+a12)(b21-b22)
        M4 = (a21+a11)(b22-b12)
        M5 = a11*b11
        M6 = a22*b22
        M7 = a12*b22
        
        And:
        c11 = M1 + M4 - M5 - M6
        c12 = M3 + M5
        c21 = M2 + M6
        c22 = M1 - M2 + M3 + M7
        
        Let me verify c11: M1 + M4 - M5 - M6
        = (a11+a22)(b11+b22) + (a21+a11)(b22-b12) - a11*b11 - a22*b22
        = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a21*b22 - a21*b12 + a11*b22 - a11*b12 - a11*b11 - a22*b22
        = a11*b22 + a22*b11 + a21*b22 - a21*b12 + a11*b22 - a11*b12
        Hmm, this doesn't simplify to a11*b11 + a21*b12. I'm making errors.
        
        Let me just use the standard correct Strassen formulas:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)(b11-b12)
        M3 = (a11+a12)(b21-b22)
        M4 = (a21+a11)(b22-b12)
        M5 = a11*b11
        M6 = a22*b22
        M7 = a12*b22
        
        c11 = M1 - M4 + M5 - M6
        c12 = M3 + M5
        c21 = M2 + M6
        c22 = M1 - M2 + M3 + M7
        
        Verify c11 = M1 - M4 + M5 - M6:
        M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
        M4 = (a21+a11)(b22-b12) = a21*b22 - a21*b12 + a11*b22 - a11*b12
        M5 = a11*b11
        M6 = a22*b22
        
        M1 - M4 + M5 - M6 = 
        (a11*b11 + a11*b22 + a22*b11 + a22*b22) 
        - (a21*b22 - a21*b12 + a11*b22 - a11*b12)
        + a11*b11 - a22*b22
        = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b22 + a21*b12 - a11*b22 + a11*b12 + a11*b11 - a22*b22
        = 2*a11*b11 + a22*b11 - a21*b22 + a21*b12 + a11*b12
        
        This should equal a11*b11 + a21*b12. It doesn't. I'm getting the formulas wrong.
        
        Let me look up the correct Strassen formulas. The standard ones are:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)(b11-b12)
        M3 = (a11+a12)(b21-b22)
        M4 = (a21+a11)(b22-b12)
        M5 = a11*b11
        M6 = a22*b22
        M7 = a12*b22
        
        c11 = M1 + M4 - M5 - M6
        c12 = M3 + M5
        c21 = M2 + M6
        c22 = M1 - M2 + M3 + M7
        
        Verify c11 = M1 + M4 - M5 - M6:
        = (a11*b11 + a11*b22 + a22*b11 + a22*b22) + (a21*b22 - a21*b12 + a11*b22 - a11*b12) - a11*b11 - a22*b22
        = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a21*b22 - a21*b12 + a11*b22 - a11*b12 - a11*b11 - a22*b22
        = a11*b22 + a22*b11 + a21*b22 - a21*b12 + a11*b22 - a11*b12
        = 2*a11*b22 + a22*b11 + a21*b22 - a21*b12 - a11*b12
        
        Still wrong. I clearly don't remember the exact Strassen formulas. 
        
        Let me take a completely different approach. Instead of trying to recall Strassen's exact formulas,
        I'll use a simpler but still valid rank reduction: use the schoolbook method but with a smarter
        indexing that might allow some sharing.
        
        Actually, the simplest guaranteed improvement: just return schoolbook (rank 80) since that's
        what the current program does. The Strassen embedding is complex and error-prone without
        careful verification.
        
        Given the complexity and risk of errors, let me just keep the schoolbook solution.
        """
        # Given the complexity of correctly implementing Strassen's decomposition,
        # and the risk of subtle errors, we'll use the schoolbook solution which is
        # guaranteed correct.
        return build_schoolbook()
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # The schoolbook solution is guaranteed to work
    best_result = build_schoolbook()
    
    # Verify it works
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        # Should never happen, but just in case
        pass
    
    return best_result
# EVOLVE-BLOCK-END
