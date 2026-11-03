"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 8 matrix.
    
    Strategy: Split m=3 into m=2 + m=1.
    - 2x2x8 part: 4 blocks of 2x2x2, each using Strassen (rank 7) -> 28 terms
    - 2x1x8 part: 4 blocks of 2x1x2, each using rank-3 decomposition -> 12 terms
    Total rank: 28 + 12 = 40
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill
    
    def verify(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition."""
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
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2x2():
        """
        Return Strassen decomposition for 2x2x2 matrix multiplication.
        Returns lists of (i,j,k) -> coefficient for U, V, W.
        Strassen: 7 multiplications
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # Each term: (U_idx, V_idx, W_idx, coeff)
        # U indices: (i,j) where i in {0,1}, j in {0,1} -> 0,1,2,3
        # V indices: (j,k) where j in {0,1}, k in {0,1} -> 0,1,2,3
        # W indices: (k,i) where k in {0,1}, i in {0,1} -> 0,1,2,3
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=0, A11=3; V: B00=0, B11=3; W: C00=0, C11=3
        # C00 = M1, C11 = M1
        terms.append((0, 0, 0, 1))   # A00*B00 -> C00
        terms.append((0, 3, 3, 1))   # A00*B11 -> C11
        terms.append((3, 0, 0, 1))   # A11*B00 -> C00
        terms.append((3, 3, 3, 1))   # A11*B11 -> C11
        
        # M2 = (A10+A11)B00
        # U: A10=2, A11=3; V: B00=0; W: C10=2, C11=3
        # C10 = M2, C11 = -M2
        terms.append((2, 0, 2, 1))   # A10*B00 -> C10
        terms.append((3, 0, 2, 1))   # A11*B00 -> C10
        terms.append((2, 0, 3, -1))  # A10*B00 -> -C11
        terms.append((3, 0, 3, -1))  # A11*B00 -> -C11
        
        # M3 = A00(B10-B11)
        # U: A00=0; V: B10=1, B11=3; W: C01=1, C10=2
        # C01 = M3, C10 = M3
        terms.append((0, 1, 1, 1))   # A00*B10 -> C01
        terms.append((0, 3, 1, -1))  # A00*B11 -> -C01
        terms.append((0, 1, 2, 1))   # A00*B10 -> C10
        terms.append((0, 3, 2, -1))  # A00*B11 -> -C10
        
        # M4 = A11(B01-B00)
        # U: A11=3; V: B01=2, B00=0; W: C00=0, C10=2
        # C00 = M4, C10 = M4
        terms.append((3, 2, 0, 1))   # A11*B01 -> C00
        terms.append((3, 0, 0, -1))  # A11*B00 -> -C00
        terms.append((3, 2, 2, 1))   # A11*B01 -> C10
        terms.append((3, 0, 2, -1))  # A11*B00 -> -C10
        
        # M5 = (A00+A01)B11
        # U: A00=0, A01=1; V: B11=3; W: C00=0, C01=1
        # C00 = -M5, C01 = M5
        terms.append((0, 3, 0, -1))  # A00*B11 -> -C00
        terms.append((1, 3, 0, -1))  # A01*B11 -> -C00
        terms.append((0, 3, 1, 1))   # A00*B11 -> C01
        terms.append((1, 3, 1, 1))   # A01*B11 -> C01
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10=2, A00=0; V: B00=0, B10=1; W: C11=3
        # C11 = M6
        terms.append((2, 0, 3, 1))   # A10*B00 -> C11
        terms.append((0, 0, 3, -1))  # A00*B00 -> -C11
        terms.append((2, 1, 3, 1))   # A10*B10 -> C11
        terms.append((0, 1, 3, -1))  # A00*B10 -> -C11
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01=1, A11=3; V: B01=2, B11=3; W: C00=0
        # C00 = M7
        terms.append((1, 2, 0, 1))   # A01*B01 -> C00
        terms.append((3, 2, 0, -1))  # A11*B01 -> -C00
        terms.append((1, 3, 0, 1))   # A01*B11 -> C00
        terms.append((3, 3, 0, -1))  # A11*B11 -> -C00
        
        return terms
    
    def rank3_2x1x2():
        """
        Return rank-3 decomposition for 2x1x2 (column times row).
        C[0,0] = A[0,0]*B[0,0]
        C[0,1] = A[0,0]*B[0,1]
        C[1,0] = A[1,0]*B[0,0]
        C[1,1] = A[1,0]*B[0,1]
        
        Rank-3 decomposition:
        M1 = A[0,0]*B[0,0]
        M2 = A[1,0]*B[0,1]
        M3 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1])
        
        C[0,0] = M1
        C[1,1] = M2
        C[0,1] = M3 - M1 - M2
        C[1,0] = M3 - M1 - M2
        
        U indices: (i,0) where i in {0,1} -> 0, 1
        V indices: (0,k) where k in {0,1} -> 0, 1
        W indices: (k,i) where k in {0,1}, i in {0,1} -> 0,1,2,3
        W: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        """
        terms = []
        
        # M1 = A[0,0]*B[0,0] -> C[0,0]
        terms.append((0, 0, 0, 1))
        
        # M2 = A[1,0]*B[0,1] -> C[1,1]
        terms.append((1, 1, 3, 1))
        
        # M3 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1]) -> contributes to C[0,1] and C[1,0]
        # C[0,1] = M3 - M1 - M2
        # C[1,0] = M3 - M1 - M2
        terms.append((0, 0, 1, 1))   # A[0,0]*B[0,0] -> C[0,1]
        terms.append((0, 1, 1, 1))   # A[0,0]*B[0,1] -> C[0,1]
        terms.append((1, 0, 1, 1))   # A[1,0]*B[0,0] -> C[0,1]
        terms.append((1, 1, 1, 1))   # A[1,0]*B[0,1] -> C[0,1]
        
        terms.append((0, 0, 2, 1))   # A[0,0]*B[0,0] -> C[1,0]
        terms.append((0, 1, 2, 1))   # A[0,0]*B[0,1] -> C[1,0]
        terms.append((1, 0, 2, 1))   # A[1,0]*B[0,0] -> C[1,0]
        terms.append((1, 1, 2, 1))   # A[1,0]*B[0,1] -> C[1,0]
        
        # Now subtract M1 and M2 from C[0,1] and C[1,0]
        # M1 = A[0,0]*B[0,0]: subtract from C[0,1] and C[1,0]
        terms.append((0, 0, 1, -1))  # -A[0,0]*B[0,0] -> C[0,1]
        terms.append((0, 0, 2, -1))  # -A[0,0]*B[0,0] -> C[1,0]
        
        # M2 = A[1,0]*B[0,1]: subtract from C[0,1] and C[1,0]
        terms.append((1, 1, 1, -1))  # -A[1,0]*B[0,1] -> C[0,1]
        terms.append((1, 1, 2, -1))  # -A[1,0]*B[0,1] -> C[1,0]
        
        return terms
    
    # Build the rank-40 decomposition
    # Total rank: 28 (Strassen for 2x2x8) + 12 (rank-3 for 2x1x8) = 40
    
    U = [[0] * 40 for _ in range(n * m)]  # 6 x 40
    V = [[0] * 40 for _ in range(m * p)]  # 24 x 40
    W = [[0] * 40 for _ in range(p * n)]  # 16 x 40
    
    col = 0
    
    # Part 1: 2x2x8 using 4 Strassen blocks
    # j in {0,1}, k_block in {0,1,2,3} where k = 2*k_block + {0,1}
    for k_block in range(4):
        strassen_terms = strassen_2x2x2()
        for (u_idx, v_idx, w_idx, coeff) in strassen_terms:
            # u_idx: (i,j) where j in {0,1} -> U row index = i * m + j
            i = u_idx // 2
            j = u_idx % 2
            u_row = i * m + j  # i in {0,1}, j in {0,1}
            
            # v_idx: (j,k) where j in {0,1}, k in {0,1} -> V row index = j * p + (2*k_block + k)
            jj = v_idx // 2
            kk = v_idx % 2
            v_row = jj * p + (2 * k_block + kk)
            
            # w_idx: (k,i) where k in {0,1}, i in {0,1} -> W row index = (2*k_block + k) * n + i
            kk2 = w_idx // 2
            ii = w_idx % 2
            w_row = (2 * k_block + kk2) * n + ii
            
            U[u_row][col] += coeff
            V[v_row][col] += coeff
            W[w_row][col] += coeff
        
        col += 7  # 7 terms per Strassen block
    
    # Part 2: 2x1x8 using 4 rank-3 blocks
    # j=2, k_block in {0,1,2,3} where k = 2*k_block + {0,1}
    for k_block in range(4):
        rank3_terms = rank3_2x1x2()
        for (u_idx, v_idx, w_idx, coeff) in rank3_terms:
            # u_idx: i in {0,1} -> U row index = i * m + 2
            i = u_idx
            u_row = i * m + 2  # j=2
            
            # v_idx: k in {0,1} -> V row index = 2 * p + (2*k_block + k)
            kk = v_idx
            v_row = 2 * p + (2 * k_block + kk)
            
            # w_idx: (k,i) where k in {0,1}, i in {0,1} -> W row index = (2*k_block + k) * n + i
            kk2 = w_idx // 2
            ii = w_idx % 2
            w_row = (2 * k_block + kk2) * n + ii
            
            U[u_row][col] += coeff
            V[v_row][col] += coeff
            W[w_row][col] += coeff
        
        col += 3  # 3 terms per rank-3 block
    
    # Verify
    if verify(U, V, W, 40):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, try to debug and fall back
    # Try schoolbook as fallback
    return schoolbook()
# EVOLVE-BLOCK-END
