"""Tensor decomposition for 3x3 by 3x7 matrix multiplication."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition.
    
    Returns dict with 'U', 'V', 'W' matrices.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Entries can be integers or half-integers (multiples of 1/2).
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Deadline: 145 seconds
    deadline = time.time() + 145
    
    # Verify a decomposition using the trace convention W[k*n+i, r]
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying 2x2 by 2x2:
    # U indices: (i,j) for i,j in {0,1} -> 4 entries
    # V indices: (j,k) for j,k in {0,1} -> 4 entries  
    # W indices: (k,i) for k,i in {0,1} -> 4 entries (trace convention)
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2 x 2x2 multiplication, rank 7.
        U: shape (4, 7), indexed by (i*m+j) = (i*2+j)
        V: shape (4, 7), indexed by (j*p+k) = (j*2+k)
        W: shape (4, 7), indexed by (k*n+i) = (k*2+i)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen formulas:
        # r=0: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00+=1, C11+=1
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11 (k=0,i=0 and k=1,i=1)
        
        # r=1: M2 = (A10+A11)B00
        # U: A10=1, A11=1; V: B00=1; W: C10+=1
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10 (k=0,i=1)
        
        # r=2: M3 = A00(B10-B11)
        # U: A00=1; V: B10=1, B11=-1; W: C01+=1, C11+=1
        U[0][2] = 1  # A00
        V[2][2] = 1; V[3][2] = -1  # B10, B11
        W[1][2] = 1; W[3][2] = 1  # C01 (k=1,i=0), C11 (k=1,i=1)
        
        # r=3: M4 = A11(B01-B00)
        # U: A11=1; V: B01=1, B00=-1; W: C00+=1, C10+=1
        U[3][3] = 1  # A11
        V[1][3] = 1; V[0][3] = -1  # B01, B00
        W[0][3] = 1; W[2][3] = 1  # C00 (k=0,i=0), C10 (k=0,i=1)
        
        # r=4: M5 = (A00+A01)B11
        # U: A00=1, A01=1; V: B11=1; W: C00-=1, C01+=1
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        W[0][4] = -1; W[1][4] = 1  # C00, C01
        
        # r=5: M6 = (A10-A00)(B00+B10)
        # U: A10=1, A00=-1; V: B00=1, B10=1; W: C11+=1
        U[2][5] = 1; U[0][5] = -1  # A10, A00
        V[0][5] = 1; V[2][5] = 1  # B00, B10
        W[3][5] = 1  # C11
        
        # r=6: M7 = (A01-A11)(B01+B11)
        # U: A01=1, A11=-1; V: B01=1, B11=1; W: C00+=1
        U[1][6] = 1; U[3][6] = -1  # A01, A11
        V[1][6] = 1; V[3][6] = 1  # B01, B11
        W[0][6] = 1  # C00
        
        return U, V, W
    
    # 1x2 x 2x1 multiplication (rank 2): A is 1x2, B is 2x1, C is 1x1
    # C = A0*B0 + A1*B1
    # U: shape (2, 2), V: shape (2, 2), W: shape (1, 2)
    def rank2_1x2x2x1():
        U = [[1, 0], [0, 1]]  # A0, A1
        V = [[1, 0], [0, 1]]  # B0, B1
        W = [[1, 1]]           # C0 += both
        return U, V, W
    
    # 1x1 x 1x1 multiplication (rank 1)
    def rank1_1x1():
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    # Build the full 3x3 x 3x7 decomposition
    # Strategy: partition rows of A into {0,1} and {2}
    # Partition columns of B into {0,1}, {2,3}, {4,5}, {6}
    # For each 2-column group: use Strassen 2x2x2 for rows {0,1}, and rank-2 for row {2}
    # For the 1-column group: use rank-3 (schoolbook for 3x1x1)
    
    def build_decomposition():
        """Build 3x3 x 3x7 decomposition using block partitioning.
        
        Partition A rows: {0,1} and {2}
        Partition B cols: {0,1}, {2,3}, {4,5}, {6}
        
        For each 2-col group (k0,k1):
          - Rows {0,1} x cols {k0,k1}: This is a 2x3 x 3x2 problem.
            We decompose it as: (2x2 x 2x2 for j in {0,1}) + (2x1 x 1x2 for j=2)
            - 2x2 x 2x2: use Strassen (rank 7)
            - 2x1 x 1x2: rank 4 (schoolbook: 2*1*2=4)
            Total per 2-col group: 7 + 4 = 11
          - Row {2} x cols {k0,k1}: 1x3 x 3x2, rank 6 (schoolbook)
          Total per 2-col group: 11 + 6 = 17
        
        For the 1-col group (k6):
          - All rows x col {6}: 3x3 x 3x1, rank 9 (schoolbook)
        
        Total: 3*17 + 9 = 51
        """
        terms = []  # list of (u_vec, v_vec, w_vec)
        
        col_groups = [(0, 1), (2, 3), (4, 5), (6,)]
        
        for cg in col_groups:
            if len(cg) == 2:
                k0, k1 = cg
                
                # Part 1: Rows {0,1}, j in {0,1}, cols {k0,k1}: Strassen 2x2x2x2
                sU, sV, sW = strassen_2x2()
                for r in range(7):
                    u_vec = [0] * 9
                    v_vec = [0] * 21
                    w_vec = [0] * 21
                    for si in range(2):
                        for sj in range(2):
                            val = sU[si*2 + sj][r]
                            if val != 0:
                                u_vec[si*3 + sj] += val
                    for sj in range(2):
                        for sk in range(2):
                            val = sV[sj*2 + sk][r]
                            if val != 0:
                                v_vec[sj*7 + (k0 + sk)] += val
                    for sk in range(2):
                        for si in range(2):
                            val = sW[sk*2 + si][r]
                            if val != 0:
                                w_vec[(k0 + sk)*3 + si] += val
                    terms.append((u_vec, v_vec, w_vec))
                
                # Part 2: Rows {0,1}, j=2, cols {k0,k1}: 2x1 x 1x2, rank 4
                # C[i][k] += A[i][2] * B[2][k] for i in {0,1}, k in {k0,k1}
                for i in range(2):
                    for k in [k0, k1]:
                        u_vec = [0] * 9
                        v_vec = [0] * 21
                        w_vec = [0] * 21
                        u_vec[i*3 + 2] = 1
                        v_vec[2*7 + k] = 1
                        w_vec[k*3 + i] = 1
                        terms.append((u_vec, v_vec, w_vec))
                
                # Part 3: Row {2}, all j, cols {k0,k1}: 1x3 x 3x2, rank 6
                for j in range(3):
                    for k in [k0, k1]:
                        u_vec = [0] * 9
                        v_vec = [0] * 21
                        w_vec = [0] * 21
                        u_vec[2*3 + j] = 1
                        v_vec[j*7 + k] = 1
                        w_vec[k*3 + 2] = 1
                        terms.append((u_vec, v_vec, w_vec))
            else:
                # Single column: schoolbook 3x3 x 3x1, rank 9
                k0 = cg[0]
                for i in range(3):
                    for j in range(3):
                        u_vec = [0] * 9
                        v_vec = [0] * 21
                        w_vec = [0] * 21
                        u_vec[i*3 + j] = 1
                        v_vec[j*7 + k0] = 1
                        w_vec[k0*3 + i] = 1
                        terms.append((u_vec, v_vec, w_vec))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(21)]
        W = [[0] * rank for _ in range(21)]
        
        for r, (u_vec, v_vec, w_vec) in enumerate(terms):
            for idx in range(9):
                U[idx][r] = u_vec[idx]
            for idx in range(21):
                V[idx][r] = v_vec[idx]
            for idx in range(21):
                W[idx][r] = w_vec[idx]
        
        return {"U": U, "V": V, "W": W}
    
    result = build_decomposition()
    
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook
    rank = n * m * p  # 63
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
# EVOLVE-BLOCK-END
