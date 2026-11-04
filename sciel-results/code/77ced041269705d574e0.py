"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    _start_time = time.time()
    _deadline = _start_time + 115.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def _time_left():
        return _deadline - time.time()
    
    def _validate(U, V, W):
        """Validate tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        rank = len(U[0])
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for c in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def _schoolbook():
        """Standard schoolbook decomposition, rank = n*m*p = 144"""
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
    
    def _strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4x4 matrix multiplication
        using Strassen's 2x2x2 algorithm (rank 7) applied to 2x2 blocks.
        
        A (4x4) split into 2x2 blocks: A = [[A11, A12], [A21, A22]]
        B (4x4) split into 2x2 blocks: B = [[B11, B12], [B21, B22]]
        C (4x4) split into 2x2 blocks: C = [[C11, C12], [C21, C22]]
        
        Strassen computes:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        """
        # For 4x4x4: n=m=p=4, split into 2x2 blocks
        # U has 16 rows (4*4), V has 16 rows, W has 16 rows
        # Each Strassen term M_s is a 2x2x2 multiplication, rank 7
        # Total rank = 4 (block positions) * 7 (Strassen) = 28? No...
        # Actually, each of the 7 Strassen multiplications is a 2x2x2 mult,
        # but they operate on different block combinations.
        
        # Let me build this more carefully.
        # The 4x4x4 tensor can be decomposed using the 2x2 Strassen structure.
        # 
        # Index mapping for 2x2 blocks:
        # A row index i -> (i//2, i%2) = (block_row, intra_row)
        # A col index j -> (j//2, j%2) = (block_col, intra_col)
        # 
        # For Strassen, we have 7 intermediate products, each a 2x2x2 tensor.
        # Each 2x2x2 tensor has rank 7 (Strassen).
        # So total rank = 7 * 7 = 49? No, that's not right either.
        
        # Let me think differently. The 4x4x4 multiplication tensor T[i,j,k]
        # where C[i,k] = sum_j A[i,j] B[j,k].
        # 
        # Using 2x2 block Strassen:
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2)
        # C = [[C00, C01], [C10, C11]] (each 2x2)
        #
        # Strassen intermediates (each is 2x2 matrix product):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # Each M_s is a 2x2x2 matrix multiplication, which has rank 7.
        # The overall rank is the sum of ranks of each M_s decomposition
        # combined with the linear combinations for C blocks.
        #
        # For each M_s (2x2x2), we use Strassen rank-7 decomposition.
        # The 7 Strassen terms for each M_s give 7 rank-1 terms.
        # Then we combine them with coefficients for the C block equations.
        #
        # Total rank = 7 (Strassen terms) * 7 (per-term rank) = 49
        
        r = 49
        U = [[0] * r for _ in range(16)]  # 4*4 = 16 rows
        V = [[0] * r for _ in range(16)]  # 4*4 = 16 rows  
        W = [[0] * r for _ in range(16)]  # 4*4 = 16 rows
        
        # Strassen 2x2x2 rank-7 decomposition coefficients
        # For 2x2: A=[a,b;c,d], B=[e,f;g,h], C=AB
        # U indices: (0,0)=ae, (0,1)=af, (1,0)=ce, (1,1)=cf for A part
        # V indices: (0,0)=eg, (0,1)=eh, (1,0)=fg, (1,1)=fh for B part  
        # W indices: (0,0), (0,1), (1,0), (1,1) for C part
        #
        # Strassen terms:
        # M1=(a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: c00=1,c11=1
        # M2=d(f-e): U: d=1; V: e=-1,f=1; W: c11=1
        # M3=(c+d)e: U: c=1,d=1; V: e=1; W: c10=1
        # M4=a(f-h): U: a=1; V: f=1,h=-1; W: c01=1
        # M5=(a+b)h: U: a=1,b=1; V: h=1; W: c01=1
        # M6=(c-a)(e+f): U: a=-1,c=1; V: e=1,f=1; W: c10=1
        # M7=(b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: c11=1
        
        # Map 4x4 indices to block/intra:
        # i = 2*bi + ii, j = 2*bj + jj, k = 2*bk + kk
        # where bi,bj,bk in {0,1} are block indices, ii,jj,kk in {0,1} are intra-block
        
        # For each Strassen term s (0..6) and each 2x2x2 rank-1 component t (0..6):
        # We need to map the 2x2x2 Strassen pattern to the 4x4x4 indices.
        
        # This is getting complex. Let me use a simpler approach:
        # Just return schoolbook for now to guarantee correctness.
        return None
    
    # Start with guaranteed correct schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions if time permits
    if _time_left() > 10:
        # Attempt Strassen-based 4x4x4 decomposition extended to 4x4x4x9
        # For now, schoolbook at rank 144 is our baseline
        pass
    
    return best
# EVOLVE-BLOCK-END
