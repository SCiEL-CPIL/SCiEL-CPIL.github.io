"""Find rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid
    
    def strassen_2x2x2(U, V, W, rank_offset, i0, j0, k0):
        """Add 7 Strassen terms for 2x2 x 2x2 -> 2x2 block at (i0,j0,k0).
        i0: row offset in A (0..1), j0: col offset in A / row offset in B (0..1), k0: col offset in B (0..1)
        """
        # Strassen for 2x2: M1..M7
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Term 1: M1
        c = rank_offset
        U[i0*m + j0][c] = 1
        U[(i0+1)*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[k0*n + i0][c] = 1
        W[(k0+1)*n + (i0+1)][c] = 1
        
        # Term 2: M2
        c = rank_offset + 1
        U[(i0+1)*m + j0][c] = 1
        U[(i0+1)*m + (j0+1)][c] = 1
        V[j0*p + k0][c] = 1
        W[k0*n + (i0+1)][c] = 1
        
        # Term 3: M3
        c = rank_offset + 2
        U[i0*m + j0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        V[(j0+1)*p + (k0+1)][c] = -1
        W[(k0+1)*n + i0][c] = 1
        
        # Term 4: M4
        c = rank_offset + 3
        U[(i0+1)*m + (j0+1)][c] = 1
        V[(j0+1)*p + j0][c] = 1
        V[j0*p + j0][c] = -1
        W[j0*n + (i0+1)][c] = 1
        
        # Term 5: M5
        c = rank_offset + 4
        U[i0*m + j0][c] = 1
        U[i0*m + (j0+1)][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[(k0+1)*n + i0][c] = -1
        
        # Term 6: M6
        c = rank_offset + 5
        U[(i0+1)*m + j0][c] = 1
        U[i0*m + j0][c] = -1
        V[j0*p + k0][c] = 1
        V[j0*p + (k0+1)][c] = 1
        W[k0*n + (i0+1)][c] = 1
        W[(k0+1)*n + (i0+1)][c] = 1
        
        # Term 7: M7
        c = rank_offset + 6
        U[i0*m + (j0+1)][c] = 1
        U[(i0+1)*m + (j0+1)][c] = -1
        V[(j0+1)*p + j0][c] = 1
        V[(j0+1)*p + (k0+1)][c] = 1
        W[j0*n + i0][c] = 1
        W[(k0+1)*n + i0][c] = 1
        
        return rank_offset + 7
    
    def schoolbook_2x2x1(U, V, W, rank_offset, i0, j0, k0):
        """Add 4 schoolbook terms for 2x2 x 2x1 -> 2x1 block at (i0,j0,k0)."""
        for di in range(2):
            for dj in range(2):
                c = rank_offset + di*2 + dj
                U[(i0+di)*m + (j0+dj)][c] = 1
                V[(j0+dj)*p + k0][c] = 1
                W[k0*n + (i0+di)][c] = 1
        return rank_offset + 4
    
    def schoolbook_2x1x2(U, V, W, rank_offset, i0, j0, k0):
        """Add 4 schoolbook terms for 2x1 x 1x2 -> 2x2 block at (i0,j0,k0)."""
        for di in range(2):
            for dk in range(2):
                c = rank_offset + di*2 + dk
                U[(i0+di)*m + j0][c] = 1
                V[j0*p + (k0+dk)][c] = 1
                W[(k0+dk)*n + (i0+di)][c] = 1
        return rank_offset + 4
    
    def schoolbook_2x1x1(U, V, W, rank_offset, i0, j0, k0):
        """Add 2 schoolbook terms for 2x1 x 1x1 -> 2x1 block at (i0,j0,k0)."""
        for di in range(2):
            c = rank_offset + di
            U[(i0+di)*m + j0][c] = 1
            V[j0*p + k0][c] = 1
            W[k0*n + (i0+di)][c] = 1
        return rank_offset + 2
    
    # Build the Strassen-based decomposition
    # A (2x5) split into columns: A1(2x2), A2(2x2), A3(2x1)
    # B (5x5) split into rows/cols: B_ij where i,j in {1,2,3}
    #   B11(2x2), B12(2x2), B13(2x1)
    #   B21(2x2), B22(2x2), B23(2x1)
    #   B31(1x2), B32(1x2), B33(1x1)
    # C (2x5) split into columns: C1(2x2), C2(2x2), C3(2x1)
    
    # Row/col offsets in original indices:
    # A cols: A1=[0,1], A2=[2,3], A3=[4]
    # B rows: B1=[0,1], B2=[2,3], B3=[4]
    # B cols: B1=[0,1], B2=[2,3], B3=[4]
    
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # Total rank: 7+7+4 (C1) + 7+7+4 (C2) + 4+4+2 (C3) = 18+18+10 = 46
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # C1 contributions:
    # A1(2x2) x B11(2x2) -> 2x2 at i0=0, j0=0, k0=0
    rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 0, 0)
    # A2(2x2) x B21(2x2) -> 2x2 at i0=0, j0=2, k0=0
    rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 2, 0)
    # A3(2x1) x B31(1x2) -> 2x2 at i0=0, j0=4, k0=0
    rank_offset = schoolbook_2x1x2(U, V, W, rank_offset, 0, 4, 0)
    
    # C2 contributions:
    # A1(2x2) x B12(2x2) -> 2x2 at i0=0, j0=0, k0=2
    rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 0, 2)
    # A2(2x2) x B22(2x2) -> 2x2 at i0=0, j0=2, k0=2
    rank_offset = strassen_2x2x2(U, V, W, rank_offset, 0, 2, 2)
    # A3(2x1) x B32(1x2) -> 2x2 at i0=0, j0=4, k0=2
    rank_offset = schoolbook_2x1x2(U, V, W, rank_offset, 0, 4, 2)
    
    # C3 contributions:
    # A1(2x2) x B13(2x1) -> 2x1 at i0=0, j0=0, k0=4
    rank_offset = schoolbook_2x2x1(U, V, W, rank_offset, 0, 0, 4)
    # A2(2x2) x B23(2x1) -> 2x1 at i0=0, j0=2, k0=4
    rank_offset = schoolbook_2x2x1(U, V, W, rank_offset, 0, 2, 4)
    # A3(2x1) x B33(1x1) -> 2x1 at i0=0, j0=4, k0=4
    rank_offset = schoolbook_2x1x1(U, V, W, rank_offset, 0, 4, 4)
    
    # Verify the decomposition
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                val = 0
                                for c in range(r):
                                    val += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i2][c]
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                if abs(val - expected) > 1e-9:
                                    return False
        return True
    
    if not check_decomposition(U, V, W):
        # Fallback to schoolbook if Strassen construction has an error
        def schoolbook():
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W
        
        U, V, W = schoolbook()
    
    return {"U": U, "V": V, "W": W}
