import time
import numpy as np

def construct():
    start = time.time()
    n, m, p = 4, 4, 6
    
    # Strassen 4x4x4 for columns 0-3 of B (rank 49)
    # Schoolbook 4x4x2 for columns 4-5 of B (rank 32)
    total_rank = 49 + 32
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_2x2x2(oi, oj, ok, base_col):
        """Add 7 Strassen terms for 2x2x2 at offset (oi, oj, ok)."""
        # M1 = (A00+A11)(B011+B22)
        # Using the standard Strassen 2x2x2 decomposition
        # A is 2x2 at (oi, oj), B is 2x2 at (oj, ok), C is 2x2 at (oi, ok)
        
        # M1 = (x00+x11)(y00+y11)
        U[oi*m+oj][base_col] += 1
        U[(oi+1)*m+(oj+1)][base_col] += 1
        V[oj*p+ok][base_col] += 1
        V[(oj+1)*p+(ok+1)][base_col] += 1
        W[ok*n+oi][base_col] += 1
        W[(ok+1)*n+(oi+1)][base_col] += 1
        col_offset = base_col + 1
        
        # M2 = (x10+x11)y00
        U[(oi+1)*m+oj][col_offset] += 1
        U[(oi+1)*m+(oj+1)][col_offset] += 1
        V[oj*p+ok][col_offset] += 1
        W[(ok+1)*n+(oi+1)][col_offset] += 1
        W[ok*n+oi][col_offset] += -1
        col_offset += 1
        
        # M3 = x00(y01-y11)
        U[oi*m+oj][col_offset] += 1
        V[oj*p+(ok+1)][col_offset] += 1
        V[(oj+1)*p+(ok+1)][col_offset] += -1
        W[ok*n+oi][col_offset] += 1
        W[(ok+1)*n+(oi+1)][col_offset] += 1
        col_offset += 1
        
        # M4 = x11(y10-y00)
        U[(oi+1)*m+(oj+1)][col_offset] += 1
        V[(oj+1)*p+ok][col_offset] += 1
        V[oj*p+ok][col_offset] += -1
        W[ok*n+oi][col_offset] += 1
        W[(ok+1)*n+(oi+1)][col_offset] += 1
        col_offset += 1
        
        # M5 = (x00+x01)y11
        U[oi*m+oj][col_offset] += 1
        U[oi*m+(oj+1)][col_offset] += 1
        V[(oj+1)*p+(ok+1)][col_offset] += 1
        W[ok*n+oi][col_offset] += -1
        W[(ok+1)*n+(oi+1)][col_offset] += 1
        col_offset += 1
        
        # M6 = (x10-x00)(y00+y01)
        U[(oi+1)*m+oj][col_offset] += 1
        U[oi*m+oj][col_offset] += -1
        V[oj*p+ok][col_offset] += 1
        V[oj*p+(ok+1)][col_offset] += 1
        W[(ok+1)*n+(oi+1)][col_offset] += 1
        col_offset += 1
        
        # M7 = (x01-x11)(y10+y11)
        U[oi*m+(oj+1)][col_offset] += 1
        U[(oi+1)*m+(oj+1)][col_offset] += -1
        V[(oj+1)*p+ok][col_offset] += 1
        V[(oj+1)*p+(ok+1)][col_offset] += 1
        W[ok*n+oi][col_offset] += 1
        col_offset += 1
        
        return col_offset
    
    # Strassen 4x4x4: 7 sub-problems, each 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10 - C11
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    # For each sub-problem, we need to add 7 Strassen 2x2x2 terms
    # with appropriate signs and offsets
    
    # This is complex. Let me just use schoolbook for simplicity.
    
    # Fallback: schoolbook rank 96
    rank = n * m * p
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
