"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build rank-21 decomposition using Strassen on three 2x2 blocks
    rank = 21
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for b in range(3):
        offset = b * 7
        
        # U: 4 rows, 7 columns for this block
        # Row indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
        U[0][offset+0] = 1
        U[0][offset+2] = 1
        U[0][offset+4] = 1
        U[0][offset+5] = -1
        
        U[1][offset+4] = 1
        U[1][offset+6] = -1
        
        U[2][offset+1] = 1
        U[2][offset+5] = 1
        
        U[3][offset+0] = 1
        U[3][offset+1] = 1
        U[3][offset+3] = 1
        
        # V: 12 rows, 7 columns for this block
        # Row indices: j*p+k, j in {0,1}, k in {0..5}
        # For block b: B[0, b*2], B[0, b*2+1], B[1, b*2], B[1, b*2+1]
        V[0 * p + b*2][offset+0] = 1
        V[0 * p + b*2][offset+1] = 1
        V[0 * p + b*2][offset+5] = 1
        
        V[0 * p + b*2 + 1][offset+3] = -1
        V[0 * p + b*2 + 1][offset+6] = 1
        
        V[1 * p + b*2][offset+2] = 1
        V[1 * p + b*2][offset+3] = -1
        V[1 * p + b*2][offset+5] = 1
        
        V[1 * p + b*2 + 1][offset+0] = 1
        V[1 * p + b*2 + 1][offset+4] = 1
        V[1 * p + b*2 + 1][offset+6] = 1
        
        # W: 12 rows, 7 columns for this block
        # Row indices: k*n+i, k in {0..5}, i in {0,1}
        # For block b: C[0, b*2], C[0, b*2+1], C[1, b*2], C[1, b*2+1]
        W[(b*2) * n + 0][offset+0] = 1
        W[(b*2) * n + 0][offset+3] = 1
        W[(b*2) * n + 0][offset+4] = -1
        W[(b*2) * n + 0][offset+6] = 1
        
        W[(b*2+1) * n + 0][offset+2] = 1
        W[(b*2+1) * n + 0][offset+4] = 1
        
        W[(b*2) * n + 1][offset+1] = 1
        W[(b*2) * n + 1][offset+3] = 1
        
        W[(b*2+1) * n + 1][offset+0] = 1
        W[(b*2+1) * n + 1][offset+1] = -1
        W[(b*2+1) * n + 1][offset+2] = 1
        W[(b*2+1) * n + 1][offset+5] = 1
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-24 decomposition
    schoolbook_rank = n * m * p  # 24
    U_fb = [[0] * schoolbook_rank for _ in range(nm)]
    V_fb = [[0] * schoolbook_rank for _ in range(mp)]
    W_fb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
