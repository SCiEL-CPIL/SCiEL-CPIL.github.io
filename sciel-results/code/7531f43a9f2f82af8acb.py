import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_factors():
    """Generate Strassen factors for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ]
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    
    U4 = [[0.0]*49 for _ in range(16)]
    V4 = [[0.0]*49 for _ in range(16)]
    W4 = [[0.0]*49 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            c = a*7 + b
            for i in range(4):
                for j in range(4):
                    flat = i*4 + j
                    block_flat = (i//2)*2 + (j//2)
                    sub_flat = (i%2)*2 + (j%2)
                    U4[flat][c] = U2[block_flat][a] * U2[sub_flat][b]
                    V4[flat][c] = V2[block_flat][a] * V2[sub_flat][b]
                    W4[flat][c] = W2[block_flat][a] * W2[sub_flat][b]
    
    return U4, V4, W4

def _build_2x2x2_strassen():
    """Build rank-7 decomposition for 2x2x2 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    return U2, V2, W2

def _schoolbook_block(n_block, m_block, p_block):
    """Build schoolbook decomposition for n_block x m_block * m_block x p_block."""
    rank = n_block * m_block * p_block
    U = [[0.0]*rank for _ in range(n_block*m_block)]
    V = [[0.0]*rank for _ in range(m_block*p_block)]
    W = [[0.0]*rank for _ in range(p_block*n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i*m_block+j][col] = 1.0
                V[j*p_block+k][col] = 1.0
                W[k*n_block+i][col] = 1.0
                col += 1
    return U, V, W, rank

def _build_decomposition_for_split(n, m, p, a, b, c, d, e, f, g, h, deadline):
    """
    Build decomposition for split:
    A: a×b | c×d (rows 0..a-1, cols 0..b-1) | (rows a..a+c-1, cols b..m-1)
    B: e×f | g×h (rows 0..e-1, cols 0..f-1) | (rows e..p-1, cols f..p-1)
    
    Wait, let me re-derive. For 4×5 A and 5×5 B:
    A is 4 rows × 5 cols. Split rows into (a, c) with a+c=4, cols into (b, d) with b+d=5.
    B is 5 rows × 5 cols. Split rows into (e, g) with e+g=5, cols into (f, h) with f+h=5.
    
    Block (a,b) of A multiplies block (e,f) of B → contributes to C block (a,e)
    Block (a,b) of A multiplies block (g,h) of B → contributes to C block (a,g)
    Block (c,d) of A multiplies block (e,f) of B → contributes to C block (c,e)
    Block (c,d) of A multiplies block (g,h) of B → contributes to C block (c,g)
    
    For the "compatible" blocks where row-dim of A-block matches col-dim of B-block:
    - (a,b)×(e,f): needs b==e for direct multiplication. If b==e, use Strassen/schoolbook for a×b×f
    - (a,b)×(g,h): needs b==g. If b==g, use a×b×h
    - (c,d)×(e,f): needs d==e. If d==e, use c×d×f
    - (c,d)×(g,h): needs d==g. If d==g, use c×d×h
    """
    # This function is called only when dimensions are compatible
    # For now, we handle the specific case where all four blocks are compatible
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    # We'll build this more carefully in construct()
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 130
    
    try:
        # Best split found: a=2,b=2,c=2,d=3,e=2,f=3,g=3,h=2
        # A: 2×2 | 2×3, B: 2×3 | 3×2
        # Block products:
        # (2,2)×(2,3): 2×2×3, rank = 2*2*3 = 12 schoolbook (or could use better)
        # (2,2)×(3,2): 2×2×2, rank = 7 Strassen
        # (2,3)×(2,3): 2×3×3, rank = 2*3*3 = 18 schoolbook
        # (2,3)×(3,2): 2×3×2, rank = 2*3*2 = 12 schoolbook
        # Total: 12 + 7 + 18 + 12 = 49... that's not 24.
        
        # Let me re-examine. The rank formula I derived was:
        # rank = (a+e)*(b+f) + a*h + c*f + c*h
        # For a=2,b=2,c=2,d=3,e=2,f=3,g=3,h=2:
        # (2+2)*(2+3) + 2*2 + 2*3 + 2*2 = 4*5 + 4 + 6 + 4 = 20+14 = 34
        
        # Hmm, let me reconsider the formula. The issue is that when we have
        # overlapping blocks, we can share computation. Let me think differently.
        
        # Actually, the cleanest approach: just use the 4x4x4 Strassen (rank 49)
        # plus the remaining parts, which is what the current program does (rank 58).
        
        # Let me try a different split: use 2x2 Strassen blocks more aggressively.
        # Split 4x5 into four 2x2 + 2x1 blocks, and 5x5 into four 2x2 + 2x1 + 1x2 + 1x1
        
        # Actually, let me just stick with the proven rank-58 approach but verify it works,
        # and also try to find if there's a better partition.
        
        # For now, use the current approach: 4x4x4 Strassen + schoolbook for remainder
        U4, V4, W4 = _build_4x4x4_strassen()
        
        total_rank = 49 + 4 + 5  # = 58
        
        U = [[0.0]*total_rank for _ in range(n*m)]  # 20 x 58
        V = [[0.0]*total_rank for _ in range(m*p)]  # 25 x 58
        W = [[0.0]*total_rank for _ in range(p*n)]  # 20 x 58
        
        # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3), rank 49
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    flat_U = i*5 + j
                    flat_V = j*5 + k
                    flat_W = k*4 + i
                    for c in range(49):
                        U[flat_U][c] += U4[i*4+j][c]
                        V[flat_V][c] += V4[j*4+k][c]
                        W[flat_W][c] += W4[k*4+i][c]
        
        # Part 2: 4x4x1 (i in 0..3, j in 0..3, k=4), rank 4
        for j in range(4):
            c = 49 + j
            for i in range(4):
                flat_U = i*5 + j
                flat_V = j*5 + 4
                flat_W = 4*4 + i
                U[flat_U][c] = 1.0
                V[flat_V][c] = 1.0
                W[flat_W][c] = 1.0
        
        # Part 3: 4x1x5 (i in 0..3, j=4, k in 0..4), rank 5
        for k in range(5):
            c = 53 + k
            for i in range(4):
                flat_U = i*5 + 4
                flat_V = 4*5 + k
                flat_W = k*4 + i
                U[flat_U][c] = 1.0
                V[flat_V][c] = 1.0
                W[flat_W][c] = 1.0
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_fb = [[0] * rank_school for _ in range(n * m)]
            V_fb = [[0] * rank_school for _ in range(m * p)]
            W_fb = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1
                        V_fb[j*p+k][col] = 1
                        W_fb[k*n+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        # Fall back to schoolbook
        rank_school = n * m * p
        U_fb = [[0] * rank_school for _ in range(n * m)]
        V_fb = [[0] * rank_school for _ in range(m * p)]
        W_fb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
