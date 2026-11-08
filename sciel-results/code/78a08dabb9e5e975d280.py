"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) lists of shape (4, 7), (4, 7), (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build Strassen decomposition for 4x4 * 4x4 -> 4x4.
    Uses 2-level Strassen: split into 2x2 blocks, apply Strassen at each level.
    Rank = 7 * 7 = 49
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # The 7 outer Strassen products, each applied to 2x2 blocks
    # For each outer product s (0..6) and inner product t (0..6):
    # Column = s*7 + t
    
    # Outer Strassen structure (same as 2x2 but at block level)
    outer_U = [
        [(0,0,1),(1,1,1)],  # M1: (A00+A11)
        [(1,0,1),(1,1,1)],  # M2: (A10+A11)
        [(0,0,1)],           # M3: A00
        [(1,1,1)],           # M4: A11
        [(0,0,1),(0,1,1)],  # M5: (A00+A01)
        [(1,0,1),(0,0,-1)], # M6: (A10-A00)
        [(0,1,1),(1,1,-1)], # M7: (A01-A11)
    ]
    outer_V = [
        [(0,0,1),(1,1,1)],  # (B00+B11)
        [(0,0,1)],           # B00
        [(0,1,1),(1,1,-1)], # (B01-B11)
        [(1,0,1),(0,0,-1)], # (B10-B00)
        [(1,1,1)],           # B11
        [(0,0,1),(0,1,1)],  # (B00+B01)
        [(1,0,1),(1,1,1)],  # (B10+B11)
    ]
    outer_W = [
        [(0,0,1),(1,1,1)],  # (C00+C11)
        [(1,0,1),(1,1,-1)], # (C10-C11)
        [(0,1,1),(1,1,1)],  # (C01+C11)
        [(1,0,1),(0,0,-1)], # (C10-C00)
        [(0,0,-1),(0,1,1)], # (C01-C00)
        [(1,1,1)],           # C11
        [(0,0,1)],           # C00
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Apply outer U factors to inner U
            for (bi, bj, coef) in outer_U[s]:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        inner_val = U2[a_loc][t]
                        if inner_val != 0:
                            U[gi * 4 + gj][col] += inner_val * coef
            
            # Apply outer V factors to inner V
            for (bj, bk, coef) in outer_V[s]:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        inner_val = V2[b_loc][t]
                        if inner_val != 0:
                            V[gj * 4 + gk][col] += inner_val * coef
            
            # Apply outer W factors to inner W
            for (bk, bi, coef) in outer_W[s]:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        inner_val = W2[c_loc][t]
                        if inner_val != 0:
                            W[gk * 4 + gi][col] += inner_val * coef
    
    return U, V, W


def _build_4x4x2_strassen():
    """
    Build decomposition for (4x4) * (4x2) -> (4x2).
    Uses Strassen on the 4x4 part, schoolbook on the 2 columns.
    Rank = 49 * 2 = 98? No, we can do better.
    
    Actually, for 4x4 * 4x2:
    Split B into two 4x1 columns
    For each column, we need 4x4 * 4x1 -> 4x1
    This is rank 16 each (schoolbook), so total 32
    
    But we can use Strassen structure. Let me think...
    Actually, the cleanest is: treat as 4x4 * 4x2
    Split A into 2x2 blocks, B into 2x2 blocks (but B is 4x2, so two 2x1 blocks per row-block)
    
    Simpler: just use schoolbook for the 4x4x2 part: rank 32
    """
    # Schoolbook for 4x4 * 4x2 -> 4x2: rank = 4*4*2 = 32
    U = [[0]*32 for _ in range(16)]
    V = [[0]*32 for _ in range(8)]
    W = [[0]*32 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j][col] = 1
                V[j * 2 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return U, V, W


def _build_4x1x4_outer():
    """
    Build decomposition for (4x1) * (1x4) -> (4x4).
    Rank = 4*4 = 16
    """
    U = [[0]*16 for _ in range(4)]
    V = [[0]*16 for _ in range(4)]
    W = [[0]*16 for _ in range(16)]
    
    col = 0
    for i in range(4):
        for k in range(4):
            U[i][col] = 1
            V[k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return U, V, W


def _build_4x1x2_outer():
    """
    Build decomposition for (4x1) * (1x2) -> (4x2).
    Rank = 4*2 = 8
    """
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(2)]
    W = [[0]*8 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for k in range(2):
            U[i][col] = 1
            V[k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return U, V, W


def _combine_parts(parts, n, m, p):
    """
    Combine multiple decomposition parts into a single decomposition.
    parts: list of (U, V, W, (i0, j0, ii, jj, k0, kk))
    where the part handles A[i0:i0+ii, j0:j0+jj] * B[j0:j0+jj, k0:k0+kk] -> C[i0:i0+ii, k0:k0+kk]
    """
    total_rank = sum(len(part[0][0]) for part in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA, mA = ii, jj  # A is nA x mA
        nB = kk           # B is mA x nB
        
        # Fill U: rows are (i, j) for i in 0..nA-1, j in 0..mA-1
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][offset + c] = Us[r][c]
        
        # Fill V: rows are (j, k) for j in 0..mA-1, k in 0..nB-1
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][offset + c] = Vs[r][c]
        
        # Fill W: rows are (k, i) for k in 0..nB-1, i in 0..nA-1
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][offset + c] = Ws[r][c]
        
        offset += len(Us[0])
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U[0])):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    deadline = time.time() + 110  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    try:
        # Build the 4x4x4 core using 2-level Strassen (rank 49)
        U_core, V_core, W_core = _build_4x4x4_strassen()
        
        # Build the 4x1x4 cross term (rank 16)
        U_cross1, V_cross1, W_cross1 = _build_4x1x4_outer()
        
        # Build the 4x4x2 cross term (rank 32)
        U_cross2, V_cross2, W_cross2 = _build_4x4x2_strassen()
        
        # Build the 4x1x2 cross term (rank 8)
        U_cross3, V_cross3, W_cross3 = _build_4x1x2_outer()
        
        # Combine all parts
        # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
        # Part 2: A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4]
        # Part 3: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6]
        # Part 4: A[0:4, 4:5] * B[4:5, 4:6] -> C[0:4, 4:6]
        
        parts = [
            (U_core, V_core, W_core, (0, 0, 4, 4, 0, 4)),      # 4x4 * 4x4 -> 4x4, rank 49
            (U_cross1, V_cross1, W_cross1, (0, 4, 4, 1, 0, 4)), # 4x1 * 1x4 -> 4x4, rank 16
            (U_cross2, V_cross2, W_cross2, (0, 0, 4, 4, 4, 2)), # 4x4 * 4x2 -> 4x2, rank 32
            (U_cross3, V_cross3, W_cross3, (0, 4, 4, 1, 4, 2)), # 4x1 * 1x2 -> 4x2, rank 8
        ]
        
        U, V, W = _combine_parts(parts, n, m, p)
        
        # Verify
        if _verify(U, V, W, n, m, p):
            total_rank = len(U[0])
            if total_rank < best_rank:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
        
        if time.time() > deadline:
            return best_result
        
    except Exception as e:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
