import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank below schoolbook.
    Returns {'U': U, 'V': V, 'W': W} where U is (9, rank), V is (24, rank), W is (24, rank).
    """
    n, m, p = 3, 3, 8
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    def get_strassen_2x2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)] # 2*2
        V = [[0]*rank for _ in range(4)] # 2*2
        W = [[0]*rank for _ in range(4)] # 2*2
        
        # P1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # P3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # P4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # P5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # P6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # P7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """
        Verifies if the decomposition correctly represents matrix multiplication.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for r in range(rank):
                        val += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that non-multiplication terms are 0? 
        # The tensor T_{a,b,c} is non-zero only if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
        # If we construct it correctly, other terms should be 0.
        # For safety, we can check a few random non-matching indices if time permits, 
        # but the main constraint is the matching ones being 1.
        # Given the structure of Strassen+Schoolbook, cross-talk is unlikely but possible.
        # Let's do a full check if rank is small enough or if we have time.
        # Actually, the evaluator checks the exact tensor. 
        # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1, else 0.
        # We must ensure sum_r U[a,r]V[b,r]W[c,r] == 0 for all (a,b,c) not of the form above.
        
        # Efficient check: 
        # We can't iterate all (9*24*24) easily in pure python if rank is high? 
        # 9*24*24 = 5184 entries. Rank ~ 68. 
        # 5184 * 68 * 3 ops ~ 1 million ops. Fast enough in Python? 
        # 1M ops is ~0.1s. So yes.
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a valid multiplication index
                    # a = i*m + j => i = a//m, j = a%m
                    # b = j*p + k => j2 = b//p, k = b%p
                    # c = k*n + i => k2 = c//n, i2 = c%n
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    is_valid = (i == i2) and (j == j2) and (k == k2)
                    target = 1.0 if is_valid else 0.0
                    
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    
                    if abs(val - target) > tol:
                        return False
        return True

    # Strategy:
    # 1. Identify disjoint 2x2x2 blocks in the 3x3x8 tensor.
    # 2. Apply Strassen (rank 7) to each block.
    # 3. Apply Schoolbook (rank 1 per entry) to the rest.
    # 
    # Dimensions: i in [0,2], j in [0,2], k in [0,7].
    # A 2x2x2 block covers i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1].
    # We need to pack as many disjoint 2x2x2 blocks as possible.
    # i has size 3, so we can only fit one block in i-dim (0-1 or 1-2).
    # j has size 3, so we can only fit one block in j-dim (0-1 or 1-2).
    # k has size 8, so we can fit 4 blocks in k-dim (0-1, 2-3, 4-5, 6-7).
    #
    # So max blocks = 1 (i) * 1 (j) * 4 (k) = 4 blocks.
    # Let's use i0=0, j0=0 for all 4 blocks.
    # Blocks at k=0, k=2, k=4, k=6.
    # Remaining entries:
    # i=2 (entire row) is not covered by i0=0 blocks.
    # j=2 (entire col) is not covered by j0=0 blocks.
    # So all entries where i=2 OR j=2 are covered by schoolbook.
    # Entries where i in {0,1} AND j in {0,1} are covered by Strassen blocks.
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    
    # Calculate expected rank
    # 4 Strassen blocks * 7 = 28
    # Total entries = 3*3*8 = 72
    # Covered entries = 4 * 8 = 32
    # Remaining entries = 72 - 32 = 40
    # Total rank = 28 + 40 = 68
    
    U, V, W = _build_decomposition(n, m, p, blocks)
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n * m * p
        U_s = [[0]*rank for _ in range(n*m)]
        V_s = [[0]*rank for _ in range(m*p)]
        W_s = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

def _build_decomposition(n, m, p, blocks):
    """
    Builds the decomposition using Strassen for specified blocks and schoolbook for the rest.
    """
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(blocks) + n_school
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    sU, sV, sW = get_strassen_2x2x2()
    
    for (i0, j0, k0) in blocks:
        for c in range(7):
            # Map Strassen indices to global indices
            # U local index: di*2 + dj. Global: (i0+di)*m + (j0+dj)
            # V local index: dj*2 + dk. Global: (j0+dj)*p + (k0+dk)
            # W local index: dk*2 + di. Global: (k0+dk)*n + (i0+di)
            
            for di in range(2):
                for dj in range(2):
                    u_idx_local = di*2 + dj
                    u_idx_global = (i0+di)*m + (j0+dj)
                    U[u_idx_global][col+c] = sU[u_idx_local][c]
            
            for dj in range(2):
                for dk in range(2):
                    v_idx_local = dj*2 + dk
                    v_idx_global = (j0+dj)*p + (k0+dk)
                    V[v_idx_global][col+c] = sV[v_idx_local][c]
            
            for dk in range(2):
                for di in range(2):
                    w_idx_local = dk*2 + di
                    w_idx_global = (k0+dk)*n + (i0+di)
                    W[w_idx_global][col+c] = sW[w_idx_local][c]
        col += 7
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
                    
    return U, V, W

# EVOLVE-BLOCK-END
