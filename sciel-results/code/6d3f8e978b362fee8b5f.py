"""Matrix multiplication 3x5x5: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # Check: for all i in range(n), j in range(m), k in range(p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a simplified check for the diagonal of the tensor.
    # The full tensor check requires checking off-diagonals too, 
    # but if the construction is correct, this is sufficient for the rank minimality task 
    # assuming the structure is preserved.
    # Actually, we must check the full tensor identity:
    # sum_c U[i*m+j, c] V[j'*p+k', c] W[k'*n+i', c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # We will do a sparse check or full check if rank is small enough.
    # For rank ~70, full check is 75 * 75 * 75 * 70 ~ 30M ops, which is fast.
    
    # Flatten indices
    # U index: i*m + j
    # V index: j'*p + k'
    # W index: k'*n + i'
    
    # We check for a sample of (i,j,k,i',j',k') or all if fast.
    # Let's check all (i,j,k) for the diagonal (i=i', j=j', k=k')
    # and a few off-diagonals.
    # Actually, let's just do the full verification to be safe.
    # n=3, m=5, p=5.
    # U dim: 15. V dim: 25. W dim: 15.
    # r ~ 70.
    # Loop i,j,k,i',j',k': 3*5*5 * 3*5*5 = 75*75 = 5625 cases.
    # Inner sum r ~ 70.
    # Total ops: 5625 * 70 ~ 400k. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1 if (i==ip and j==jp and k==kp) else 0
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[jp*p+kp, c] * W[kp*n+ip, c]
                            if s != expected:
                                return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen products M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: W index is k*2+i.
    # C(0,0) is i=0,k=0 -> W[0]
    # C(0,1) is i=0,k=1 -> W[1]
    # C(1,0) is i=1,k=0 -> W[2]
    # C(1,1) is i=1,k=1 -> W[3]
    
    # M1
    # A: (0,0)=1, (1,1)=1
    # B: (0,0)=1, (1,1)=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2
    # A: (1,0)=1, (1,1)=1
    # B: (0,0)=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21
    
    # M3
    # A: (0,0)=1
    # B: (1,0)=1, (1,1)=-1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4
    # A: (1,1)=1
    # B: (0,1)=1, (0,0)=-1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11
    W[2][3] = 1  # C21
    
    # M5
    # A: (0,0)=1, (0,1)=1
    # B: (1,1)=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C11
    W[1][4] = 1  # C12
    
    # M6
    # A: (1,0)=1, (0,0)=-1
    # B: (0,0)=1, (0,1)=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22
    
    # M7
    # A: (0,1)=1, (1,1)=-1
    # B: (1,0)=1, (1,1)=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11
    # Wait, C22 also has M6, but M7 is only in C11?
    # Formula: C22 = M1 - M2 + M3 + M6.
    # C11 = M1 + M4 - M5 + M7.
    # So M7 is only in C11.
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank_school = n * m * p  # 75
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to improve using Strassen 2x2x2 blocks
    # Identify 4 disjoint 2x2x2 blocks
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Blocks:
    # 1. i:0-1, j:0-1, k:0-1
    # 2. i:0-1, j:2-3, k:0-1
    # 3. i:0-1, j:0-1, k:2-3
    # 4. i:0-1, j:2-3, k:2-3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # For each block, we replace 8 schoolbook terms with 7 Strassen terms.
    # We need to build new U, V, W with rank 75 - 4 = 71.
    
    new_rank = 71
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    # We will fill the columns.
    # Columns 0..69 correspond to the 71 terms.
    # We need to map the 71 terms to the indices.
    
    # Strategy:
    # Iterate through all 75 (i,j,k) tuples.
    # If (i,j,k) is in a Strassen block, skip it for schoolbook.
    # Instead, add the Strassen terms for that block.
    
    # This is tricky because Strassen terms span multiple (i,j,k).
    # Better: 
    # 1. Add all schoolbook terms that are NOT in any block.
    # 2. For each block, add the 7 Strassen terms.
    
    # Identify which (i,j,k) are in blocks.
    in_block = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    in_block.add((i0+di, j0+dj, k0+dk))
    
    col_idx = 0
    # 1. Add schoolbook terms for non-block indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in in_block:
                    new_U[i*m+j][col_idx] = 1
                    new_V[j*p+k][col_idx] = 1
                    new_W[k*n+i][col_idx] = 1
                    col_idx += 1
    
    # 2. Add Strassen terms for each block
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    for (i0, j0, k0) in blocks:
        for c in range(7):
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    u_val = strassen_U[li*2+lj][c]
                    if u_val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        new_U[gi*m+gj][col_idx] += u_val
            
            for lj in range(2):
                for lk in range(2):
                    v_val = strassen_V[lj*2+lk][c]
                    if v_val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        new_V[gj*p+gk][col_idx] += v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = strassen_W[lk*2+li][c]
                    if w_val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        new_W[gk*n+gi][col_idx] += w_val
            
            col_idx += 1
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        best = {"U": new_U, "V": new_V, "W": new_W}
        best_rank = new_rank
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return best
    
    return best
# EVOLVE-BLOCK-END
