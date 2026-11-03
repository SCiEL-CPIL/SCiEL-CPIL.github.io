"""Combined: block decomposition with Strassen rank-7 on 2x2x2 sub-blocks."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Strassen rank-7 for 2x2x2. Returns U,V,W each list of 7 flat 4-vectors."""
    U = [
        [1,0,0,0],
        [1,1,0,0],
        [0,0,1,1],
        [0,0,0,1],
        [1,0,0,1],
        [0,1,0,-1],
        [0,-1,1,0],
    ]
    V = [
        [1,0,0,-1],
        [0,0,0,1],
        [1,0,0,0],
        [0,-1,0,1],
        [1,0,0,1],
        [0,1,1,1],
        [1,1,0,0],
    ]
    W = [
        [0,1,1,0],
        [-1,1,0,0],
        [0,0,1,-1],
        [1,0,1,0],
        [1,0,0,1],
        [0,0,0,-1],
        [-1,0,0,0],
    ]
    return U, V, W

def _verify(n, m, p, U, V, W):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-9

def _build_decomposition():
    n, m, p = 3, 5, 6
    sU, sV, sW = _strassen_2x2()
    
    # We'll build the full decomposition by combining:
    # 1. Schoolbook for the parts not covered by 2x2x2 Strassen blocks
    # 2. Strassen for 2x2x2 sub-blocks
    #
    # Strategy: iterate over all (i,j,k) and assign terms.
    # For a 2x2x2 block at rows {i0,i0+1}, cols {j0,j0+1}, out {k0,k0+1},
    # use 7 Strassen terms instead of 8 schoolbook terms.
    #
    # We'll greedily find non-overlapping 2x2x2 blocks and apply Strassen,
    # then schoolbook for the rest.
    
    # Determine which (i,j,k) triples are covered by Strassen blocks.
    # n=3, m=5, p=6.
    # Possible 2x2x2 blocks: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # We want to cover as much as possible with non-overlapping blocks.
    #
    # Let's use a tiling approach:
    # For i=0,1 (2 rows), we can tile j and k.
    # j: 0-4 (5 cols) -> blocks at j=0,2,4 (3 blocks, covering j=0-5 but j only 0-4, so j=0,2,4 -> covers 0,1,2,3,4? j=0 block covers 0,1; j=2 covers 2,3; j=4 covers 4,5 but 5 out of range. So j=0,2 -> covers 0-3, j=4 leftover.
    # Actually for m=5: j blocks at 0 (covers 0,1), 2 (covers 2,3), and j=4 is single.
    # For p=6: k blocks at 0,2,4 (covers 0-5 fully).
    #
    # So for i in {0,1}:
    #   j in {0,1}: k in {0,1,2,3,4,5} -> 3 Strassen blocks (k=0,2,4)
    #   j in {2,3}: k in {0,1,2,3,4,5} -> 3 Strassen blocks
    #   j in {4}:    k in {0,1,2,3,4,5} -> schoolbook (1 col)
    # For i=2: all schoolbook.
    #
    # Count: 
    # i in {0,1}, j in {0,1}: 3 blocks * 7 = 21
    # i in {0,1}, j in {2,3}: 3 blocks * 7 = 21
    # i in {0,1}, j=4: 6 schoolbook = 6
    # i=2: 5*6 = 30 schoolbook
    # Total: 21+21+6+30 = 78
    
    # Let me recount. Each Strassen block covers 2x2x2 = 8 (i,j,k) triples with 7 terms.
    # Schoolbook covers 1 triple with 1 term.
    #
    # i in {0,1}, j in {0,1}, k in {0..5}: 3 Strassen blocks (k=0,2,4), each 7 terms = 21
    # i in {0,1}, j in {2,3}, k in {0..5}: 3 Strassen blocks, 7 terms each = 21
    # i in {0,1}, j=4, k in {0..5}: 6 schoolbook = 6
    # i=2, j in {0..4}, k in {0..5}: 30 schoolbook = 30
    # Total = 21+21+6+30 = 78
    
    # Build the decomposition
    U = {}  # (i*m+j) -> list of (rank_idx, value)
    V = {}  # (j*p+k) -> list of (rank_idx, value)
    W = {}  # (k*n+i) -> list of (rank_idx, value)
    
    rank = 0
    def add_term(Uidx, Vidx, Widx, uval, vval, wval):
        nonlocal rank
        U.setdefault(Uidx, []).append((rank, uval))
        V.setdefault(Vidx, []).append((rank, vval))
        W.setdefault(Widx, []).append((rank, wval))
        rank += 1
    
    def add_strassen_block(i0, j0, k0):
        """Add a 2x2x2 Strassen block at top-left (i0,j0,k0)."""
        for t in range(7):
            # U: 2x2 block at rows i0,i0+1, cols j0,j0+1
            for di in range(2):
                for dj in range(2):
                    i = i0 + di
                    j = j0 + dj
                    Uidx = i * m + j
                    add_term(Uidx, (j0 + dj) * p + (k0 + 0), 0, 0, 0, 0)  # placeholder, we'll do properly
            # Actually, let's directly map the 4 entries of each 2x2 matrix
            # U[t] is a 2x2 matrix [a,b;c,d] -> U[i0*m+j0]=a, U[i0*m+j0+1]=b, U[(i0+1)*m+j0]=c, U[(i0+1)*m+j0+1]=d
            u_mat = sU[t]  # [a,b,c,d]
            v_mat = sV[t]
            w_mat = sW[t]
            
            Uidx00 = i0 * m + j0
            Uidx01 = i0 * m + (j0+1)
            Uidx10 = (i0+1) * m + j0
            Uidx11 = (i0+1) * m + (j0+1)
            
            Vidx00 = j0 * p + k0
            Vidx01 = j0 * p + (k0+1)
            Vidx10 = (j0+1) * p + k0
            Vidx11 = (j0+1) * p + (k0+1)
            
            Widx00 = k0 * n + i0
            Widx01 = (k0+1) * n + i0
            Widx10 = k0 * n + (i0+1)
            Widx11 = (k0+1) * n + (i0+1)
            
            # W is indexed by [k*n + i], so W[k0*n+i0] corresponds to output (i0, k0)
            # The 2x2 W matrix [a,b;c,d] means:
            # W[i0,k0]=a, W[i0,k0+1]=b, W[i0+1,k0]=c, W[i0+1,k0+1]=d
            # In our indexing: W[k0*n+i0]=a, W[(k0+1)*n+i0]=b, W[k0*n+(i0+1)]=c, W[(k0+1)*n+(i0+1)]=d
            
            a, b, c, d = w_mat
            # Note: w_mat is [W[i0,k0], W[i0,k0+1], W[i0+1,k0], W[i0+1,k0+1]]
            # = [W[k0*n+i0], W[(k0+1)*n+i0], W[k0*n+i0+1], W[(k0+1)*n+i0+1]]
            
            # Add term t
            r = rank
            rank += 1
            U.setdefault(Uidx00, []).append((r, u_mat[0]))
            U.setdefault(Uidx01, []).append((r, u_mat[1]))
            U.setdefault(Uidx10, []).append((r, u_mat[2]))
            U.setdefault(Uidx11, []).append((r, u_mat[3]))
            V.setdefault(Vidx00, []).append((r, v_mat[0]))
            V.setdefault(Vidx01, []).append((r, v_mat[1]))
            V.setdefault(Vidx10, []).append((r, v_mat[2]))
            V.setdefault(Vidx11, []).append((r, v_mat[3]))
            W.setdefault(Widx00, []).append((r, a))
            W.setdefault(Widx01, []).append((r, b))
            W.setdefault(Widx10, []).append((r, c))
            W.setdefault(Widx11, []).append((r, d))
    
    def add_schoolbook(i, j, k):
        r = rank
        rank += 1
        U.setdefault(i * m + j, []).append((r, 1))
        V.setdefault(j * p + k, []).append((r, 1))
        W.setdefault(k * n + i, []).append((r, 1))
    
    # Apply Strassen blocks
    # i in {0,1}, j in {0,2}, k in {0,2,4}
    for i0 in [0, 1]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    add_strassen_block(i0, j0, k0)
    
    # Schoolbook for remaining
    # Track which (i,j,k) are covered
    covered = set()
    for i0 in [0, 1]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                if i0+1 < n and j0+1 < m and k0+1 < p:
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    # Convert to dense matrices
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for idx, terms in U.items():
        for r, val in terms:
            U_mat[idx][r] = val
    for idx, terms in V.items():
        for r, val in terms:
            V_mat[idx][r] = val
    for idx, terms in W.items():
        for r, val in terms:
            W_mat[idx][r] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    try:
        result, rank = _build_decomposition()
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            if time.time() < deadline:
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
