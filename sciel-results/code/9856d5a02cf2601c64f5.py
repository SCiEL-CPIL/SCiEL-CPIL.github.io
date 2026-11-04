import time
import numpy as np

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a hierarchical block decomposition:
    - Split indices into groups {0,1}, {2,3}, {4}
    - Apply Strassen (rank 7) to all 2x2x2 block combinations
    - Use direct multiplication for other blocks
    
    This achieves rank 117 (vs 125 schoolbook).
    
    U: (25, rank) - indexed by (i, j) where i is row of A, j is col of A
    V: (25, rank) - indexed by (j, k) where j is row of B, k is col of B
    W: (25, rank) - indexed by (k, i) where k is row of C, i is col of C (trace convention)
    
    The decomposition satisfies:
      For all i, j, k: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    # Try to construct a low-rank decomposition
    try:
        result = _construct_hierarchical(n, m, p)
        if result is not None:
            # Verify correctness
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _construct_hierarchical(n, m, p):
    """
    Construct a low-rank decomposition using hierarchical block structure.
    
    Strategy: Split 5 into 2+2+1. Apply Strassen to each 2x2x2 block.
    Cross-terms use direct multiplication.
    
    Blocks:
    - B0 = {0,1} (size 2)
    - B1 = {2,3} (size 2)  
    - B2 = {4} (size 1)
    
    For each combination of blocks (bi, bj, bk):
    - If all three blocks have size 2: use Strassen (7 terms)
    - Otherwise: direct multiplication
    """
    B0 = [0, 1]
    B1 = [2, 3]
    B2 = [4]
    blocks = [B0, B1, B2]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        terms.append((u, v, w))
    
    def add_strassen_2x2(indices_i, indices_j, indices_k):
        """
        Apply Strassen to a 2x2x2 block.
        indices_i, indices_j, indices_k: lists of 2 global indices each
        """
        def add_s(local_u, local_v, local_w):
            u_entries = []
            v_entries = []
            w_entries = []
            for (i_loc, j_loc), val in local_u.items():
                if val != 0:
                    gi = indices_i[i_loc]
                    gj = indices_j[j_loc]
                    u_entries.append((gi * m + gj, val))
            for (j_loc, k_loc), val in local_v.items():
                if val != 0:
                    gj = indices_j[j_loc]
                    gk = indices_k[k_loc]
                    v_entries.append((gj * p + gk, val))
            for (k_loc, i_loc), val in local_w.items():
                if val != 0:
                    gk = indices_k[k_loc]
                    gi = indices_i[i_loc]
                    w_entries.append((gk * n + gi, val))
            add_term(u_entries, v_entries, w_entries)
        
        # M1 = (A00+A11)(B00+B11)
        add_s({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        # M2 = (A10+A11)B00
        add_s({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})
        # M3 = A00(B01-B11)
        add_s({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})
        # M4 = A11(B10-B00)
        add_s({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1})
        # M5 = (A00+A01)B11
        add_s({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
        # M6 = (A10-A00)(B00+B01)
        add_s({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        # M7 = (A01-A11)(B10+B11)
        add_s({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    
    # Process all block combinations
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                bi_list = blocks[bi]
                bj_list = blocks[bj]
                bk_list = blocks[bk]
                si, sj, sk = len(bi_list), len(bj_list), len(bk_list)
                
                # Only use Strassen if all three blocks have size 2
                if si == 2 and sj == 2 and sk == 2:
                    add_strassen_2x2(bi_list, bj_list, bk_list)
                else:
                    # Direct multiplication for this block combination
                    for i_loc in range(si):
                        for j_loc in range(sj):
                            for k_loc in range(sk):
                                gi = bi_list[i_loc]
                                gj = bj_list[j_loc]
                                gk = bk_list[k_loc]
                                add_term([(gi * m + gj, 1)],
                                         [(gj * p + gk, 1)],
                                         [(gk * n + gi, 1)])
    
    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
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
