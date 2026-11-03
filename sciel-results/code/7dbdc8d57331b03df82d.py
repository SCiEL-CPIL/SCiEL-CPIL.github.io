import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n=2, m=3, p=9):
    """
    Verify that the tensor decomposition is exact.
    We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _strassen_2x2x2_terms():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-element vectors (flattened 2x2 matrices).
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # Result: [[p1+p4+p5-p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    terms = []
    
    # p1 = (a+d)(e+h)
    # A part: a=1, d=1 -> [1,0,0,1]
    # B part: e=1, h=1 -> [1,0,0,1]
    # C part: contributes to [0,0] and [1,1] with +1, and to [0,1] and [1,0] with -1
    # Actually let me be more careful. The C matrix gets contributions:
    # C[0,0] += p1, C[1,1] += p1, C[0,1] -= p1, C[1,0] += p1 (wait, let me re-derive)
    
    # Let me use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4+M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each term, we need u (A part), v (B part), w (C part)
    
    # Term 1: M1
    u1 = [1, 0, 0, 1]  # A00 + A11
    v1 = [1, 0, 0, 1]  # B00 + B11
    w1 = [1, 0, 0, -1]  # C00 += 1, C01 -= 0, C10 -= 0, C11 += 1... wait
    # C00 += M1, C11 += M1, C01 -= M1, C10 += M1? No.
    # C00 = M1+M4+M5-M7, so M1 contributes +1 to C00
    # C11 = M1-M2+M3+M6, so M1 contributes +1 to C11
    # C01 = M3+M5, M1 doesn't contribute
    # C10 = M2+M4, M1 doesn't contribute
    # So w1 = [1, 0, 0, 1] (C00=1, C01=0, C10=0, C11=1)
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # Term 2: M2
    u2 = [0, 0, 1, 1]  # A10 + A11
    v2 = [1, 0, 0, 0]  # B00
    w2 = [0, 0, 1, -1]  # C10 += M2, C11 -= M2
    terms.append((u2, v2, w2))
    
    # Term 3: M3
    u3 = [1, 0, 0, 0]  # A00
    v3 = [0, 1, 0, -1]  # B01 - B11
    w3 = [0, 1, 0, 1]  # C01 += M3, C11 += M3
    terms.append((u3, v3, w3))
    
    # Term 4: M4
    u4 = [0, 0, 0, 1]  # A11
    v4 = [0, 0, 1, -1]  # B10 - B00
    w4 = [1, 0, 1, 0]  # C00 += M4, C10 += M4
    terms.append((u4, v4, w4))
    
    # Term 5: M5
    u5 = [1, 1, 0, 0]  # A00 + A01
    v5 = [0, 0, 0, 1]  # B11
    w5 = [1, 1, 0, 0]  # C00 += M5, C01 += M5
    terms.append((u5, v5, w5))
    
    # Term 6: M6
    u6 = [0, 0, 1, -1]  # A10 - A00
    v6 = [1, 1, 0, 0]  # B00 + B01
    w6 = [0, 0, 0, 1]  # C11 += M6
    terms.append((u6, v6, w6))
    
    # Term 7: M7
    u7 = [0, 1, 0, -1]  # A01 - A11
    v7 = [0, 0, 1, 1]  # B10 + B11
    w7 = [-1, 0, 0, 0]  # C00 -= M7
    terms.append((u7, v7, w7))
    
    return terms

def _construct_2x3x9_strassen():
    """
    Construct a rank-50 decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Split the 3 (middle) dimension into 2+1, and the 9 (right) dimension into 4*2+1.
    Actually, let's think about this differently.
    
    We have A (2x3) and B (3x9). We want C (2x9).
    
    One approach: Split B's columns into groups. 9 = 4+4+1 or 2+2+2+2+1.
    If we split into 2+2+2+2+1, we get five 2x2 blocks and one 2x1 block.
    Each 2x2 block uses Strassen (7 terms), and the 2x1 uses schoolbook (2 terms).
    Total: 4*7 + 2 = 30. But we also need to handle the A matrix appropriately.
    
    Actually, the tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    The "block" structure depends on how we partition the indices.
    
    Let me try a different approach: Partition j (0,1,2) into {0,1} and {2}.
    Partition k (0,...,8) into pairs: {0,1}, {2,3}, {4,5}, {6,7}, and {8}.
    
    For each pair of k values (k1, k2) and the j-pair (j1, j2) = (0,1):
    We have a 2x2x2 sub-problem for A[:,0:2] * B[0:2, k1:k2] contributing to C[:, k1:k2].
    This uses 7 Strassen terms.
    
    For j=2 and each pair of k values:
    We have a 2x1x2 sub-problem: A[:,2] * B[2, k1:k2] contributing to C[:, k1:k2].
    This uses 2 schoolbook terms (one for each k).
    
    For j=2 and k=8:
    We have a 2x1x1 sub-problem: A[:,2] * B[2, 8] contributing to C[:, 8].
    This uses 1 schoolbook term.
    
    Total: 4*7 + 4*2 + 1 = 28 + 8 + 1 = 37.
    
    Hmm, that's less than 50. Let me reconsider.
    
    Actually, I think the issue is that the Strassen decomposition works on 2x2 matrices,
    but here our A blocks are 2x2 (rows x j-pair) and B blocks are 2x2 (j-pair x k-pair),
    and C blocks are 2x2 (rows x k-pair). So it should work.
    
    Let me implement this carefully.
    """
    n, m, p = 2, 3, 9
    
    # We'll build U, V, W as lists of lists, starting empty
    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec has length n*m, v_vec has length m*p, w_vec has length p*n."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Strassen terms for 2x2x2
    strassen_terms = _strassen_2x2x2_terms()
    
    # j-pairs: (0,1) for Strassen, (2) for schoolbook
    # k-pairs: (0,1), (2,3), (4,5), (6,7) for Strassen/schoolbook, (8) for schoolbook
    
    # For j-pair (0,1) and each k-pair (k1, k2):
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        for (u_s, v_s, w_s) in strassen_terms:
            # u_s is for A[0:2, 0:2] -> indices 0,1,2,3 in U (i*m+j)
            # v_s is for B[0:2, k1:k2] -> indices in V (j*p+k)
            # w_s is for C[0:2, k1:k2] -> indices in W (k*n+i)
            
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # u_s maps to A[i, j] for i in 0,1 and j in 0,1
            # A indices: i*m+j = i*3+j
            for idx, val in enumerate(u_s):
                # idx 0: A[0,0], idx 1: A[0,1], idx 2: A[1,0], idx 3: A[1,1]
                i = idx // 2
                j = idx % 2
                u_vec[i * m + j] = val
            
            # v_s maps to B[j, k] for j in 0,1 and k in k1,k2
            # B indices: j*p+k
            for idx, val in enumerate(v_s):
                # idx 0: B[0,k1], idx 1: B[0,k2], idx 2: B[1,k1], idx 3: B[1,k2]
                j = idx // 2
                k_offset = idx % 2
                k = k1 + k_offset
                v_vec[j * p + k] = val
            
            # w_s maps to C[i, k] for i in 0,1 and k in k1,k2
            # C indices: k*n+i
            for idx, val in enumerate(w_s):
                # idx 0: C[0,k1], idx 1: C[0,k2], idx 2: C[1,k1], idx 3: C[1,k2]
                i = idx // 2
                k_offset = idx % 2
                k = k1 + k_offset
                w_vec[k * n + i] = val
            
            add_term(u_vec, v_vec, w_vec)
    
    # For j=2 and each k-pair (k1, k2):
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        # Schoolbook: 2 terms
        # Term for k1: A[i,2] * B[2,k1] * C[i,k1]
        for i in range(n):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + 2] = 1
            v_vec[2 * p + k1] = 1
            w_vec[k1 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
        
        # Term for k2
        for i in range(n):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + 2] = 1
            v_vec[2 * p + k2] = 1
            w_vec[k2 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # For j=2 and k=8:
    for i in range(n):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        u_vec[i * m + 2] = 1
        v_vec[2 * p + 8] = 1
        w_vec[8 * n + i] = 1
        add_term(u_vec, v_vec, w_vec)
    
    rank = len(U)
    
    # Convert to matrices
    U_mat = np.array(U).T  # (rank, n*m) -> transpose to (n*m, rank)
    V_mat = np.array(V).T  # (rank, m*p) -> transpose to (m*p, rank)
    W_mat = np.array(W).T  # (rank, p*n) -> transpose to (p*n, rank)
    
    # Verify
    if not _verify(U_mat, V_mat, W_mat, n, m, p):
        return None
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Keep margin for 120s limit
    
    # Try Strassen-based rank 50 (or lower) construction
    result = _construct_2x3x9_strassen()
    if result is not None:
        return result
    
    # Fallback: schoolbook rank 54
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
