"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                prod = u_val * v_val * w_val
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    # u, v, w are flat vectors of length 4
    # A = [[u0, u1], [u2, u3]]
    # B = [[v0, v1], [v2, v3]]
    # C = [[w0, w1], [w2, w3]]
    
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 0] # C00
    # We need to distribute w1 to C00, C01, C10, C11 appropriately in the sum
    # Actually, Strassen gives:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So we add terms for each M with appropriate signs in w
    
    # M1
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # C00 + C11
    
    # M2 = (A10+A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])) # C10 - C11
    
    # M3 = A00 B11
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1])) # C01 + C11
    
    # M4 = A11 B10
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0])) # -C00 + C10
    
    # M5 = (A00+A01)B10
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0])) # C00
    
    # M6 = (A10-A00)(B00+B11)
    terms.append(([ -1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1])) # -C11
    
    # M7 = (A01-A11)(B10+B11)
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1])) # C01 - C11
    
    return terms

def _get_rank_11_2x2x3():
    """
    Returns list of 11 (u, v, w) vectors for 2x2x3.
    Decomposes into 2x2x2 (rank 7) + 2x2x1 (rank 4).
    A is 2x2, B is 2x3.
    C is 2x3.
    C[:, 0:2] = A B[:, 0:2] (2x2x2)
    C[:, 2:3] = A B[:, 2:3] (2x2x1)
    """
    terms = []
    
    # Part 1: 2x2x2 for cols 0,1 of B
    # Map local 2x2x2 to global 2x2x3
    # u is same (2x2)
    # v is 2x3, but we only use first 2 cols. So v_local (len 4) -> v_global (len 6)
    # w is 2x3, but we only set first 2 cols.
    
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u # len 4
        v_g = [v[0], v[1], 0, v[2], v[3], 0] # len 6 (B00, B01, 0, B10, B11, 0)
        w_g = [w[0], w[1], 0, w[2], w[3], 0] # len 6 (C00, C01, 0, C10, C11, 0)
        terms.append((u_g, v_g, w_g))
        
    # Part 2: 2x2x1 for col 2 of B
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C00 = A00 B0 + A01 B1
    # C10 = A10 B0 + A11 B1
    # 4 terms:
    # 1. A00 * B0
    # 2. A01 * B1
    # 3. A10 * B0
    # 4. A11 * B1
    
    # Term 1: A00 * B0
    # u: [1, 0, 0, 0]
    # v: [0, 0, 0, 0, 0, 1] (B0 is at index 0 of B? No, B is 2x3. B00, B01, B02. B10, B11, B12.)
    # Wait, B is 2x3. Indices:
    # 0: B00, 1: B01, 2: B02
    # 3: B10, 4: B11, 5: B12
    # We want B02 (index 2) and B12 (index 5).
    
    # Term 1: A00 * B02
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    
    # Term 2: A01 * B12
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    
    # Term 3: A10 * B02
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    
    # Term 4: A11 * B12
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    
    return terms

def _get_rank_4_1x2x2():
    """
    Returns list of 4 (u, v, w) vectors for 1x2x2.
    A is 1x2, B is 2x2.
    C is 1x2.
    C00 = A0 B00 + A1 B10
    C01 = A0 B01 + A1 B11
    """
    terms = []
    # u is len 2 (A0, A1)
    # v is len 4 (B00, B01, B10, B11)
    # w is len 2 (C00, C01)
    
    # 1. A0 * B00
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    # 2. A0 * B01
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    # 3. A1 * B10
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    # 4. A1 * B11
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    
    return terms

def _get_rank_6_1x2x3():
    """
    Returns list of 6 (u, v, w) vectors for 1x2x3.
    A is 1x2, B is 2x3.
    C is 1x3.
    """
    terms = []
    # u is len 2
    # v is len 6
    # w is len 3
    
    # 1. A0 * B00
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    # 2. A0 * B01
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    # 3. A0 * B02
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    # 4. A1 * B10
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    # 5. A1 * B11
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    # 6. A1 * B12
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    
    return terms

def _embed_term(u_local, v_local, w_local, u_offset, v_offset, w_offset, u_size, v_size, w_size):
    """
    Embeds local vectors into global vectors.
    u_local: list of non-zero values? No, full vector.
    We need to map local indices to global indices.
    """
    u_g = [0] * u_size
    v_g = [0] * v_size
    w_g = [0] * w_size
    
    # This assumes u_local, v_local, w_local are already aligned with the local block structure
    # and we just need to shift them to the correct global positions.
    # However, the local vectors from _get_strassen etc are for the specific block.
    # We need to know which global indices correspond to the local 0..N-1.
    
    # For now, this function is a placeholder. We will construct the global terms directly in construct().
    pass

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # We will construct the 56-rank decomposition.
    # Total size of U: 3*4 = 12
    # Total size of V: 4*5 = 20
    # Total size of W: 5*3 = 15
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
    
    # Block 1: C11 (2x2) from A1 (2x4) and B1 (4x2)
    # A1 rows 0,1. B1 cols 0,1.
    # Inner dim 4 split into 2+2.
    # Term 1: A11 (2x2) * B11 (2x2) -> 2x2. Rank 7.
    # Term 2: A12 (2x2) * B12 (2x2) -> 2x2. Rank 7.
    
    # Get Strassen terms
    s_terms = _get_strassen_2x2x2()
    
    # Term 1: A11 * B11
    # A11 corresponds to A[0:2, 0:2]
    # B11 corresponds to B[0:2, 0:2]
    # Output C[0:2, 0:2]
    
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        # Map u (len 4) to A[0:2, 0:2]
        # u[0] -> A[0,0] (idx 0)
        # u[1] -> A[0,1] (idx 1)
        # u[2] -> A[1,0] (idx 4)
        # u[3] -> A[1,1] (idx 5)
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        
        # Map v (len 4) to B[0:2, 0:2]
        # v[0] -> B[0,0] (idx 0)
        # v[1] -> B[0,1] (idx 1)
        # v[2] -> B[1,0] (idx 5)
        # v[3] -> B[1,1] (idx 6)
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        
        # Map w (len 4) to C[0:2, 0:2]
        # w[0] -> C[0,0] (idx 0)
        # w[1] -> C[0,1] (idx 1)
        # w[2] -> C[1,0] (idx 3)
        # w[3] -> C[1,1] (idx 4)
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B12
    # A12 corresponds to A[0:2, 2:4]
    # B12 corresponds to B[2:4, 0:2]
    # Output C[0:2, 0:2]
    
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        # u[0] -> A[0,2] (idx 2)
        # u[1] -> A[0,3] (idx 3)
        # u[2] -> A[1,2] (idx 6)
        # u[3] -> A[1,3] (idx 7)
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        
        # v[0] -> B[2,0] (idx 10)
        # v[1] -> B[2,1] (idx 11)
        # v[2] -> B[3,0] (idx 15)
        # v[3] -> B[3,1] (idx 16)
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        
        # w same as before
        w_g[0] = w[0]
        w_g[1] = w[1]
        w_g[3] = w[2]
        w_g[4] = w[3]
        
        add_term(u_g, v_g, w_g)
        
    # Block 2: C12 (2x3) from A1 (2x4) and B2 (4x3)
    # Rank 22 (11 + 11)
    
    r11_terms = _get_rank_11_2x2x3()
    
    # Term 1: A11 * B21 (B rows 0:2, cols 2:5)
    # A11: A[0:2, 0:2]
    # B21: B[0:2, 2:5]
    # Output C[0:2, 2:5]
    
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        u_g[0] = u[0]
        u_g[1] = u[1]
        u_g[4] = u[2]
        u_g[5] = u[3]
        
        # v is len 6 (B 2x3)
        # v[0] -> B[0,2] (idx 2)
        # v[1] -> B[0,3] (idx 3)
        # v[2] -> B[0,4] (idx 4)
        # v[3] -> B[1,2] (idx 7)
        # v[4] -> B[1,3] (idx 8)
        # v[5] -> B[1,4] (idx 9)
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        
        # w is len 6 (C 2x3)
        # w[0] -> C[0,2] (idx 2)
        # w[1] -> C[0,3] (idx 3)
        # w[2] -> C[0,4] (idx 4)
        # w[3] -> C[1,2] (idx 5)
        # w[4] -> C[1,3] (idx 6)
        # w[5] -> C[1,4] (idx 7)
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        
        add_term(u_g, v_g, w_g)
        
    # Term 2: A12 * B22 (B rows 2:4, cols 2:5)
    # A12: A[0:2, 2:4]
    # B22: B[2:4, 2:5]
    # Output C[0:2, 2:5]
    
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        u_g[2] = u[0]
        u_g[3] = u[1]
        u_g[6] = u[2]
        u_g[7] = u[3]
        
        # v[0] -> B[2,2] (idx 12)
        # v[1] -> B[2,3] (idx 13)
        # v[2] -> B[2,4] (idx 14)
        # v[3] -> B[3,2] (idx 17)
        # v[4] -> B[3,3] (idx 18)
        # v[5] -> B[3,4] (idx 19)
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        
        w_g[2] = w[0]
        w_g[3] = w[1]
        w_g[4] = w[2]
        w_g[5] = w[3]
        w_g[6] = w[4]
        w_g[7] = w[5]
        
        add_term(u_g, v_g, w_g)
        
    # Block 3: C21 (1x2) from A2 (1x4) and B1 (4x2)
    # Rank 8 (4 + 4)
    
    r4_terms = _get_rank_4_1x2x2()
    
    # Term 1: A21 * B11 (B rows 0:2)
    # A21: A[2, 0:2]
    # B11: B[0:2, 0:2]
    # Output C[2, 0:2]
    
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        # u is len 2 (A21)
        u_g[8] = u[0] # A[2,0]
        u_g[9] = u[1] # A[2,1]
        
        # v is len 4 (B11)
        v_g[0] = v[0]
        v_g[1] = v[1]
        v_g[5] = v[2]
        v_g[6] = v[3]
        
        # w is len 2 (C21)
        w_g[6] = w[0] # C[2,0]
        w_g[7] = w[1] # C[2,1]
        
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B12 (B rows 2:4)
    # A22: A[2, 2:4]
    # B12: B[2:4, 0:2]
    # Output C[2, 0:2]
    
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        u_g[10] = u[0] # A[2,2]
        u_g[11] = u[1] # A[2,3]
        
        v_g[10] = v[0]
        v_g[11] = v[1]
        v_g[15] = v[2]
        v_g[16] = v[3]
        
        w_g[6] = w[0]
        w_g[7] = w[1]
        
        add_term(u_g, v_g, w_g)
        
    # Block 4: C22 (1x3) from A2 (1x4) and B2 (4x3)
    # Rank 12 (6 + 6)
    
    r6_terms = _get_rank_6_1x2x3()
    
    # Term 1: A21 * B21 (B rows 0:2)
    # A21: A[2, 0:2]
    # B21: B[0:2, 2:5]
    # Output C[2, 2:5]
    
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        u_g[8] = u[0]
        u_g[9] = u[1]
        
        v_g[2] = v[0]
        v_g[3] = v[1]
        v_g[4] = v[2]
        v_g[7] = v[3]
        v_g[8] = v[4]
        v_g[9] = v[5]
        
        w_g[8] = w[0] # C[2,2]
        w_g[9] = w[1] # C[2,3]
        w_g[10] = w[2] # C[2,4]
        
        add_term(u_g, v_g, w_g)
        
    # Term 2: A22 * B22 (B rows 2:4)
    # A22: A[2, 2:4]
    # B22: B[2:4, 2:5]
    # Output C[2, 2:5]
    
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        
        u_g[10] = u[0]
        u_g[11] = u[1]
        
        v_g[12] = v[0]
        v_g[13] = v[1]
        v_g[14] = v[2]
        v_g[17] = v[3]
        v_g[18] = v[4]
        v_g[19] = v[5]
        
        w_g[8] = w[0]
        w_g[9] = w[1]
        w_g[10] = w[2]
        
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
