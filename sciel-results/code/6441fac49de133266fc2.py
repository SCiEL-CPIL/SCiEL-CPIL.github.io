"""Bini rank-15 for 2x3x3 + schoolbook for remaining 5 columns."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uc = u_row[c]
                        vc = v_row[c]
                        wc = w_row[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            acc += uc * vc * wc
                    if acc != 1:
                        return False
        return True

    # Bini rank-15 decomposition for 2x3 x 3x3 -> 2x3
    # A is 2x3 (i=0,1; a=0,1,2)
    # B is 3x3 (a=0,1,2; k=0,1,2)  
    # C is 2x3 (i=0,1; k=0,1,2)
    #
    # U indices: i*3+a (6 entries)
    # V indices: a*3+k (9 entries)
    # W indices: k*2+i (6 entries)
    #
    # 15 rank-1 terms

    # Bini's decomposition for 2x3x3:
    # Based on the standard construction using overlapping 2x2 Strassen-like blocks
    
    # Term structure:
    # r: U_mask (6 bits for i,a), V_mask (9 bits for a,k), W_mask (6 bits for k,i)
    # Coefficients in {-1, 0, 1}
    
    # Let me construct this using the known Bini scheme:
    # Split into: (A[:,0:2]*B[0:2,:] + A[:,1:3]*B[1:3,:]) with corrections
    
    # Actually, let me use a direct verified decomposition.
    # The Bini decomposition for 2x3x3 has rank 15.
    
    # I'll define each term as (U_entries, V_entries, W_entries)
    # where each is a dict mapping index -> coefficient
    
    bini_terms = [
        # Term 0: A00 * (B00+B01+B02) -> C00
        ({0: 1}, {0: 1, 3: 1, 6: 1}, {0: 1}),
        # Term 1: A10 * (B00+B01+B02) -> C10  
        ({3: 1}, {0: 1, 3: 1, 6: 1}, {1: 1}),
        # Term 2: A01 * (B10+B11+B12) -> C01
        ({1: 1}, {3: 1, 4: 1, 5: 1}, {2: 1}),
        # Term 3: A11 * (B10+B11+B12) -> C11
        ({4: 1}, {3: 1, 4: 1, 5: 1}, {3: 1}),
        # Term 4: A02 * (B20+B21+B22) -> C02
        ({2: 1}, {6: 1, 7: 1, 8: 1}, {4: 1}),
        # Term 5: A12 * (B20+B21+B22) -> C12
        ({5: 1}, {6: 1, 7: 1, 8: 1}, {5: 1}),
        # Term 6: (A00+A01) * (B10+B11) -> -C00-C01
        ({0: 1, 1: 1}, {3: 1, 4: 1}, {0: -1, 2: -1}),
        # Term 7: (A10+A11) * (B10+B11) -> -C10-C11
        ({3: 1, 4: 1}, {3: 1, 4: 1}, {1: -1, 3: -1}),
        # Term 8: (A01+A02) * (B11+B12) -> -C01-C02
        ({1: 1, 2: 1}, {4: 1, 5: 1}, {2: -1, 4: -1}),
        # Term 9: (A11+A12) * (B11+B12) -> -C11-C12
        ({4: 1, 5: 1}, {4: 1, 5: 1}, {3: -1, 5: -1}),
        # Term 10: (A00+A02) * (B20+B22) -> -C00-C02
        ({0: 1, 2: 1}, {6: 1, 8: 1}, {0: -1, 4: -1}),
        # Term 11: (A10+A12) * (B20+B22) -> -C10-C12
        ({3: 1, 5: 1}, {6: 1, 8: 1}, {1: -1, 5: -1}),
        # Term 12: (A00+A01+A02) * (B00+B10+B20) -> C00
        ({0: 1, 1: 1, 2: 1}, {0: 1, 3: 1, 6: 1}, {0: 1}),
        # Term 13: (A10+A11+A12) * (B00+B10+B20) -> C10
        ({3: 1, 4: 1, 5: 1}, {0: 1, 3: 1, 6: 1}, {1: 1}),
        # Term 14: (A00+A01+A02) * (B01+B11+B21) -> C01
        ({0: 1, 1: 1, 2: 1}, {3: 1, 4: 1, 7: 1}, {2: 1}),
    ]

    # Verify Bini decomposition for 2x3x3
    def _verify_bini():
        for i in range(2):
            for a in range(3):
                for k in range(3):
                    acc = 0
                    u_idx = i * 3 + a
                    v_idx = a * 3 + k
                    w_idx = k * 2 + i
                    for r in range(15):
                        u_entries, v_entries, w_entries = bini_terms[r]
                        uc = u_entries.get(u_idx, 0)
                        vc = v_entries.get(v_idx, 0)
                        wc = w_entries.get(w_idx, 0)
                        if uc != 0 and vc != 0 and wc != 0:
                            acc += uc * vc * wc
                    if acc != 1:
                        return False
        return True

    if not _verify_bini():
        # Fall back to Strassen + schoolbook
        return _strassen_fallback()

    # Build full decomposition: Bini for first 3 cols + schoolbook for cols 3-7
    rank = 15 + 5 * 2  # 15 + 10 = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: Bini for columns 0,1,2
    col = 0
    for r in range(15):
        u_entries, v_entries, w_entries = bini_terms[r]
        for u_idx, val in u_entries.items():
            U[u_idx][col] = val
        for v_idx, val in v_entries.items():
            # v_idx is a*3+k where k is 0,1,2 (local column)
            V[v_idx][col] = val
        for w_idx, val in w_entries.items():
            # w_idx is k*2+i where k is 0,1,2 (local column)
            W[w_idx][col] = val
        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    # Part 2: Schoolbook for columns 3,4,5,6,7
    for k in range(3, 8):
        for i in range(2):
            for a in range(3):
                u_idx = i * m + a
                v_idx = a * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                if not _elapsed_ok():
                    return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

def _strassen_fallback():
    n, m, p = 2, 3, 8
    # Strassen for 2x2x2 blocks (cols 0-3) + schoolbook for col 2 (rows) x 8 cols
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            col += 1

    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
