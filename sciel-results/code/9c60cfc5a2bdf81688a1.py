"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Split 5 into 4+1
    # 2x4x6: two 2x2x6 blocks, each with rank 21 (3 Strassen 2x2x2 blocks)
    # 2x1x6: rank 12 (standard)
    # Total: 2*21 + 12 = 54
    
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1
    
    # Strassen block for 2x2x2: (i∈{0,1}, j∈{ja,jb}, k∈{ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        # Local indices in the 2x2x2 block
        # A: rows 0,1; cols ja, jb
        # B: rows ja, jb; cols ka, kb
        # C: rows ka, kb; cols 0,1
        
        # U indices: i * m + j
        u_0_ja = 0 * m + ja
        u_0_jb = 0 * m + jb
        u_1_ja = 1 * m + ja
        u_1_jb = 1 * m + jb
        
        # V indices: j * p + k
        v_ja_ka = ja * p + ka
        v_ja_kb = ja * p + kb
        v_jb_ka = jb * p + ka
        v_jb_kb = jb * p + kb
        
        # W indices: k * n + i
        w_ka_0 = ka * n + 0
        w_ka_1 = ka * n + 1
        w_kb_0 = kb * n + 0
        w_kb_1 = kb * n + 1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11 -> u_0_ja, u_1_jb
        # V: B00, B11 -> v_ja_ka, v_jb_kb
        # W: C00, C11 -> w_ka_0, w_kb_1
        add_term([(u_0_ja, 1), (u_1_jb, 1)],
                 [(v_ja_ka, 1), (v_jb_kb, 1)],
                 [(w_ka_0, 1), (w_kb_1, 1)])
        
        # M2: (A10+A11)(B00)
        # U: A10, A11 -> u_1_ja, u_1_jb
        # V: B00 -> v_ja_ka
        # W: C10, C11 -> w_ka_1, w_kb_1
        add_term([(u_1_ja, 1), (u_1_jb, 1)],
                 [(v_ja_ka, 1)],
                 [(w_ka_1, 1), (w_kb_1, -1)])
        
        # M3: (A00)(B01-B11)
        # U: A00 -> u_0_ja
        # V: B01, B11 -> v_ja_kb, v_jb_kb
        # W: C01, C11 -> w_kb_0, w_kb_1
        add_term([(u_0_ja, 1)],
                 [(v_ja_kb, 1), (v_jb_kb, -1)],
                 [(w_kb_0, 1), (w_kb_1, 1)])
        
        # M4: (A11)(B10-B00)
        # U: A11 -> u_1_jb
        # V: B10, B00 -> v_jb_ka, v_ja_ka
        # W: C00, C10 -> w_ka_0, w_ka_1
        add_term([(u_1_jb, 1)],
                 [(v_jb_ka, 1), (v_ja_ka, -1)],
                 [(w_ka_0, 1), (w_ka_1, 1)])
        
        # M5: (A00+A01)(B11)
        # U: A00, A01 -> u_0_ja, u_0_jb
        # V: B11 -> v_jb_kb
        # W: C00, C01 -> w_ka_0, w_kb_0
        add_term([(u_0_ja, 1), (u_0_jb, 1)],
                 [(v_jb_kb, 1)],
                 [(w_ka_0, -1), (w_kb_0, 1)])
        
        # M6: (A10-A00)(B00+B01)
        # U: A10, A00 -> u_1_ja, u_0_ja
        # V: B00, B01 -> v_ja_ka, v_ja_kb
        # W: C11 -> w_kb_1
        add_term([(u_1_ja, 1), (u_0_ja, -1)],
                 [(v_ja_ka, 1), (v_ja_kb, 1)],
                 [(w_kb_1, 1)])
        
        # M7: (A01-A11)(B10+B11)
        # U: A01, A11 -> u_0_jb, u_1_jb
        # V: B10, B11 -> v_jb_ka, v_jb_kb
        # W: C00 -> w_ka_0
        add_term([(u_0_jb, 1), (u_1_jb, -1)],
                 [(v_jb_ka, 1), (v_jb_kb, 1)],
                 [(w_ka_0, 1)])
    
    # 2x4x6: j in {0,1,2,3}, k in {0,...,5}
    # Block 1: j in {0,1}, k in {0,1}
    strassen_block(0, 1, 0, 1)
    # Block 2: j in {0,1}, k in {2,3}
    strassen_block(0, 1, 2, 3)
    # Block 3: j in {0,1}, k in {4,5}
    strassen_block(0, 1, 4, 5)
    # Block 4: j in {2,3}, k in {0,1}
    strassen_block(2, 3, 0, 1)
    # Block 5: j in {2,3}, k in {2,3}
    strassen_block(2, 3, 2, 3)
    # Block 6: j in {2,3}, k in {4,5}
    strassen_block(2, 3, 4, 5)
    
    # 2x1x6: j=4, k in {0,...,5}
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    # Verify the decomposition
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    S = np.einsum('ijt,jkt,kit->ijk', Ua, Va, Wa)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        # Fallback to naive if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    def to_clean(arr):
        out = []
        for row in arr:
            out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
        return out
    
    return {"U": to_clean(U), "V": to_clean(V), "W": to_clean(W)}
# EVOLVE-BLOCK-END
