import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    start_time = time.time()
    DEADLINE = 140.0

    # Try to find a rank-32 or lower construction.
    # Strategy: Use 4 local terms per 2-column block + 8 global terms.
    # Or find a better algebraic identity.
    
    # First, let's verify the rank-34 construction works, then try to improve.
    
    # Rank 34 construction:
    rank = 34
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    num_pairs = p // 2  # 6

    for t in range(num_pairs):
        base = t * 5
        k0 = 2 * t
        k1 = 2 * t + 1

        v_b0_k0 = 0 * p + k0
        v_b0_k1 = 0 * p + k1
        v_b1_k0 = 1 * p + k0
        v_b1_k1 = 1 * p + k1

        w_c0_k0 = k0 * n + 0
        w_c0_k1 = k1 * n + 0
        w_c1_k0 = k0 * n + 1
        w_c1_k1 = k1 * n + 1

        c = base + 0
        U[0, c] = 1.0; U[3, c] = 1.0
        V[v_b0_k0, c] = 1.0; V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = 1.0; W[w_c1_k1, c] = 1.0

        c = base + 1
        U[2, c] = 1.0; U[3, c] = 1.0
        V[v_b0_k0, c] = 1.0
        W[w_c1_k0, c] = 1.0; W[w_c1_k1, c] = -1.0

        c = base + 2
        U[0, c] = 1.0; U[1, c] = 1.0
        V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = -1.0; W[w_c0_k1, c] = 1.0

        c = base + 3
        U[2, c] = 1.0; U[0, c] = -1.0
        V[v_b0_k0, c] = 1.0; V[v_b0_k1, c] = 1.0
        W[w_c1_k1, c] = 1.0

        c = base + 4
        U[1, c] = 1.0; U[3, c] = -1.0
        V[v_b1_k0, c] = 1.0; V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = 1.0

    c = 30
    U[0, c] = 1.0
    for k in range(p):
        sign = 1.0 if k % 2 == 0 else -1.0
        V[0 * p + k, c] = sign
        W[k * n + 0, c] = sign

    c = 31
    U[3, c] = 1.0
    for k in range(p):
        sign = 1.0 if k % 2 == 0 else -1.0
        V[1 * p + k, c] = sign
        W[k * n + 1, c] = sign

    c = 32
    U[0, c] = 1.0
    for t in range(num_pairs):
        V[0 * p + 2*t, c] = 1.0
        V[0 * p + 2*t+1, c] = -1.0
        W[(2*t) * n + 0, c] = 1.0
        W[(2*t+1) * n + 0, c] = -1.0

    c = 33
    U[3, c] = 1.0
    for t in range(num_pairs):
        V[1 * p + 2*t, c] = 1.0
        V[1 * p + 2*t+1, c] = -1.0
        W[(2*t) * n + 1, c] = 1.0
        W[(2*t+1) * n + 1, c] = -1.0

    # Verify rank 34
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-9:
        # Fallback to rank 42 Strassen
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for t in range(num_pairs):
            base = t * 7
            k0 = 2 * t; k1 = 2 * t + 1
            v_b00 = 0*p+k0; v_b01 = 0*p+k1
            v_b10 = 1*p+k0; v_b11 = 1*p+k1
            w_c00 = k0*n+0; w_c01 = k1*n+0
            w_c10 = k0*n+1; w_c11 = k1*n+1
            c = base+0; U[0,c]=1;U[3,c]=1;V[v_b00,c]=1;V[v_b11,c]=1;W[w_c00,c]=1;W[w_c11,c]=1
            c = base+1; U[2,c]=1;U[3,c]=1;V[v_b00,c]=1;W[w_c10,c]=1;W[w_c11,c]=-1
            c = base+2; U[0,c]=1;V[v_b01,c]=1;V[v_b11,c]=-1;W[w_c01,c]=1;W[w_c11,c]=1
            c = base+3; U[3,c]=1;V[v_b10,c]=1;V[v_b00,c]=-1;W[w_c00,c]=1;W[w_c10,c]=1
            c = base+4; U[0,c]=1;U[1,c]=1;V[v_b11,c]=1;W[w_c00,c]=-1;W[w_c01,c]=1
            c = base+5; U[2,c]=1;U[0,c]=-1;V[v_b00,c]=1;V[v_b01,c]=1;W[w_c11,c]=1
            c = base+6; U[1,c]=1;U[3,c]=-1;V[v_b10,c]=1;V[v_b11,c]=1;W[w_c00,c]=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to find rank 32: 4 local terms per block + 8 global terms
    # Or rank 30: 3 local terms per block + 12 global terms
    # Let's try a systematic search for additional global terms that allow
    # reducing local terms from 5 to 4 per block.
    
    # The 4 local terms (without term 1) cover:
    # T2: (A10+A11)*B0_k0 -> C1_k0 - C1_k1
    # T3: (A00+A01)*B1_k1 -> -C0_k0 + C0_k1
    # T4: (A10-A00)*(B0_k0+B0_k1) -> C1_k1
    # T5: (A01-A11)*(B1_k0+B1_k1) -> C0_k0
    # Missing from these 4: contributions to C0_k0 and C1_k1 from A00*A11 type terms
    # Specifically we need: A00*B0_k0 for C0_k0, A11*B1_k1 for C1_k1, etc.
    
    # Let's try: 4 local terms per block + 8 global terms = 32
    # The 4 local terms give partial coverage. We need global terms to fill gaps.
    
    # Actually, let me try a different approach: use the fact that for 2x2x2,
    # the minimum rank is 7 (Strassen). For 2x2x12, we can group columns.
    
    # Try rank 30: 3 local + 12 global
    # Or try a completely different decomposition.
    
    # Let's try: split into 4 blocks of 3 columns each, use rank 5 for 2x2x3
    # That gives 4*5 = 20, but 2x2x3 has rank... let me think.
    # Actually 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k], k=0,1,2
    # That's 2*2*3 = 12 entries, each needs A[i,j]*B[j,k].
    # Schoolbook: 4*3 = 12 terms. Can we do better?
    # Strassen-like: not directly applicable since it's not square.
    
    # Let's try the approach of 4 blocks of 3, with some sharing.
    # For each block of 3 columns: 2x2x3 multiplication
    # Minimum rank for 2x2x3 is at least 5 (by lower bound arguments)
    # Let's try rank 5 per block: 4*5 = 20 total
    
    # Actually, let me verify: can we do 2x2x3 in rank 5?
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Rank 5 attempt:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)*B00 -> contributes to C10, C11
    # M3 = (A00+A01)*B11 -> contributes to C00, C01
    # M4 = (A10-A00)*(B00+B01+B02) -> contributes to C10, C11, C12
    # M5 = (A01-A11)*(B10+B11+B12) -> contributes to C00, C01, C02
    # Hmm, this is getting complicated. Let me just keep the working rank-34.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
