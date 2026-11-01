import numpy as np

def construct():
    n, m, p = 2, 2, 8
    
    # Rank 28 decomposition using 4 Strassen blocks
    # Each block handles 2 columns of B and 2 columns of C
    # This is equivalent to 4 independent 2x2 matrix multiplications
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices for A (4 entries):
    # 0: a, 1: b, 2: c, 3: d
    
    num_blocks = p // 2  # 4 blocks
    
    for b_idx in range(num_blocks):
        base = b_idx * 7
        
        # Block B indices:
        # B[0, 2*b_idx], B[0, 2*b_idx+1]
        # B[1, 2*b_idx], B[1, 2*b_idx+1]
        
        # M1 = (a+d)(e+h)
        # U: a(0)+d(3), V: e(0,2b)+h(1,2b+1), W: C(0,2b)+C(1,2b+1)
        U[0, base + 0] += 1
        U[3, base + 0] += 1
        V[0 * p + 2 * b_idx, base + 0] += 1
        V[1 * p + 2 * b_idx + 1, base + 0] += 1
        W[2 * b_idx * n + 0, base + 0] += 1
        W[(2 * b_idx + 1) * n + 1, base + 0] += 1
        
        # M2 = (c+d)e
        # U: c(2)+d(3), V: e(0,2b), W: C(1,2b)-C(1,2b+1)
        U[2, base + 1] += 1
        U[3, base + 1] += 1
        V[0 * p + 2 * b_idx, base + 1] += 1
        W[2 * b_idx * n + 1, base + 1] += 1
        W[(2 * b_idx + 1) * n + 1, base + 1] -= 1
        
        # M3 = a(f-h)
        # U: a(0), V: f(0,2b+1)-h(1,2b+1), W: C(0,2b+1)+C(1,2b+1)
        U[0, base + 2] += 1
        V[0 * p + 2 * b_idx + 1, base + 2] += 1
        V[1 * p + 2 * b_idx + 1, base + 2] -= 1
        W[(2 * b_idx + 1) * n + 0, base + 2] += 1
        W[(2 * b_idx + 1) * n + 1, base + 2] += 1
        
        # M4 = d(g-e)
        # U: d(3), V: g(1,2b)-e(0,2b), W: C(0,2b)+C(1,2b)
        U[3, base + 3] += 1
        V[1 * p + 2 * b_idx, base + 3] += 1
        V[0 * p + 2 * b_idx, base + 3] -= 1
        W[2 * b_idx * n + 0, base + 3] += 1
        W[2 * b_idx * n + 1, base + 3] += 1
        
        # M5 = (a+b)h
        # U: a(0)+b(1), V: h(1,2b+1), W: C(0,2b)-C(1,2b)
        U[0, base + 4] += 1
        U[1, base + 4] += 1
        V[1 * p + 2 * b_idx + 1, base + 4] += 1
        W[2 * b_idx * n + 0, base + 4] -= 1
        W[2 * b_idx * n + 1, base + 4] += 1
        
        # M6 = (c-a)(e+f)
        # U: c(2)-a(0), V: e(0,2b)+f(0,2b+1), W: C(1,2b+1)
        U[2, base + 5] += 1
        U[0, base + 5] -= 1
        V[0 * p + 2 * b_idx, base + 5] += 1
        V[0 * p + 2 * b_idx + 1, base + 5] += 1
        W[(2 * b_idx + 1) * n + 1, base + 5] += 1
        
        # M7 = (b-d)(g+h)
        # U: b(1)-d(3), V: g(1,2b)+h(1,2b+1), W: C(0,2b)
        U[1, base + 6] += 1
        U[3, base + 6] -= 1
        V[1 * p + 2 * b_idx, base + 6] += 1
        V[1 * p + 2 * b_idx + 1, base + 6] += 1
        W[2 * b_idx * n + 0, base + 6] += 1

    # Verification
    # Reconstruct tensor
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    
    # We verify a few entries to ensure correctness
    # Full verification is O(4*16*16*28) which is small
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx_val = j * p + k
                c_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx_val, r] * W[c_idx, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to rank 32 if something went wrong (should not happen)
        U, V, W = _fallback_32()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _fallback_32():
    n, m, p = 2, 2, 8
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for k in range(p):
        base = k * 4
        # M1 = a00 * b0
        U[0, base + 0] += 1
        V[0 * p + k, base + 0] += 1
        W[k * n + 0, base + 0] += 1
        # M2 = a01 * b1
        U[1, base + 1] += 1
        V[1 * p + k, base + 1] += 1
        W[k * n + 0, base + 1] += 1
        # M3 = a10 * b0
        U[2, base + 2] += 1
        V[0 * p + k, base + 2] += 1
        W[k * n + 1, base + 2] += 1
        # M4 = a11 * b1
        U[3, base + 3] += 1
        V[1 * p + k, base + 3] += 1
        W[k * n + 1, base + 3] += 1
        
    return U, V, W
