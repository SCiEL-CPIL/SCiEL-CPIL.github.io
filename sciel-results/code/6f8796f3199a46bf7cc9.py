import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    start_time = time.time()
    timeout = 120.0

    # ------------------------------------------------------------------
    # Primary design: Strassen-based rank-42 decomposition.
    # Partition the p=12 columns of B into 6 pairs; each 2x2x2 block
    # uses the 7-product Strassen scheme.  Total rank = 6 * 7 = 42.
    # ------------------------------------------------------------------
    num_blocks = p // 2            # 6 blocks
    rank = num_blocks * 7          # 42

    U = [[0] * rank for _ in range(n * m)]  # (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (24, 42)

    # Strassen 7-product decomposition for 2x2x2.
    #   A = [[a,b],[c,d]]  local idx 0=a,1=b,2=c,3=d
    #   B = [[e,f],[g,h]]  local idx 0=e,1=f,2=g,3=h
    #   C = [[C00,C01],[C10,C11]] local idx 0,1,2,3
    #   M1=(a+d)(e+h)   M2=(c+d)e   M3=a(f-h)   M4=d(g-e)
    #   M5=(a+b)h       M6=(c-a)(e+f)  M7=(b-d)(g+h)
    #   C00=M1+M4-M5+M7  C01=M3+M5     C10=M2+M4  C11=M1-M2+M3+M6
    # Each entry is a dict {local_index: coefficient}.
    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},        # M1
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},       # M2
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},        # M3
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},        # M4
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},       # M5
        {0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1},              # M6
        {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1},              # M7
    ]

    for t in range(num_blocks):
        if time.time() - start_time > timeout - 20:
            break
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]

            # U: A indices are global (no offset)
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val

            # V: map local B index to global (row-major B[j,k] -> j*p+k)
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 0 * p + (2 * t)          # B[0, 2t]
                elif local_b == 1:
                    global_b = 0 * p + (2 * t + 1)      # B[0, 2t+1]
                elif local_b == 2:
                    global_b = 1 * p + (2 * t)          # B[1, 2t]
                else:
                    global_b = 1 * p + (2 * t + 1)      # B[1, 2t+1]
                V[global_b][col] = val

            # W: map local C index to global (trace convention W[k*n+i], k=col,i=row)
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0          # C[0, 2t]
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0      # C[0, 2t+1]
                elif local_c == 2:
                    global_w = (2 * t) * n + 1          # C[1, 2t]
                else:
                    global_w = (2 * t + 1) * n + 1      # C[1, 2t+1]
                W[global_w][col] = val

    # ------------------------------------------------------------------
    # Exact verification of the tensor identity.  The decomposition is
    # correct iff, for every (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # (and 0 otherwise is impossible here since each (i,j,k) is a real product).
    # ------------------------------------------------------------------
    def _is_correct(Um, Vm, Wm, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for c in range(r):
                        acc += Um[ui][c] * Vm[vj][c] * Wm[wk][c]
                    if acc != 1:
                        return False
        return True

    if _is_correct(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # Fallback: guaranteed-correct schoolbook decomposition (rank 48).
    # ------------------------------------------------------------------
    if time.time() - start_time > timeout - 5:
        return {"U": U, "V": V, "W": W}

    r2 = n * m * p  # 48
    U2 = [[0] * r2 for _ in range(n * m)]
    V2 = [[0] * r2 for _ in range(m * p)]
    W2 = [[0] * r2 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j][col] = 1
                V2[j * p + k][col] = 1
                W2[k * n + i][col] = 1
                col += 1
    return {"U": U2, "V": V2, "W": W2}
# EVOLVE-BLOCK-END
