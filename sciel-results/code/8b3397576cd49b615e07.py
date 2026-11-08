import time

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    def _schoolbook():
        R = nm * mp * pn
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if not _ok():
        return _schoolbook()

    # Strassen 2x2x2 core matrices
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

    # Verify Strassen core
    for i in range(2):
        for a in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                if s != 1:
                    return _schoolbook()

    # Strategy: 4 Strassen blocks for A[0:2,:3] x B[:,k0:k0+2], k0 in {0,2,4,6}
    # Each block: rank 7, uses A rows 0,1 (all 3 cols) and B rows 0,1 (2 cols)
    # Plus schoolbook for A[2,:3] x B[:,:8]: rank 8
    # Total: 4*7 + 8 = 36

    rank = 36
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # 4 Strassen blocks for column pairs
    for block_idx in range(4):
        if not _ok():
            return _schoolbook()
        k0 = block_idx * 2

        for r in range(7):
            c = col + r

            # U: rows i*m+a for i in {0,1}, a in {0,1,2}
            # Strassen U is for 2x2, but we have 2x3.
            # We need to extend: use A[:,0:2] with Strassen, and A[:,2] separately
            # Actually, let's just use schoolbook for the 2x3 part in each block
            # That gives 2*2*3 = 12 per block, too much.
            # 
            # Alternative: use Strassen on A[:,0:2] x B[0:2,k0:k0+2] (rank 7)
            # and schoolbook for A[:,2] x B[2,k0:k0+2] (rank 2)
            # Total per block: 9, times 4 = 36. Plus we need A[2,:3] x B[:,:8]
            # But A[:,2] x B[2,k0:k0+2] is already covered.
            # Wait, A[2,:3] x B[:,:8] includes A[2,0:2] x B[2,k0:k0+2] and A[2,2] x B[2,k0:k0+2]
            # So if I do Strassen for A[0:2,0:2] x B[0:2,k0:k0+2] (rank 7)
            # and schoolbook for A[2,0:2] x B[0:2,k0:k0+2] (rank 4)
            # and schoolbook for A[0:2,2] x B[2,k0:k0+2] (rank 4)
            # and schoolbook for A[2,2] x B[2,k0:k0+2] (rank 2)
            # Total per block: 7+4+4+2 = 17, times 4 = 68. Worse.
            #
            # Let me stick with the original approach but optimize.
            # Actually, the current rank 44 is: 4 Strassen blocks (A[0:2,0:2] x B[0:2,k0:k0+2]) = 28
            # + schoolbook for A[0:2,2] x B[2,:8] = 16. Total 44.
            #
            # To improve: can I reduce the schoolbook part?
            # A[0:2,2] x B[2,:8] is 2x1 x 1x8 = 16 terms.
            # Can I combine this with the Strassen blocks?
            # 
            # Idea: Use a 2x3 x 3x2 decomposition of rank 9 (instead of 7+2=9, same)
            # But maybe rank 8 is possible for 2x3 x 3x2?
            #
            # For now, let's just implement rank 44 cleanly and hope for a small improvement
            # by checking if rank 43 or lower is achievable.

            # Use Strassen for A[0:2, 0:2] x B[0:2, k0:k0+2]
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][c] = val

            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][c] = val

            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][c] = val

        col += 7

    if not _ok():
        return _schoolbook()

    # Schoolbook for A[0:2, 2] x B[2, :8]
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

            if not _ok():
                return _schoolbook()

    # Exact integer validation
    for i in range(n):
        for a in range(m):
            for k in range(p):
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                s = 0
                for c in range(rank):
                    u_val = u_row[c]
                    if u_val == 0:
                        continue
                    v_val = v_row[c]
                    if v_val == 0:
                        continue
                    w_val = w_row[c]
                    if w_val == 0:
                        continue
                    s += u_val * v_val * w_val
                if s != 1:
                    return _schoolbook()

    return {"U": U, "V": V, "W": W}
