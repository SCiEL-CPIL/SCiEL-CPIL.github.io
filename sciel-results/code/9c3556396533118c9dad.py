import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Strassen 2x2x2 terms
    # Returns list of (u, v, w) for 2x2 * 2x2 -> 2x2
    # u, v, w are length 4 vectors (flattened 2x2)
    def strassen_2x2():
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1]
        yield (u, v, w)
        
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1]
        yield (u, v, w)
        
        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1]
        yield (u, v, w)
        
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0]
        yield (u, v, w)
        
        # M5 = (A11+A12)B22 -> C12 += M5, C11 -= M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [0, 1, -1, 0]
        yield (u, v, w)
        
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1]
        yield (u, v, w)
        
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0]
        yield (u, v, w)

    # Schoolbook 2x2x1 terms
    # Returns list of (u, v, w) for 2x2 * 2x1 -> 2x1
    # u, v are length 4, w is length 2
    def schoolbook_2x2x1():
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        for i in range(2):
            for j in range(2):
                u = [0]*4
                v = [0]*4
                w = [0]*2
                u[i*2+j] = 1
                v[j*2+0] = 1 # B is 2x1, so B[j,0]
                w[i] = 1
                yield (u, v, w)
                # Wait, B is 2x1. Flattened B is [B00, B10].
                # v[j*2+0] is correct for B[j,0].
                # But we need to multiply by B1 and B2 separately?
                # No, the term is A[i,j] * B[j,0].
                # So for each i,j, we have one term.
                # Total 4 terms.
                # But the loop above yields 4 terms.
                # Let's re-verify.
                # C_i = sum_j A_ij B_j0.
                # Terms: A00 B00, A01 B10, A10 B00, A11 B10.
                # Yes, 4 terms.

    def expand_2x2_to_4x4(u_2x2, v_2x2, w_2x2, i_off, j_off, k_off, p_dim):
        """
        Expands a 2x2x2 or 2x2x1 term to the full 4x4x7 space.
        u_2x2: length 4
        v_2x2: length 4 (if 2x2x2) or length 2 (if 2x2x1? No, B is 4x7, so B block is 2x7 or 2x2)
        
        Actually, let's handle the blocks explicitly.
        """
        pass

    # Instead of complex expansion, let's use the block indices directly.
    
    def get_idx_A(i, j): return i * m + j
    def get_idx_B(j, k): return j * p + k
    def get_idx_W(k, i): return k * n + i

    # 1. 4x4x4 part (cols 0-3 of B)
    # 4 blocks of 2x2x2
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            k_off = 0
            # This 2x2x2 block multiplies A[i_off:i_off+2, j_off:j_off+2] 
            # by B[j_off:j_off+2, k_off:k_off+2]
            # But B is 4x7. The k_off here is the column offset in B.
            # For the 4x4x4 part, we are multiplying A (4x4) by B[:, 0:4] (4x4).
            # So k_off should be 0, 1, 2, 3?
            # No, the block is 2x2 in A, 2x2 in B.
            # B's rows are j, cols are k.
            # So for block (i_b, j_b), we use A rows i_off.., cols j_off..
            # And B rows j_off.., cols k_off..
            # Here k_off is the column index in B.
            # For the first 4 columns of B, we have 2 blocks of 2 columns.
            # So k_b in 0, 1.
            for k_b in range(2):
                k_off = k_b * 2
                for u, v, w in strassen_2x2():
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    for r in range(4):
                        lr = r // 2
                        lc = r % 2
                        U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                        V[get_idx_B(j_off+lr, k_off+lc)] += v[r]
                        W[get_idx_W(k_off+lc, i_off+lr)] += w[r]
                    add_term(U, V, W)

    # 2. 4x4x2 part (cols 4-5 of B)
    # 2 blocks of 2x2x2? No, B is 4x2.
    # A is 4x4.
    # We can split A into 4 blocks of 2x2.
    # B is 4x2. Split into 2 blocks of 2x2? No, B is 4 rows.
    # B[:, 4:6] is 4x2.
    # Split B into B_top (2x2) and B_bot (2x2).
    # A blocks: A00, A01, A10, A11.
    # C_top = A00 B_top + A01 B_bot.
    # C_bot = A10 B_top + A11 B_bot.
    # Each is 2x2x2.
    for i_b in range(2):
        i_off = i_b * 2
        for j_b in range(2):
            j_off = j_b * 2
            # We need to multiply A[i_off:i_off+2, j_off:j_off+2] 
            # with B[j_off:j_off+2, 4:6]
            # But B's rows are j. So B[j_off:j_off+2, 4:6] is 2x2.
            # This works.
            k_off = 4
            for u, v, w in strassen_2x2():
                U = [0] * (n * m)
                V = [0] * (m * p)
                W = [0] * (p * n)
                for r in range(4):
                    lr = r // 2
                    lc = r % 2
                    U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                    V[get_idx_B(j_off+lr, k_off+lc)] += v[r]
                    W[get_idx_W(k_off+lc, i_off+lr)] += w[r]
                add_term(U, V, W)

    # 3. 4x4x1 part (col 6 of B)
    # A is 4x4. B is 4x1.
    # Split A into 4 blocks of 2x2.
    # Split B into 2 blocks of 2x1.
    # C_top = A00 b_top + A01 b_bot.
    # C_bot = A10 b_top + A11 b_bot.
    # Each is 2x2x1.
    for i_b in range(2):
        i_off = i_b * 2
        for j_b in range(2):
            j_off = j_b * 2
            k_off = 6
            for u, v, w in schoolbook_2x2x1():
                U = [0] * (n * m)
                V = [0] * (m * p)
                W = [0] * (p * n)
                for r in range(4):
                    lr = r // 2
                    lc = r % 2
                    U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                    # v is length 2? No, schoolbook_2x2x1 yields v of length 4?
                    # In my schoolbook_2x2x1, v was length 4.
                    # But B is 2x1. Flattened B is length 2.
                    # Let's fix schoolbook_2x2x1 to yield v of length 2.
                    pass

    # Re-implement schoolbook_2x2x1 correctly
    def schoolbook_2x2x1_correct():
        for i in range(2):
            for j in range(2):
                u = [0]*4
                v = [0]*2
                w = [0]*2
                u[i*2+j] = 1
                v[j] = 1
                w[i] = 1
                yield (u, v, w)

    # Redo part 3
    terms.clear()
    # Part 1
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            for k_b in range(2):
                k_off = k_b * 2
                for u, v, w in strassen_2x2():
                    U = [0] * (n * m)
                    V = [0] * (m * p)
                    W = [0] * (p * n)
                    for r in range(4):
                        lr = r // 2
                        lc = r % 2
                        U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                        V[get_idx_B(j_off+lr, k_off+lc)] += v[r]
                        W[get_idx_W(k_off+lc, i_off+lr)] += w[r]
                    add_term(U, V, W)

    # Part 2
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            k_off = 4
            for u, v, w in strassen_2x2():
                U = [0] * (n * m)
                V = [0] * (m * p)
                W = [0] * (p * n)
                for r in range(4):
                    lr = r // 2
                    lc = r % 2
                    U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                    V[get_idx_B(j_off+lr, k_off+lc)] += v[r]
                    W[get_idx_W(k_off+lc, i_off+lr)] += w[r]
                add_term(U, V, W)

    # Part 3
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            k_off = 6
            for u, v, w in schoolbook_2x2x1_correct():
                U = [0] * (n * m)
                V = [0] * (m * p)
                W = [0] * (p * n)
                for r in range(4):
                    lr = r // 2
                    lc = r % 2
                    U[get_idx_A(i_off+lr, j_off+lc)] += u[r]
                for r in range(2):
                    V[get_idx_B(j_off+r, k_off)] += v[r]
                for r in range(2):
                    W[get_idx_W(k_off, i_off+r)] += w[r]
                add_term(U, V, W)

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
