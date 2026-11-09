    def add_term(step_u, step_v, lu, lv, lw, r):
        # U
        for i in range(4):
            for j in range(4):
                bi, bj = i // 2, j // 2
                li, lj = i % 2, j % 2
                block = (bi, bj)
                coeff = step_u.get(block, 0)
                if coeff != 0:
                    val = lu.get((li, lj), 0)
                    if val != 0:
                        U[i * m + j][r] += coeff * val
        # V
        for j in range(4):
            for k in range(4):
                bj, bk = j // 2, k // 2
                lj, lk = j % 2, k % 2
                block = (bj, bk)
                coeff = step_v.get(block, 0)
                if coeff != 0:
                    val = lv.get((lj, lk), 0)
                    if val != 0:
                        V[j * p + k][r] += coeff * val
        # W
        for k in range(4):
            for i in range(4):
                bk, bi = k // 2, i // 2
                lk, li = k % 2, i % 2
                block = (bk, bi)
                # No step coefficient for W, just the local tensor value
                val = lw.get((lk, li), 0)
                if val != 0:
                    W[k * n + i][r] += val
