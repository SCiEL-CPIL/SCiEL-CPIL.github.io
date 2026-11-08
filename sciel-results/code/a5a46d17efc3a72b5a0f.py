    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c]
                        if uv == 0:
                            continue
                        vv = v_row[c]
                        if vv == 0:
                            continue
                        ww = w_row[c]
                        if ww == 0:
                            continue
                        acc += uv * vv * ww
                    if acc != 1:
                        return False
        return True
