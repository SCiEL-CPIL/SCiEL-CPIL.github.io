    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    for r in range(rank):
                        u_val = ui[r]
                        if u_val == 0:
                            continue
                        v_val = vj[r]
                        if v_val == 0:
                            continue
                        w_val = wk[r]
                        if w_val == 0:
                            continue
                        total += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if total != 1:
                        return False
        return True
    