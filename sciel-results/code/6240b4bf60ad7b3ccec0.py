    def _validate(candidate):
        # ...
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    acc = 0
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        if u_val:
                            v_val = v_row[c]
                            if v_val:
                                acc += u_val * v_val * w_row[c]
                    if acc != 1:
                        return False
        # Check invalid entries are 0
        # ...
