<<<<<<< SEARCH
    # Add 4x4x4 block at (0,0,0)
    for t1 in strassen_terms:
        for t2 in strassen_terms:
            U1, V1, W1 = t1
            U2, V2, W2 = t2
            add_term(kron_u(U1, U2), kron_v(V1, V2), kron_w(W1, W2))
=======
    def remap_u(d):
        res = {}
        for idx, val in d.items():
            i = idx // 4
            k = idx % 4
            new_idx = i * m + k
            if new_idx in res:
                res[new_idx] += val
            else:
                res[new_idx] = val
        return res

    def remap_v(d):
        res = {}
        for idx, val in d.items():
            k = idx // 4
            j = idx % 4
            new_idx = k * p + j
            if new_idx in res:
                res[new_idx] += val
            else:
                res[new_idx] = val
        return res

    def remap_w(d):
        res = {}
        for idx, val in d.items():
            j = idx // 4
            i = idx % 4
            new_idx = j * n + i
            if new_idx in res:
                res[new_idx] += val
            else:
                res[new_idx] = val
        return res

    # Add 4x4x4 block at (0,0,0)
    for t1 in strassen_terms:
        for t2 in strassen_terms:
            U1, V1, W1 = t1
            U2, V2, W2 = t2
            U_new = remap_u(kron_u(U1, U2))
            V_new = remap_v(kron_v(V1, V2))
            W_new = remap_w(kron_w(W1, W2))
            add_term(U_new, V_new, W_new)
>>>>>>> REPLACE
