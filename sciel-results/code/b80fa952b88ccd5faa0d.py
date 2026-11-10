    def build_tensor_product(outer_terms, inner_terms, outer_dim, inner_dim):
        rank = len(outer_terms) * len(inner_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for (u_o, v_o, w_o) in outer_terms:
            for (u_i, v_i, w_i) in inner_terms:
                for ik_idx, co in u_o.items():
                    i, k = divmod(ik_idx, outer_dim)
                    for elem, ci in u_i.items():
                        a, b = divmod(elem, inner_dim)
                        U[(outer_dim*i + a) * m + (outer_dim*k + b)][col] += co * ci
                for kj_idx, co in v_o.items():
                    k, j = divmod(kj_idx, outer_dim)
                    for elem, ci in v_i.items():
                        a, b = divmod(elem, inner_dim)
                        V[(outer_dim*k + a) * p + (outer_dim*j + b)][col] += co * ci
                for ij_idx, co in w_o.items():
                    i, j = divmod(ij_idx, outer_dim)
                    for elem, ci in w_i.items():
                        a, b = divmod(elem, inner_dim)
                        W[(outer_dim*j + b) * n + (outer_dim*i + a)][col] += co * ci
                col += 1
        return U, V, W
