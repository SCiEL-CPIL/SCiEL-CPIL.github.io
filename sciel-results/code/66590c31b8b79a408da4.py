                    for v in verts_in_P:
                        new_P = P_mask & adj[v]
                        expand(new_P, size + 1, ...)
                    