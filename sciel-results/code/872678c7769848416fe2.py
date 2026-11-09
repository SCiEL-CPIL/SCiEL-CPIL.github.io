                                if deg[w] > min_deg:
                                    # Perform swap
                                    remove_edge(v, w)
                                    add_edge(u, v)
