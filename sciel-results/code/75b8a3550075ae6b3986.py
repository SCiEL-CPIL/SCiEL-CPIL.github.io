                    else:
                        # Revert
                        for t_idx in added:
                            remove_triple(t_idx)
                        add_triple(t_rem1)
                        add_triple(t_rem2)
