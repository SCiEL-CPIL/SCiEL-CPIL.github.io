                for nb_idx in neighbor_indices[c_idx]:
                    cov[nb_idx] -= 1
                    affected.append(nb_idx)
                    if cov[nb_idx] == 0:
                        valid = False
                        break
