                            # Block I, J of C
                            # Row in C: 2I + pz//2 (this is i)
                            # Col in C: 2J + pz%2 (this is k)
                            i_c = 2 * I + (pz // 2)
                            k_c = 2 * J + (pz % 2)
                            W[k_c * 6 + i_c][column] += c
