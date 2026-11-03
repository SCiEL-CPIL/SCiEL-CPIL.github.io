<<<<<<< SEARCH
    # Fill U, V, W
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    # U: A components
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob][r] = beta[s][I][J]
                                
                    # V: B components
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob][r] = gamma[s][K][L]
                                
                    # W: C components
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                i_glob = I * 3 + a
                                k_glob = K * 3 + c
                                W[k_glob * n + i_glob][r] = alpha[s][I][K]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    # Fill U, V, W
    for s in range(7):
        if time.time() > deadline - 5.0:
            break
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    # U: A components
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob][r] = beta[s][I][J]
                                
                    # V: B components
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob][r] = gamma[s][K][L]
                                
                    # W: C components
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                i_glob = I * 3 + a
                                k_glob = K * 3 + c
                                W[k_glob * n + i_glob][r] = alpha[s][I][K]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
