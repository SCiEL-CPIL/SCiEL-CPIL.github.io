                if error < 1e-6:
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify
                    error_rounded = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                                error_rounded += (val - 1) ** 2
                    
                    if error_rounded < 1e-6:
                        converged = True
                        U = U_rounded
                        V = V_rounded
                        W = W_rounded
                        break
