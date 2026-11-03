                # Update U
                VW = np.kron(V, W) # Shape (9*6, R) = (54, R)
                for i in range(n*m):
                    vec_Ti = T[i, :, :].flatten()
                    u_i, residuals, rank, s = np.linalg.lstsq(VW, vec_Ti, rcond=None)
                    U[i, :] = u_i
