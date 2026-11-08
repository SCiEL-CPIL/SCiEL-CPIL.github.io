                    val = np.dot(
                        U_mat[i * m + j, :] * V_mat[j * p + k, :],
                        W_mat[k * n + i, :]
                    )
                    if abs(val - 1.0) > 1e-9:
                        return False
