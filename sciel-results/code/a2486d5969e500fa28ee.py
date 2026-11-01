            if np.allclose(C_decomp, C_expected, atol=1e-10):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
