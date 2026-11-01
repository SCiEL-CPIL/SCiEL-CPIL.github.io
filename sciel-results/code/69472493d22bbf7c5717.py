<<<<<<< SEARCH
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
                col += 1
                
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to trivial rank 48 if verification fails (should not happen)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = (i * m + k) * p + j
                    U_f[i * m + k][idx] = 1
                    V_f[k * p + j][idx] = 1
                    W_f[j * n + i][idx] = 1
        return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
