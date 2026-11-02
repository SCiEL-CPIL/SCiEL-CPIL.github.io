def _verify_decomposition(U, V, W, n, m, p):
    # ...
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s_sum = 0.0
                for j in range(m):
                    for s in range(rank):
                        s_sum += U_arr[i * m + j, s] * V_arr[j * p + k, s] * W_arr[k * n + i, s]
                C_computed[i, k] = s_sum
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True
