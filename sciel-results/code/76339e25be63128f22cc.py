<<<<<<< SEARCH
    # Verify the 6x6 decomposition using vectorized operations
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test

        U66_r = U66.reshape(n, m, rank)  # [i, j, r]
        V66_r = V66.reshape(m, p, rank)  # [j, k, r]
        W66_r = W66.reshape(p, n, rank)  # [k, i, r]
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U66_r[:, :, r]  # [i, j]
            Vr = V66_r[:, :, r]  # [j, k]
            Wr = W66_r[:, :, r]  # [k, i]
            A_ur = A_test * Ur
            B_vr = B_test * Vr
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += A_ur[i, j] * B_vr[j, k] * Wr[k, i]
                    C_computed[i, k] += s

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to 3x3 schoolbook x Strassen
            return _construct_3x3_strassen(n, m, p, deadline)
=======
    # Verify the 6x6 decomposition using vectorized operations
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test

        U66_r = U66.reshape(n, m, rank)  # [i, j, r]
        V66_r = V66.reshape(m, p, rank)  # [j, k, r]
        W66_r = W66.reshape(p, n, rank)  # [k, i, r]
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U66_r[:, :, r]  # [i, j]
            Vr = V66_r[:, :, r]  # [j, k]
            Wr = W66_r[:, :, r]  # [k, i]
            A_ur = A_test * Ur
            B_vr = B_test * Vr
            AB = A_ur @ B_vr  # [i, k]
            C_computed += AB * Wr.T  # [i, k] * [i, k] (since Wr[k,i] = Wr.T[i,k])

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to 3x3 schoolbook x Strassen
            return _construct_3x3_strassen(n, m, p, deadline)
>>>>>>> REPLACE

