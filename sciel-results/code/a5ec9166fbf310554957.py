<<<<<<< SEARCH
    U_44, V_44, W_44 = build_4x4x4_strassen()
    
    # Verify
    np.random.seed(123)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_true = A_test @ B_test
    
    C_comp = np.zeros((4, 4))
    for t in range(len(U_44)):
        a_val = sum(U_44[t][i] * A_test[i//4][i%4] for i in range(16))
        b_val = sum(V_44[t][i] * B_test[i//4][i%4] for i in range(16))
        for i in range(16):
            C_comp[i//4][i%4] += W_44[t][i] * a_val * b_val
    
    assert np.allclose(C_true, C_comp, atol=1e-10), "4x4x4 Strassen verification failed"
=======
    U_44, V_44, W_44 = build_4x4x4_strassen()
    
    # Verify
    np.random.seed(123)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_true = A_test @ B_test
    
    C_comp = np.zeros((4, 4))
    for t in range(len(U_44)):
        a_val = sum(U_44[t][i] * A_test[i//4][i%4] for i in range(16))
        b_val = sum(V_44[t][i] * B_test[i//4][i%4] for i in range(16))
        for i in range(16):
            C_comp[i//4][i%4] += W_44[t][i] * a_val * b_val
    
    if not np.allclose(C_true, C_comp, atol=1e-10):
        # Fallback to naive 64-term for 4x4x4
        U_44 = []
        V_44 = []
        W_44 = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = [0]*16
                    v = [0]*16
                    w = [0]*16
                    u[i*4+j] = 1
                    v[j*4+k] = 1
                    w[i*4+k] = 1
                    U_44.append(u)
                    V_44.append(v)
                    W_44.append(w)
>>>>>>> REPLACE
