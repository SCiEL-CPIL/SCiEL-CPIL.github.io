<<<<<<< SEARCH
    if time.time() - _start > 145:
        raise TimeoutError("Exceeded time budget")
    return {"U": U, "V": V, "W": W}
=======
    # Verify correctness: check that sum_k U[:,k] * V[:,k] * W[:,k] = tensor
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # The tensor T[i, j, k] for matrix mult: T[a*m+i, b*p+j, j'*n+i'] = delta(i,i')delta(a,b)delta(j,j')
    # Where a indexes rows of A (n), i indexes cols of A (m)
    #       b indexes rows of B (m), j indexes cols of B (p)  
    #       j' indexes rows of C (p), i' indexes cols of C (n)
    # Actually: C[i',j'] = sum_{a,b} A[i',a] * B[a,j'] ... 
    # Standard: A is n×m, B is m×p, C is n×p
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: T[(i,k),(k,j),(i,j)] = 1 if indices match, else 0
    # U has n*m = 8 rows, V has m*p = 24 rows, W has p*n = 12 rows
    
    # Check: for each rank-1 component r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # should sum to the multiplication tensor
    # Tensor T[u,v,w] where u in [0,8), v in [0,24), w in [0,12)
    # u = i*m + k, v = k'*p + j, w = j'*n + i'
    # T[u,v,w] = 1 if k==k' and i==i' and j==j' else 0
    
    # Compute the reconstructed tensor and compare
    # For efficiency, check a few random entries
    np.random.seed(42)
    max_err = 0.0
    for _ in range(200):
        u = np.random.randint(0, U_rows)
        v = np.random.randint(0, V_rows)
        w = np.random.randint(0, W_rows)
        # Expected value
        i, k = divmod(u, m)
        k2, j = divmod(v, p)
        j2, i2 = divmod(w, n)
        expected = 1.0 if (k == k2 and i == i2 and j == j2) else 0.0
        # Computed value: sum_r U[u,r] * V[v,r] * W[w,r]
        computed = np.dot(np.dot(U_arr[u, :], V_arr[v, :]), W_arr[w, :])
        err = abs(computed - expected)
        if err > max_err:
            max_err = err
    
    if max_err > 1e-10:
        # Fallback: return a known-valid schoolbook rank-48 solution
        rank48 = 48
        U48 = [[0.0] * rank48 for _ in range(U_rows)]
        V48 = [[0.0] * rank48 for _ in range(V_rows)]
        W48 = [[0.0] * rank48 for _ in range(W_rows)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U48[i*m + k][col] = 1.0
                    V48[k*p + j][col] = 1.0
                    W48[j*n + i][col] = 1.0
                    col += 1
        if time.time() - _start > 145:
            raise TimeoutError("Exceeded time budget")
        return {"U": U48, "V": V48, "W": W48}
    
    if time.time() - _start > 145:
        raise TimeoutError("Exceeded time budget")
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
