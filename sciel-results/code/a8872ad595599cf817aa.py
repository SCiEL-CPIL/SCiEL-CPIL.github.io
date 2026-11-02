def objective_vectorized(params, rank):
    nm = n * m
    mp = m * p
    pn = p * n
    U = params[:rank * nm].reshape(rank, nm).T
    V = params[rank * nm : rank * (nm + mp)].reshape(rank, mp).T
    W = params[rank * (nm + mp) :].reshape(rank, pn).T
    
    # Compute all T_ijk
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T_ijk = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # dot product of U[u_idx, :], V[v_idx, :], W[w_idx, :]
                # This is not a simple dot product of two vectors, it's a sum of products of three vectors.
                # val = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                # We can compute this as:
                # temp_c = U[u_idx, c] * V[v_idx, c]
                # val = dot(temp_c, W[w_idx, :])
                temp = U[u_idx, :] * V[v_idx, :]
                val = np.dot(temp, W[w_idx, :])
                errors.append((val - 1.0) ** 2)
                
    return sum(errors)
