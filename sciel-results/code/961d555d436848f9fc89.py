def compute_residual(params, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[0:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:size_u + size_v + size_w].reshape(p * n, rank)
    
    # We need T[i, j, k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
    # This is a bit tricky to vectorize directly without loops over i,j,k because of the index permutations.
    # However, we can permute the axes of U, V, W to align indices.
    
    # U indices: (i, k, r)
    U_3d = U.reshape(n, m, rank)
    # V indices: (k, j, r)
    V_3d = V.reshape(m, p, rank)
    # W indices: (j, i, r)
    W_3d = W.reshape(p, n, rank)
    
    # We want to sum over r.
    # Term for (i, j, k) is U_3d[i, k, r] * V_3d[k, j, r] * W_3d[j, i, r]
    
    # We can use einsum
    # 'ikr,kjr,jir->ijk'
    T = np.einsum('ikr,kjr,jir->ijk', U_3d, V_3d, W_3d)
    
    return (T - 1).flatten()
