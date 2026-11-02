def compute_subgrad(h, k):
    N = len(h)
    G = np.zeros(N)
    # c_k = sum_{i} h_i h_{i+k}
    # d c_k / d h_j = h_{j+k} + h_{j-k}
    
    if k >= 0:
        # h_{j+k} is non-zero for j such that j+k < N
        # indices j: 0 to N-1-k
        # G[0:N-k] += h[k:N]
        G[:N-k] += h[k:]
        
        # h_{j-k} is non-zero for j such that j-k >= 0
        # indices j: k to N-1
        # G[k:N] += h[0:N-k]
        G[k:] += h[:N-k]
    else:
        k_abs = -k
        # Same logic, just symmetric
        G[:N-k_abs] += h[k_abs:]
        G[k_abs:] += h[:N-k_abs]
        
    return G
