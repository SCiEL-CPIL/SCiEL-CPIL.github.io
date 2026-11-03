def _verify_full(U, V, W, n, m, p):
    rank = U.shape[1]
    # Reshape U to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank) -> W[k, i, r]
    W3 = W.reshape(p, n, rank)
    
    # We need to check if sum_r U[i', j', r] V[j', k', r] W[k, i, r] == delta[i', i] * delta[k', k]
    # Let's compute the LHS tensor T[i', j', k', i, k]
    # Using einsum: 'ijr,jkr,ki r->ijkik' ? No, indices are i', j', k', i, k.
    # U3: i' j' r
    # V3: j' k' r
    # W3: k i r  (Note: W is indexed by k, i in the target C[i,k])
    
    # T[i', j', k', i, k] = sum_r U3[i', j', r] * V3[j', k', r] * W3[k, i, r]
    T = np.einsum('ijr,jkr,ki r->ijkik', U3, V3, W3)
    
    # Construct Target Tensor
    # Target[i', j', k', i, k] = 1 if i'==i and k'==k, else 0
    # Note: j' is free in the target?
    # If i'==i and k'==k, the term is A[i, j'] B[j', k]. This is part of the sum.
    # So Target is 1 for all j' if i'==i and k'==k.
    
    # Create indices
    i_prime = np.arange(n)
    j_prime = np.arange(m)
    k_prime = np.arange(p)
    i = np.arange(n)
    k = np.arange(p)
    
    # Meshgrid
    I_p, J_p, K_p, I, K = np.meshgrid(i_prime, j_prime, k_prime, i, k, indexing='ij')
    
    # Target condition: I_p == I and K_p == K
    Target = np.zeros((n, m, p, n, p))
    mask = (I_p == I) & (K_p == K)
    Target[mask] = 1.0
    
    if not np.allclose(T, Target, atol=1e-10):
        return False
    return True
