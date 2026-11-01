def verify_broadcast(U, V, W, rank):
    # U: (9, rank), V: (9, rank), W: (9, rank)
    # Reshape
    U3 = U.reshape(3, 3, rank)
    V3 = V.reshape(3, 3, rank)
    W3 = W.reshape(3, 3, rank)
    
    # T[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # Use einsum
    # 'ijr,jkr,kr->ijk' ? No, W3[k, i, r] is W3[k, i, r].
    # W3 is (p, n, rank) = (3, 3, rank).
    # We need W3[k, i, r].
    # So T[i, j, k] = sum_r U3[i, j, r] V3[j, k, r] W3[k, i, r]
    
    # einsum: 'ijr,jkr,kr->ijk' is wrong because W3 index is k, i.
    # Let's swap W3 axes: W3_swapped[i, k, r] = W3[k, i, r]
    W3_swapped = W3.transpose(1, 0, 2) # (n, p, rank) -> (i, k, r)
    
    # Now T[i, j, k] = sum_r U3[i, j, r] V3[j, k, r] W3_swapped[i, k, r]
    # einsum: 'ijr,jkr,ikr->ijk'
    T = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3_swapped)
    
    return np.all(T == 1)
