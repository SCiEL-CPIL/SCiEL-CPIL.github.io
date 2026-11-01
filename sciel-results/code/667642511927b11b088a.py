import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors
    # U: (i, val), V: (offset, val), W: (offset, val)
    # Offsets for V: 0->B00, 1->B01, 8->B10, 9->B11
    # Offsets for W: 0->C00, 1->C01, 2->C10, 3->C11
    
    factors = [
        # M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (9, 1)], 'W': [(0, 1), (3, 1)]},
        # M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # M3
        {'U': [(0, 1)], 'V': [(1, 1), (9, -1)], 'W': [(1, 1), (3, 1)]},
        # M4
        {'U': [(3, 1)], 'V': [(8, 1), (0, -1)], 'W': [(0, 1), (2, 1)]},
        # M5
        {'U': [(0, 1), (1, 1)], 'V': [(9, 1)], 'W': [(0, -1), (1, 1)]},
        # M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7
        {'U': [(1, 1), (2, -1)], 'V': [(8, 1), (9, 1)], 'W': [(0, 1)]},
    ]
    
    for j in range(4):
        for s in range(7):
            r = j * 7 + s
            f = factors[s]
            for u_idx, u_val in f['U']:
                U[u_idx, r] = u_val
            for v_off, v_val in f['V']:
                if v_off == 0: v_row = 2 * j
                elif v_off == 1: v_row = 2 * j + 1
                elif v_off == 8: v_row = 8 + 2 * j
                elif v_off == 9: v_row = 8 + 2 * j + 1
                else: continue
                V[v_row, r] = v_val
            for w_off, w_val in f['W']:
                if w_off == 0: w_row = 4 * j
                elif w_off == 1: w_row = 4 * j + 2
                elif w_off == 2: w_row = 4 * j + 1
                elif w_off == 3: w_row = 4 * j + 3
                else: continue
                W[w_row, r] = w_val
                
    # Verify
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Should be 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # 0 otherwise
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], np.dot(V[b], W[c])) # This is wrong, need sum over r
                val = np.sum(U[:, r] * V[:, r] * W[:, r] for r in range(rank)) # Still slow
                # Better:
                # val = np.einsum('r,r,r->', U[:, r], V[:, r], W[:, r])? No.
                # val = np.sum(U * V * W) is not right.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # This is element-wise product of columns, then sum.
                # col_u = U[:, r]
                # col_v = V[:, r]
                # col_w = W[:, r]
                # prod = col_u * col_v * col_w
                # sum(prod)
                
    # Efficient verification:
    # Build the tensor T from U,V,W
    # T = sum_r U[:,r][:,None,:] * V[:,r][None,:,None] * W[:,r][None,None,:]
    # This is 4x16x16 = 1024 elements.
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        
    # Check support
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return {"U": [[1 if i2*m+j2 == a else 0 for r in range(n*m*p)] for i2 in range(n)]} # Fallback?
                    # Actually, just return schoolbook
                    rank = n*m*p
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m+j2, col] = 1
                                V[j2*p+k2, col] = 1
                                W[k2*n+i2, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (i == i2 and j == j2 and k == k2):
                    if abs(T[a, b, c]) > 1e-9:
                        # Fallback
                        rank = n*m*p
                        U = np.zeros((n*m, rank))
                        V = np.zeros((m*p, rank))
                        W = np.zeros((p*n, rank))
                        col = 0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    U[i2*m+j2, col] = 1
                                    V[j2*p+k2, col] = 1
                                    W[k2*n+i2, col] = 1
                                    col += 1
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
