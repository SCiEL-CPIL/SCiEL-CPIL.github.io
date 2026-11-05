<<<<<<< SEARCH
    # For each Strassen term l (0..6) and each schoolbook term (r,s,t) in 3x3:
    # col = l * 9 + r * 3 + s * 3 + t  ... let's just use a counter
    col = 0
    for l in range(7):
        for r in range(3):  # within-block row of A
            for s in range(3):  # within-block col of A / row of B
                for t in range(3):  # within-block col of B
                    # U index: i*6+j where i = 3*a+r, j = 3*b+s
                    for a in range(2):
                        for b in range(2):
                            coeff = alpha[l][a * 2 + b]
                            if coeff != 0:
                                u_idx = (3 * a + r) * 6 + (3 * b + s)
                                U[u_idx][col] += coeff
                    
                    # V index: j*6+k where j = 3*b+s, k = 3*c+t
                    for b in range(2):
                        for c in range(2):
                            coeff = beta[l][b * 2 + c]
                            if coeff != 0:
                                v_idx = (3 * b + s) * 6 + (3 * c + t)
                                V[v_idx][col] += coeff
                    
                    # W index: k*6+i where k = 3*c+t, i = 3*a+r
                    for a in range(2):
                        for c in range(2):
                            coeff = gamma[l][a * 2 + c]
                            if coeff != 0:
                                w_idx = (3 * c + t) * 6 + (3 * a + r)
                                W[w_idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
=======
    # Helper to get non-zero indices and values from a 4-element list (a*2+b)
    def get_nonzero(coeffs):
        nz = []
        for idx in range(4):
            if coeffs[idx] != 0:
                a = idx // 2
                b = idx % 2
                nz.append((a, b, coeffs[idx]))
        return nz

    for l in range(7):
        alpha_nz = get_nonzero(alpha[l])
        beta_nz = get_nonzero(beta[l])
        gamma_nz = get_nonzero(gamma[l])
        
        # Pre-compute compatible triples (a', b', c')
        # alpha has (a', b'), beta has (b', c')
        # They must share b'
        triples = []
        for a_prime, b_prime, alpha_val in alpha_nz:
            for b_prime_beta, c_prime, beta_val in beta_nz:
                if b_prime == b_prime_beta:
                    triples.append((a_prime, b_prime, c_prime, alpha_val, beta_val))
        
        for s in range(3):
            for triple_idx, (a_prime, b_prime, c_prime, alpha_val, beta_val) in enumerate(triples):
                col = l * 6 + s * 2 + triple_idx
                
                # Fill U
                # U[i, j] where i = 3*a_prime + r, j = 3*b_prime + s
                # Value is alpha_val
                for r in range(3):
                    u_idx = (3 * a_prime + r) * 6 + (3 * b_prime + s)
                    U[u_idx][col] = alpha_val
                
                # Fill V
                # V[j, k] where j = 3*b_prime + s, k = 3*c_prime + t
                # Value is beta_val
                for t in range(3):
                    v_idx = (3 * b_prime + s) * 6 + (3 * c_prime + t)
                    V[v_idx][col] = beta_val
                
                # Fill W
                # W[k, i] where k = 3*c + t, i = 3*a + r
                # Value is gamma_val for each non-zero gamma (a, c)
                for a_out, c_out, gamma_val in gamma_nz:
                    for r in range(3):
                        for t in range(3):
                            w_idx = (3 * c_out + t) * 6 + (3 * a_out + r)
                            W[w_idx][col] = gamma_val

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
