    for k in range(1, degree + 1):
        N_next = np.zeros_like(N_curr)
        for i in range(n_basis):
            # Term 1
            denom1 = knots[i+k] - knots[i]
            if denom1 > 1e-12:
                coeff1 = (x - knots[i]) / denom1
            else:
                coeff1 = 0.0
            val1 = coeff1 * N_curr[:, i]
            
            # Term 2
            if i + 1 < n_basis:
                denom2 = knots[i+k+1] - knots[i+1]
                if denom2 > 1e-12:
                    coeff2 = (knots[i+k+1] - x) / denom2
                else:
                    coeff2 = 0.0
                val2 = coeff2 * N_curr[:, i+1]
            else:
                val2 = 0.0
                
            N_next[:, i] = val1 + val2
        N_curr = N_next
