<<<<<<< SEARCH
        def growth_ode(a, y):
            D, Dp = y
            om = omega_m(a)[0]
            
            # Numerical derivative for H2
            h = 1e-5
            H2_val = H2(a)
            H2_p = (H2(a + h) - H2(a - h)) / (2.0 * h)
            H_over_H_prime = 0.5 * H2_p / H2_val
            
            coeff1 = 3.0 / a + H_over_H_prime
            coeff2 = 1.5 * om / (a * a)
            
            return [Dp, -coeff1 * Dp + coeff2 * D]
=======
        def growth_ode(a, y):
            D, Dp = y
            # Ensure om is a float
            om = float(omega_m(a))
            
            # Numerical derivative for H2
            h = 1e-5
            H2_val = float(H2(a))
            H2_p = (float(H2(a + h)) - float(H2(a - h))) / (2.0 * h)
            H_over_H_prime = 0.5 * H2_p / H2_val
            
            coeff1 = 3.0 / a + H_over_H_prime
            coeff2 = 1.5 * om / (a * a)
            
            return [Dp, -coeff1 * Dp + coeff2 * D]
>>>>>>> REPLACE

I also want to ensure `a_grid` is definitely a 1D float array to prevent any `t_eval` issues.

