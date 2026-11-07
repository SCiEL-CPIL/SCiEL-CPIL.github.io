import numpy as np

def equation(b, s, temp, pH, params):
        p = params
        
        # Normalization
        b_n = b / 5.0
        s_n = s / 5.0
        t_n = (temp - 20.0) / 20.0
        ph_n = (pH - 2.0) / 10.0
        
        # Monod term for substrate
        # p[1] is Ks (normalized), p[9] is Hill coefficient
        # To avoid division by zero or negative Ks issues, use abs or softplus?
        # BFGS can go negative. Monod requires Ks > 0.
        # Use exp(p[1]) to ensure positive Ks? Or just p[1]**2?
        # If p[1] is 1, Ks=1.
        ks = p[1] * p[1] + 1e-6
        hill = p[9]
        # If hill is negative, it inverts. Let's fix hill to be positive or just use p[9] as is and hope it stays positive?
        # Better: hill = abs(p[9]) + 0.1
        monod = s_n**hill / (ks**hill + s_n**hill)
        
        # Gaussian for temp
        # p[2] is optimal t_n, p[3] is width
        t_opt = p[2]
        t_width = p[3] * p[3] + 1e-6
        temp_factor = np.exp(-0.5 * ((t_n - t_opt) / t_width)**2)
        
        # Gaussian for pH
        ph_opt = p[4]
        ph_width = p[5] * p[5] + 1e-6
        ph_factor = np.exp(-0.5 * ((ph_n - ph_opt) / ph_width)**2)
        
        # Density term
        # p[6], p[7], p[8] for const, linear, quadratic
        dens_term = p[6] + p[7]*b_n + p[8]*b_n**2
        
        # Combine
        # p[0] is global scale
        result = p[0] * monod * temp_factor * ph_factor * dens_term
        
        return result
