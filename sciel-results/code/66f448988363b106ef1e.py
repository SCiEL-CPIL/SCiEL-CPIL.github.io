import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalization to help optimizer
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Monod term for substrate
    # p[1] is Ks (normalized), p[9] is Hill coefficient
    # Use square to ensure positive Ks
    ks = p[1] * p[1] + 1e-6
    hill = p[9]
    # Ensure hill is positive for standard Monod-like behavior, but allow flexibility
    # If hill is negative, it inverts. Let's fix hill to be positive or just use p[9] as is?
    # Better: hill = abs(p[9]) + 0.1
    # Actually, standard Monod is hill=1. Let's just use p[9] directly but ensure it doesn't cause issues.
    # If hill is very large, s_n**hill can overflow. Let's clip hill or use a safer form.
    # For now, let's stick to the current approach but maybe clamp hill to [0.5, 5] to be safe?
    # Or just use hill = 1.0 + 0.5*p[9] to keep it near 1?
    # Let's try hill = abs(p[9]) + 0.1 as suggested in comments
    hill_safe = abs(p[9]) + 0.1
    
    # Avoid overflow in s_n**hill_safe if hill is large
    # np.power can overflow. Let's use a safe power or just assume hill is reasonable.
    # Given BFGS starts at 1, hill_safe starts at 1.1.
    monod = s_n**hill_safe / (ks**hill_safe + s_n**hill_safe)
    
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
    # Let's try a simpler density inhibition term like in inspiration 2?
    # density_term = 1.0 - (p[6] * b_n) / (p[7]*p[7] + 1e-6 + b_n)
    # But the quadratic form is more flexible. Let's keep the quadratic but ensure it doesn't go too crazy.
    dens_term = p[6] + p[7]*b_n + p[8]*b_n**2
    
    # Combine
    # p[0] is global scale
    result = p[0] * monod * temp_factor * ph_factor * dens_term
    
    return result
