"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def objective_log(params):
        # params = [s0, m0, s4, m4, s8, m8]
        # coefficients: c_i = s_i * exp(m_i)
        s0, m0, s4, m4, s8, m8 = params
        c0 = s0 * np.exp(m0)
        c4 = s4 * np.exp(m4)
        c8 = s8 * np.exp(m8)
        
        # Enforce g(0) = 0 => c12 is determined
        # c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Construct g(x)
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
        
        # Compute norms
        # L2 norm
        norm2_sq = np.sum(g**2) * dx
        if norm2_sq < 1e-10:
            return 1e10
            
        # L4 norm
        norm4 = np.sum(np.abs(g)**4) * dx
        norm4_val = norm4 ** 0.25
        
        # Fourier Transform
        G = np.fft.fft(g) * dx
        
        # L2 norm of G
        norm2_hat_sq = np.sum(np.abs(G)**2) / (len(g) * dx)
        
        # L4 norm of G
        norm4_hat = np.sum(np.abs(G)**4) * (1 / (len(g) * dx))
        norm4_hat_val = norm4_hat ** 0.25
        
        ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
        
        return ratio

    # Initial guess from seed: [-113 / 100, 1 / 25, 1 / 3240]
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    
    # Convert to log-magnitude and sign representation
    # c = sign * exp(m) => sign = sign(c), m = log(|c|)
    s0_init = np.sign(c0_seed)
    m0_init = np.log(np.abs(c0_seed))
    s4_init = np.sign(c4_seed)
    m4_init = np.log(np.abs(c4_seed))
    s8_init = np.sign(c8_seed)
    m8_init = np.log(np.abs(c8_seed))
    
    x0 = np.array([s0_init, m0_init, s4_init, m4_init, s8_init, m8_init])
    
    # Use Nelder-Mead for robustness without gradient
    # We need to be careful with the sign variables. 
    # Standard Nelder-Mead works on continuous spaces. 
    # The sign variables are discrete (+1 or -1).
    # A better approach for this specific problem: 
    # The objective function depends on c0, c4, c8.
    # If we flip a sign, the value of c12 changes, and g changes.
    # Let's try a multi-start approach or just optimize over the continuous magnitudes and fix signs?
    # The directive says "Parameterize... using log-magnitude and sign representation".
    # This implies the optimizer should explore signs too.
    # However, Nelder-Mead on a mixed discrete/continuous space is tricky.
    # Let's try a different strategy: 
    # Minimize over [m0, m4, m8] for each of the 8 sign combinations?
    # That would be 8 calls to minimize.
    # Or, we can use a continuous approximation for signs? No, that breaks the math.
    
    # Let's try a different approach: 
    # The "sign" variables in the parameter vector can be optimized if we treat them as continuous
    # but penalize values not close to +/-1? Or just use a grid of signs?
    
    # Given the time limit (180s) and the complexity, let's try 8 minimizations.
    # This is robust and fits the "explore full non-linear parameter space" directive.
    
    best_c = x0[:3] # This is wrong, x0 is [s,m,s,m,s,m]. We need to return [c0, c4, c8]
    best_val = np.inf
    
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ])
    
    for s_combo in signs:
        # Initialize with magnitudes from seed
        m_init = np.array([m0_init, m4_init, m8_init])
        
        def objective_m(mags, s_c=s_combo):
            c0 = s_c[0] * np.exp(mags[0])
            c4 = s_c[1] * np.exp(mags[1])
            c8 = s_c[2] * np.exp(mags[2])
            
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            norm2_sq = np.sum(g**2) * dx
            if norm2_sq < 1e-10:
                return 1e10
                
            norm4 = np.sum(np.abs(g)**4) * dx
            norm4_val = norm4 ** 0.25
            
            G = np.fft.fft(g) * dx
            
            norm2_hat_sq = np.sum(np.abs(G)**2) / (len(g) * dx)
            norm4_hat = np.sum(np.abs(G)**4) * (1 / (len(g) * dx))
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio

        res = minimize(objective_m, m_init, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
        
        if res.fun < best_val:
            best_val = res.fun
            best_c = s_combo * np.exp(res.x)
            
    return best_c

# EVOLVE-BLOCK-END
