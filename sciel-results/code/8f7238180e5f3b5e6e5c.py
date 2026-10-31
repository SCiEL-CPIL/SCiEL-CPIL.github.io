"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration
    # Use a finer grid for better accuracy of L4 and FT
    x = np.linspace(-12, 12, 4000)
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
    
    def objective(params):
        # params: [s0, m0, s4, m4, s8, m8]
        # c_i = s_i * exp(m_i)
        try:
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
            # L2 norm squared
            norm2_sq = np.sum(g**2) * dx
            if norm2_sq < 1e-10:
                return 1e10
                
            # L4 norm
            norm4 = np.sum(np.abs(g)**4) * dx
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform
            # Use FFT for efficiency
            G = np.fft.fft(g) * dx
            # Frequency spacing dxi = 1 / (N * dx)
            # Note: The FFT sum approximates integral sum g(x) e^{-i 2 pi k x / L} dx
            # The scaling for continuous FT from DFT: 
            # F(omega) = sum g(x_j) e^{-i omega x_j} dx
            # With FFT, G_k = sum g(x_j) e^{-i 2 pi k j / N}
            # omega_k = 2 pi k / (N dx)
            # So G_k approximates F(omega_k) * N? No.
            # Let's stick to the standard discrete approximation:
            # sum |G_k|^2 dxi should equal sum |g_j|^2 dx
            # dxi = 1 / (N * dx)
            
            norm2_hat_sq = np.sum(np.abs(G)**2) * (1 / (len(g) * dx))
            norm4_hat = np.sum(np.abs(G)**4) * (1 / (len(g) * dx))
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10

    # Initial guess from seed: c0 = -113/100, c4 = 1/25, c8 = 1/3240
    c0_seed = -1.13
    c4_seed = 0.04
    c8_seed = 1.0 / 3240.0
    
    # Convert to log-magnitude and sign
    # s = sign(c), m = log(|c|)
    # Handle zero carefully, but seed coeffs are non-zero
    s0_seed = np.sign(c0_seed)
    m0_seed = np.log(np.abs(c0_seed))
    
    s4_seed = np.sign(c4_seed)
    m4_seed = np.log(np.abs(c4_seed))
    
    s8_seed = np.sign(c8_seed)
    m8_seed = np.log(np.abs(c8_seed))
    
    x0 = np.array([s0_seed, m0_seed, s4_seed, m4_seed, s8_seed, m8_seed])
    
    # Use Nelder-Mead for robustness without gradient
    # The log-space helps in exploring large variations in magnitude
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'adaptive': True})
    
    if res.success:
        s0, m0, s4, m4, s8, m8 = res.x
        return np.array([s0 * np.exp(m0), s4 * np.exp(m4), s8 * np.exp(m8)])
    else:
        # Fallback to seed if optimization fails or times out (though minimize handles timeout via maxiter usually)
        return np.array([c0_seed, c4_seed, c8_seed])

# EVOLVE-BLOCK-END
