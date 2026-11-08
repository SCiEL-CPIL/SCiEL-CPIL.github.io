import numpy as np
from scipy.optimize import minimize

def construct():
    # Seed coefficients
    c_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    def compute_c12(c0, c4, c8):
        # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
        return -(c0 + 3*c4 + 105*c8) / 10395.0

    def get_g(x, c0, c4, c8):
        c12 = compute_c12(c0, c4, c8)
        # Hermite polynomials (physicists' or probabilists'? 
        # Standard Hermite in Fourier analysis contexts usually refers to Physicists' H_n 
        # where H_0=1, H_1=2x, H_2=4x^2-2, etc.
        # Let's verify with the seed values.
        # If Probabilists' He_n: He_0=1, He_1=x, He_2=x^2-1, He_4=x^4-6x^2+3.
        # The seed uses H_4. In the context of Gaussian e^{-x^2/2}, Physicists' Hermite H_n is orthogonal.
        # H_0(x) = 1
        # H_1(x) = 2x
        # H_2(x) = 4x^2 - 2
        # H_3(x) = 8x^3 - 12x
        # H_4(x) = 16x^4 - 48x^2 + 12
        # H_8(x) = 256x^8 - 3072x^6 + 8448x^4 - 4608x^2 + 384
        # H_12(x) = ...
        
        # Let's check the seed's c4 = 1/25 = 0.04.
        # If we use Probabilists' He_n (orthogonal with e^{-x^2/2}):
        # He_0 = 1
        # He_4 = x^4 - 6x^2 + 3
        # He_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # He_12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 10395
        
        # Let's check g(0) for seed with Probabilists':
        # c0=-1.13, c4=0.04, c8=1/3240 ~ 0.000308
        # c12 = -( -1.13 + 3(0.04) + 105(0.000308) ) / 10395
        # 3*0.04 = 0.12
        # 105*0.000308 = 0.03234
        # Sum = -1.13 + 0.12 + 0.03234 = -0.97766
        # c12 = 0.97766 / 10395 ~ 0.000094
        
        # The prompt says "Hermite-polynomial times Gaussian". 
        # Usually, this implies the standard Hermite functions $h_n(x) = H_n(x) e^{-x^2/2}$.
        # The coefficients multiply $H_0, H_4, H_8$.
        # I will use Probabilists' Hermite polynomials $He_n(x)$ because they are more common in statistical contexts and often denoted just H in some physics papers when combined with $e^{-x^2/2}$. 
        # However, Physicists' $H_n$ are also very common. 
        # Let's look at the magnitude. 
        # If Physicists': H_4(0) = 12. 
        # If Probabilists': He_4(0) = 3.
        # The seed c4 is 0.04. 
        # If we assume the "Prior state of the art" is correct, we should just optimize around it.
        # The definition of the polynomial affects the value of the function, but as long as we are consistent, the optimizer will find a local min.
        # I will use Probabilists' Hermite polynomials (He_n) as they are standard for the weight $e^{-x^2/2}$.
        
        H0 = 1
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 10395
        
        poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return np.exp(-x**2 / 2) * poly

    def objective(c):
        c0, c4, c8 = c
        # Evaluate max |g(x)| on a grid
        x = np.linspace(-10, 10, 2000)
        g_vals = get_g(x, c0, c4, c8)
        # We want to minimize the "upper bound". 
        # A common bound is the max of the function.
        # Another is the max of the Fourier transform.
        # Let's try minimizing the max of |g(x)|.
        return np.max(np.abs(g_vals))

    # Initial guess is the seed
    res = minimize(objective, c_seed, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.success:
        return res.x
    else:
        return c_seed
