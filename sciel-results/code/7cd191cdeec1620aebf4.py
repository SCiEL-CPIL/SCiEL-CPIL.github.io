"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Hermite Polynomials coefficients for H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 1260x^8 - 5544x^6 + 55440x^4 - 120120x^2 + 10395
    
    # Coefficients for even powers of x: [c_0, c_1, c_2, c_3, c_4, c_5, c_6] corresponding to x^0, x^2, ..., x^12
    H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    H8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
    H12 = np.array([10395.0, -120120.0, 55440.0, -5544.0, 1260.0, -84.0, 1.0])
    
    # Precompute integrals for I_2 and I_4
    # I_2 term: int x^(2k) e^(-x^2) dx = (2k-1)!! / 2^k * sqrt(pi)
    # I_4 term: int x^(2k) e^(-2x^2) dx = (2k-1)!! / 2^k * sqrt(pi/2)
    
    sqrt_pi = np.sqrt(np.pi)
    sqrt_pi_over_2 = np.sqrt(np.pi / 2.0)
    
    # Max degree of P(x)^4 is 48, so max power of x is 48, which is 2*24.
    # Max degree of P(x)^2 is 24, so max power of x is 24, which is 2*12.
    
    # Precompute factorials for (2k-1)!!
    # (2k-1)!! = (2k)! / (2^k k!)
    
    def double_factorial_2k_minus_1(k):
        if k == 0:
            return 1.0
        # Use log gamma for stability if k is large, but k is small here (max 24)
        res = 1.0
        for i in range(1, k+1):
            res *= (2*i - 1)
        return res

    int_coeffs_I2 = np.zeros(13) # up to k=12
    int_coeffs_I4 = np.zeros(25) # up to k=24
    
    for k in range(13):
        int_coeffs_I2[k] = double_factorial_2k_minus_1(k) / (2.0 ** k) * sqrt_pi
        
    for k in range(25):
        int_coeffs_I4[k] = double_factorial_2k_minus_1(k) / (2.0 ** k) * sqrt_pi_over_2

    def objective(c):
        c0, c4, c8 = c
        # Calculate c12
        # P(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        
        # Construct P(x) coefficients
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        
        # Calculate I_2
        # P(x)^2 coefficients
        # P has non-zero coeffs at indices 0..6
        # P^2 will have coeffs at 0..12
        P_sq = np.convolve(P, P)
        # P_sq[k] corresponds to x^(2k)
        I2 = np.dot(P_sq, int_coeffs_I2)
        
        if I2 <= 0:
            return 1e100
            
        # Calculate I_4
        # P(x)^4 coefficients
        P_4 = np.convolve(P_sq, P_sq)
        # P_4[k] corresponds to x^(2k)
        I4 = np.dot(P_4, int_coeffs_I4)
        
        # Normalize by I2^2
        # We want to minimize ||g||_4^4 / ||g||_2^4 which is I4 / I2^2
        return I4 / (I2 * I2)

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0
    best_val = objective(best_x)
    
    # Optimization
    try:
        from scipy.optimize import minimize
        
        # Use Nelder-Mead as it doesn't require gradients and is robust
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if res.success:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        
        # Try L-BFGS-B as well
        res2 = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 10000})
        if res2.success:
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception as e:
        # Fallback to grid search or just return seed if optimization fails
        pass

    # Check time
    if time.time() > deadline - 5.0:
         return best_x

    return best_x
# EVOLVE-BLOCK-END
