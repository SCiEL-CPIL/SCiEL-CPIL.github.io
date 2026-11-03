import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

# Hermite polynomials coefficients in monomial basis
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 2100x^8 - 22680x^6 + 113400x^4 - 226800x^2 + 10395

# Coefficients for H_n in basis [1, x, x^2, ..., x^n]
# We only need even powers for the integrals
H_coeffs = {
    0: np.array([1.0]), # H_0
    4: np.array([3.0, 0.0, -6.0, 0.0, 1.0]), # H_4
    8: np.array([105.0, 0.0, -420.0, 0.0, 210.0, 0.0, -28.0, 0.0, 1.0]), # H_8
    12: np.array([10395.0, 0.0, -226800.0, 0.0, 113400.0, 0.0, -22680.0, 0.0, 2100.0, 0.0, -84.0, 0.0, 1.0]) # H_12
}

def get_poly_coeffs(c0, c4, c8):
    """
    Returns the coefficient vector of P(x) in the monomial basis [1, x, ..., x^12].
    P(x) = c0*H_0 + c4*H_4 + c8*H_8 + c12*H_12
    Constraint: P(0) = 0 => c0*1 + c4*3 + c8*105 + c12*10395 = 0
    => c12 = -(c0 + 3c4 + 105c8) / 10395
    """
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    # Sum the polynomials
    # Max degree is 12, so vector length 13
    coeffs = np.zeros(13)
    
    # H_0
    coeffs[0] += c0 * H_coeffs[0][0]
    
    # H_4
    coeffs[0] += c4 * H_coeffs[4][0]
    coeffs[2] += c4 * H_coeffs[4][2]
    coeffs[4] += c4 * H_coeffs[4][4]
    
    # H_8
    coeffs[0] += c8 * H_coeffs[8][0]
    coeffs[2] += c8 * H_coeffs[8][2]
    coeffs[4] += c8 * H_coeffs[8][4]
    coeffs[6] += c8 * H_coeffs[8][6]
    coeffs[8] += c8 * H_coeffs[8][8]
    
    # H_12
    coeffs[0] += c12 * H_coeffs[12][0]
    coeffs[2] += c12 * H_coeffs[12][2]
    coeffs[4] += c12 * H_coeffs[12][4]
    coeffs[6] += c12 * H_coeffs[12][6]
    coeffs[8] += c12 * H_coeffs[12][8]
    coeffs[10] += c12 * H_coeffs[12][10]
    coeffs[12] += c12 * H_coeffs[12][12]
    
    return coeffs

def compute_integrals(coeffs):
    """
    Computes I_2 = int P(x)^2 e^{-x^2} dx and I_4 = int P(x)^4 e^{-2x^2} dx.
    coeffs is the monomial coefficient vector of P(x).
    """
    # P(x)^2
    # We can use convolution
    p2_coeffs = np.convolve(coeffs, coeffs)
    # p2_coeffs[k] is coeff of x^k in P^2
    # Only even k contribute to the integral with e^{-x^2}
    
    # Integral of x^(2m) e^{-x^2} dx = (2m-1)!! / 2^m * sqrt(pi)
    # (2m-1)!! = (2m)! / (2^m m!)
    
    I_2 = 0.0
    for k in range(0, len(p2_coeffs), 2):
        m = k // 2
        if m == 0:
            val = np.sqrt(np.pi)
        else:
            # (2m-1)!! / 2^m
            double_fact = 1
            for i in range(1, 2*m, 2):
                double_fact *= i
            val = (double_fact / (2**m)) * np.sqrt(np.pi)
        I_2 += p2_coeffs[k] * val
        
    # P(x)^4
    p4_coeffs = np.convolve(p2_coeffs, p2_coeffs)
    
    # Integral of x^(2m) e^{-2x^2} dx
    # Formula: (2m-1)!! / (2^m 2^{0.5}) * sqrt(pi/2) ?
    # int x^(2m) e^{-ax^2} dx = (2m-1)!! / (2^m a^{m+0.5}) * sqrt(pi)
    # Here a=2.
    # val = (2m-1)!! / (2^m 2^{m+0.5}) * sqrt(pi) = (2m-1)!! / 2^{2m+0.5} * sqrt(pi)
    
    I_4 = 0.0
    for k in range(0, len(p4_coeffs), 2):
        m = k // 2
        if m == 0:
            val = np.sqrt(np.pi / 2.0)
        else:
            double_fact = 1
            for i in range(1, 2*m, 2):
                double_fact *= i
            val = (double_fact / (2**(2*m + 0.5))) * np.sqrt(np.pi)
        I_4 += p4_coeffs[k] * val
        
    return I_2, I_4

def objective(params):
    c0, c4, c8 = params
    coeffs = get_poly_coeffs(c0, c4, c8)
    I_2, I_4 = compute_integrals(coeffs)
    
    if I_2 <= 0:
        return 1e100
        
    # Minimize F = I_4^2 / I_2^4
    return (I_4 ** 2) / (I_2 ** 4)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    best_score = float('inf')
    best_params = np.array([-113 / 100, 1 / 25, 1 / 3240]) # Seed
    
    # Initial check
    best_score = objective(best_params)
    
    # Random restarts
    np.random.seed(42)
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
            
        # Random initialization in a reasonable range
        # The seed has c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003
        # Let's try random directions
        r0 = np.random.uniform(-2, 2)
        r4 = np.random.uniform(-0.5, 0.5)
        r8 = np.random.uniform(-0.01, 0.01)
        
        x0 = np.array([r0, r4, r8])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    # Return the coefficients
    return best_params

if __name__ == "__main__":
    print(construct())
