import numpy as np
from scipy.special import eval_genlaguerre, gammaln, roots_genlaguerre
from scipy.optimize import differential_evolution, minimize
from numpy.polynomial import polynomial as P
import time

# Precompute monomial coefficients for L_{2k}^{(-1/2)}(x) for k=0..25
# L_n^{(alpha)}(x) = sum_{j=0}^n binom(n+alpha, n-j) * (-x)^j / j!
# Coeff of x^j is binom(n+alpha, n-j) * (-1)^j / j!
# binom(n+alpha, n-j) = Gamma(n+alpha+1) / (Gamma(n-j+1) Gamma(j+alpha+1))

def get_laguerre_monomial_coeffs(n, alpha):
    coeffs = np.zeros(n + 1)
    for j in range(n + 1):
        log_val = gammaln(n + alpha + 1) - gammaln(n - j + 1) - gammaln(j + alpha + 1)
        val = np.exp(log_val)
        coeff = val * ((-1) ** j) / np.math.factorial(j)
        coeffs[j] = coeff
    return coeffs

alpha = -0.5
L_coeffs = []
for k in range(26):
    n = 2 * k
    L_coeffs.append(get_laguerre_monomial_coeffs(n, alpha))
L_coeffs = np.array(L_coeffs) # Shape (26, 51)

def score_float(z):
    z = np.array(z)
    # Check distinctness and positivity
    if np.any(z <= 0):
        return 1e6
    sorted_z = np.sort(z)
    if np.any(np.diff(sorted_z) < 1e-5):
        return 1e6
        
    # Build Matrix A (26x26)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    b[1] = 1.0 # g'(0) = 1
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = eval_genlaguerre(2*k, alpha, 0.0)
        
    # Row 1: g'(0) = 1
    for k in range(26):
        if 2*k == 0:
            deriv = 0.0
        else:
            deriv = -eval_genlaguerre(2*k - 1, alpha + 1, 0.0)
        A[1, k] = deriv
        
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0
    for i in range(12):
        zi = z[i]
        for k in range(26):
            A[2*i, k] = eval_genlaguerre(2*k, alpha, zi)
            if 2*k == 0:
                deriv = 0.0
            else:
                deriv = -eval_genlaguerre(2*k - 1, alpha + 1, zi)
            A[2*i+1, k] = deriv
            
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e6
        
    # Construct g(x) in monomial basis
    g_monomial = np.zeros(51)
    for k in range(26):
        g_monomial[:2*k+1] += c[k] * L_coeffs[k]
        
    # Construct divisor D(x) = x * prod (x - z_i)^2
    D = np.array([0.0, 1.0]) 
    for i in range(12):
        quad = np.array([z[i]**2, -2*z[i], 1.0])
        D = np.convolve(D, quad)
        
    # Polynomial division g / D
    g_high = g_monomial[::-1]
    D_high = D[::-1]
    
    q_high, rem_high = P.polydiv(g_high, D_high)
    
    # Find roots of q
    roots = np.roots(q_high)
    # Filter for real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    # Ensure they are positive? The task says "largest counted r".
    # "A real root r ... counts if ...".
    # It doesn't say r must be positive. But B = max(0, largest counted r).
    # So if all real roots are negative, B=0.
    
    if len(real_roots) == 0:
        return 0.0
        
    B = np.max(real_roots)
    if B < 0:
        return 0.0
    return B

def construct():
    start_time = time.time()
    deadline = 600 - 60 # Safety margin
    
    # Initial guess: Zeros of L_12^{(-1/2)}(x)
    x_init, _ = roots_genlaguerre(12, alpha)
    
    # Define bounds
    bounds = [(1e-3, 100.0)] * 12
    
    # Use differential evolution
    # We want to minimize score_float
    # DE is stochastic, so we run it for a while
    best_z = x_init
    best_score = score_float(x_init)
    
    # Run DE
    result = differential_evolution(score_float, bounds, maxiter=100, popsize=15, tol=1e-5, seed=42, polish=False)
    if result.fun < best_score:
        best_z = result.x
        best_score = result.fun
        
    # Refine with local optimization
    res = minimize(score_float, best_z, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
    if res.fun < best_score:
        best_z = res.x
        best_score = res.fun
        
    # Ensure distinct and positive
    best_z = np.sort(best_z)
    
    # If time permits, try a few random restarts
    while time.time() < deadline:
        # Random start
        z_rand = np.random.uniform(1e-3, 50.0, 12)
        z_rand = np.sort(z_rand)
        # Local optimization
        res_rand = minimize(score_float, z_rand, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res_rand.fun < best_score:
            best_z = res_rand.x
            best_score = res_rand.fun
        # Check time
        if time.time() > deadline - 10:
            break
            
    return [float(x) for x in best_z]
