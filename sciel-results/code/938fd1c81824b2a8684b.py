import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 30.0 # seconds
    
    # Parameters
    n_roots = 12
    alpha = -0.5
    
    # Precompute nothing specific, use scipy on the fly
    
    def score_float(z):
        # z: array of 12 floats
        # Check validity
        if np.any(z <= 0):
            return 1e10
        if np.any(np.diff(np.sort(z)) < 1e-6):
            return 1e10
            
        # Build system A c = b
        # 26 equations
        # Variables c_0 ... c_25
        
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = eval_genlaguerre(2*k, alpha, 0.0)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        # g'(x) = - sum c_k L_{2k-1}^{(alpha+1)}(x)
        # For k=0, term is 0.
        for k in range(1, 26):
            A[1, k] = -eval_genlaguerre(2*k-1, alpha+1, 0.0)
        b[1] = 1.0
        
        # Rows for z_i
        for i in range(n_roots):
            zi = z[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            for k in range(26):
                A[row_g, k] = eval_genlaguerre(2*k, alpha, zi)
            b[row_g] = 0.0
            
            for k in range(1, 26):
                A[row_gp, k] = -eval_genlaguerre(2*k-1, alpha+1, zi)
            b[row_gp] = 0.0
        
        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Check residual
        if np.linalg.norm(A @ c - b) > 1e-8:
            return 1e10
            
        # Construct g(x) in monomial basis
        # L_n^(alpha)(x) = sum_{j=0}^n binom(n+alpha, n-j) (-x)^j / j!
        # We need coefficients of g(x) = sum c_k L_{2k}^(alpha)(x)
        
        # Max degree is 50
        g_coeffs = np.zeros(51)
        
        for k in range(26):
            n = 2*k
            if c[k] == 0: continue
            # Get coeffs of L_n^(alpha)
            # binom(n+alpha, n-j)
            # Using math.comb or scipy.special.comb? 
            # n+alpha is not integer. Use gamma function or recurrence.
            # L_n^(alpha)(x) = sum_{j=0}^n (-1)^j / j! * binom(n+alpha, n-j) x^j
            # Let's compute the polynomial coefficients for each Laguerre poly
            
            # Efficient way: use recurrence or just evaluate at points and interpolate?
            # Interpolation is easier and robust for degree 50.
            # Evaluate g(x) at 51 points, interpolate to get coeffs.
            pass
            
        # Alternative: Build the polynomial directly using numpy.polynomial
        # This is cleaner.
        
        # Let's stick to interpolation for simplicity and robustness in float domain.
        # Points: 0..50
        x_pts = np.linspace(0, 10, 51) # Range might need to be larger if roots are large
        # Actually, just use the roots and 0? No, need enough points to determine degree 50 poly.
        # 51 points are enough.
        
        # But we need to divide by x * prod(x-zi)^2.
        # It is better to get the monomial coefficients.
        
        # Let's compute monomial coefficients of L_n^(alpha)(x).
        # L_n^(alpha)(x) = sum_{j=0}^n a_{n,j} x^j
        # a_{n,j} = (-1)^j / j! * binom(n+alpha, n-j)
        
        # We can precompute these for n=0, 2, ..., 50.
        
        # To avoid recomputing in score_float, move this to outer scope?
        # Yes.
        
        # ... (Precomputation logic) ...
        
        # For now, let's assume we have a function to get g_coeffs.
        
        # Division:
        # d(x) = x * prod (x-zi)^2
        # d_coeffs = [1]
        # d_coeffs = np.convolve(d_coeffs, [0, 1]) # x
        # for zi in z:
        #     d_coeffs = np.convolve(d_coeffs, [1, -2*zi, zi**2])
        
        # q_coeffs = np.polydiv(g_coeffs, d_coeffs)
        
        # Roots of q_coeffs
        # q_roots = np.roots(q_coeffs)
        # real_roots = q_roots[np.abs(q_roots.imag) < 1e-9].real
        # max_real = np.max(real_roots) if len(real_roots)>0 else 0
        # return max(0, max_real)

    # Since I can't easily fit all precomputation in the thought block, I will write the full code in the final block.
    # Key is to make it fast.
    
    # Optimization
    best_z = np.arange(1, 13, dtype=float)
    best_score = score_float(best_z)
    
    # Try a few random starts
    for i in range(5):
        if time.time() - start_time > deadline:
            break
        z0 = np.random.uniform(0.5, 15, n_roots)
        z0.sort()
        # Simple hill climbing or Nelder-Mead
        # Nelder-Mead
        res = minimize(score_float, z0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_z = res.x
            
    return best_z.tolist()
