import numpy as np
from scipy.optimize import differential_evolution

def construct():
    """
    Construct three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 coefficient is fixed to satisfy g(0) = 0 (P(0)=0).
    Uses Differential Evolution to minimize the upper bound on C4 (A(f)*A(F) product).
    """
    # Orthogonal polynomial properties:
    # <H_m, H_n> = integral(H_m * H_n * exp(-x^2) dx) = sqrt(pi) * 2^m * m! * delta_{m,n}
    
    # Recurrence for x^2 H_n(x) to compute integral of x^2 H_m H_n:
    # x H_n = 0.5 * H_{n+1} + n * H_{n-1}
    # x^2 H_n = x(0.5 H_{n+1} + n H_{n-1})
    #         = 0.5(0.5 H_{n+2} + (n+1) H_n) + n(0.5 H_n + (n-1) H_{n-2})
    #         = 0.25 H_{n+2} + 0.5(n+1) H_n + 0.5n H_n + 0.5n(n-1) H_{n-2}
    #         = 0.25 H_{n+2} + (n+0.5) H_n + 0.5n(n-1) H_{n-2}
    
    # Let D_{n,k} be the coefficient of H_k in x^2 H_n(x).
    # Then integral(x^2 H_m H_n) = sum_k D_{n,k} * <H_m, H_k>
    # Since we need symmetry for the matrix A_{mn} = integral(x^2 H_m H_n), we can precompute.
    
    # We only use even degrees: 0, 4, 8, 12.
    # Let's create a dictionary of coefficients for x^2 H_n for n in {0, 2, 4, 6, 8, 10, 12}
    # Actually, we just need the matrix M where M_{ij} = integral(x^2 H_{deg_i} H_{deg_j}) exp(-x^2)
    # and N where N_{ij} = integral(H_{deg_i} H_{deg_j}) exp(-x^2)
    
    degs = [0, 4, 8, 12]
    
    # Precompute factorials
    from math import factorial
    fact = {i: factorial(i) for i in range(15)}
    sqrt_pi = np.sqrt(np.pi)
    
    # Inner product matrix N (Gram matrix for the 4 polynomials)
    # N[i,j] = sqrt(pi) * 2^{degs[i]} * fact[degs[i]] if degs[i]==degs[j] else 0
    N = np.zeros((4,4))
    for i, d in enumerate(degs):
        N[i,i] = sqrt_pi * (2 ** d) * fact[d]
        
    # Matrix M: integral(x^2 H_a H_b) exp(-x^2)
    # M[i,j] = sum_k c_k * sqrt(pi) * 2^{degs[j]} * fact[degs[j]] * delta_{degs[i], k}
    # where x^2 H_{degs[i]} = sum_k c_k H_k
    
    # Coefficients for x^2 H_n:
    # H_0: 0.5 H_0
    # H_2: 0.25 H_4 + 2.5 H_2
    # H_4: 0.25 H_6 + 4.5 H_4 + 6 H_2
    # H_6: 0.25 H_8 + 6.5 H_6 + 15 H_4
    # H_8: 0.25 H_10 + 8.5 H_8 + 28 H_6
    # H_10: 0.25 H_12 + 10.5 H_10 + 45 H_8
    # H_12: 0.25 H_14 + 12.5 H_12 + 66 H_10
    
    coeffs_x2H = {
        0: {0: 0.5},
        4: {2: 6.0, 4: 4.5, 6: 0.25},
        8: {4: 15.0, 6: 28.0, 8: 8.5, 10: 0.25},
        12: {8: 45.0, 10: 66.0, 12: 12.5, 14: 0.25}
    }
    
    # Map degree to index in degs
    deg_to_idx = {d:i for i,d in enumerate(degs)}
    
    M = np.zeros((4,4))
    for i, d_i in enumerate(degs):
        for k, c in coeffs_x2H.get(d_i, {}).items():
            if k in deg_to_idx:
                j = deg_to_idx[k]
                # Integral x^2 H_{d_i} H_{d_j} = c * <H_k, H_{d_j}> = c * N[j,j] (since k must equal d_j for non-zero inner product? No, k is a degree in the expansion, so we sum over k. If k == d_j, inner product is N[j,j]. If k != d_j, inner product is 0)
                if k == d_j:
                    pass # handled below
                
    # Wait, the inner product <H_k, H_{d_j}> is non-zero only if k == d_j.
    # So M[i,j] = sum_{k} c_k (from expansion of x^2 H_{d_i}) * <H_k, H_{d_j}>
    # <H_k, H_{d_j}> = N[j,j] if k == d_j else 0.
    # So M[i,j] = c_{d_j} * N[j,j], where c_{d_j} is the coefficient of H_{d_j} in the expansion of x^2 H_{d_i}.
    
    M = np.zeros((4,4))
    for i, d_i in enumerate(degs):
        for j, d_j in enumerate(degs):
            c_coeff = coeffs_x2H.get(d_i, {}).get(d_j, 0.0)
            M[i,j] = c_coeff * N[j,j]
            
    # Objective function to minimize:
    # minimize (c^T M c) / (c^T N c)
    # subject to c^T h = 0, where h is the vector of H_k(0) values.
    # H_0(0) = 1
    # H_4(0) = -6
    # H_8(0) = 105
    # H_12(0) = -945
    h = np.array([1.0, -6.0, 105.0, -945.0])
    
    # We can solve the constraint for c_12 in terms of c_0, c_4, c_8.
    # c_0(1) + c_4(-6) + c_8(105) + c_12(-945) = 0
    # c_12 = (c_0 - 6 c_4 + 105 c_8) / 945
    
    def objective(vars_):
        c0, c4, c8 = vars_
        c12 = (c0 - 6*c4 + 105*c8) / 945.0
        c = np.array([c0, c4, c8, c12])
        
        num = c @ M @ c
        den = c @ N @ c
        
        if den <= 0:
            return 1e10
            
        # The constant C4 involves a factor. The task says "score is the upper bound on C4".
        # Usually this bound is proportional to (Integral x^2 f^2)/(Integral f^2) * (Integral |x|^2 |hat f|^2)/(Integral |hat f|^2).
        # By Plancherel, the Fourier side is the same if we normalize the Gaussian appropriately?
        # The seed returns [-113/100, 1/25, 1/3240].
        # Let's check the value returned by the seed's implicit ratio to calibrate the scale.
        c_seed = np.array([-113/100, 1/25, 1/3240])
        c12_seed = (c_seed[0] - 6*c_seed[1] + 105*c_seed[2]) / 945.0
        c_full_seed = np.array([c_seed[0], c_seed[1], c_seed[2], c12_seed])
        num_seed = c_full_seed @ M @ c_full_seed
        den_seed = c_full_seed @ N @ c_full_seed
        ratio_seed = num_seed / den_seed
        
        # The task asks for the score to be comparable to seed (1.0).
        # If the objective is just the ratio, and the seed corresponds to 1.0,
        # then minimizing the ratio gives a score < 1.0?
        # "score is the upper bound on C4... evaluator measures the raw value (lower is better)."
        # "search score is seed/value". This implies if my value is lower than seed's value, score > 1.
        # So I should return the raw value (the bound).
        # What is the raw value? It's the ratio I calculated, possibly multiplied by a constant.
        # Since I don't know the constant, and the interface expects `construct()` to return
        # the three coefficients, the *caller* (evaluator) computes the score from the coefficients.
        # Therefore, my local objective should just be the mathematically optimal configuration
        # for the ratio. The evaluator will calculate the actual C4 bound from my coefficients.
        
        return num / den

    # Define bounds for optimization
    # c0, c4, c8 should be bounded to keep coefficients reasonable.
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    # Use Differential Evolution for robustness
    result = differential_evolution(objective, bounds, seed=42, maxiter=1000, tol=1e-12, popsize=15)
    
    c_opt = result.x
    return np.array([c_opt[0], c_opt[1], c_opt[2]])

# Test to ensure it works quickly
if __name__ == "__main__":
    import time
    start = time.time()
    coeffs = construct()
    print(f"Optimized coefficients: {coeffs}")
    print(f"Time taken: {time.time() - start:.2f}s")
