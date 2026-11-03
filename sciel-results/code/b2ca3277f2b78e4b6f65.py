"""
Improved program for the uncertainty inequality task.

The task is to minimize the upper bound on A(f)*A(Fourier(f)) by tuning three
coefficients (c0, c4, c8) of a Hermite-polynomial times Gaussian test function.
The coefficient c12 is fixed by the constraint g(0) = 0.

The seed program returns [-113/100, 1/25, 1/3240] which scores 1.0.
We aim to beat this by performing a more thorough optimization.

Key insights:
1. The objective function is a Rayleigh quotient involving X^2 and P^2 matrices
   in the Hermite basis.
2. We can use scipy.optimize to find a better local minimum.
3. We can also try a grid search or random restarts to avoid local minima.

Let's derive the objective function properly.

The test function is g(x) = sum_{k in {0,4,8,12}} c_k * psi_k(x), where psi_k
are the orthonormal Hermite functions.

The quantity to minimize is the upper bound on A(f)*A(Fourier(f)), which is
typically related to the uncertainty product. In the context of the task, it's
the value of the bound derived from the test function.

Given the structure, we assume the objective is:
  J(c) = sqrt( (c^T X^2 c) * (c^T P^2 c) ) / (c^T c)

where X^2 and P^2 are the matrices for the position squared and momentum squared
operators in the Hermite basis.

Let's implement this and optimize.
"""

import time
import numpy as np
from scipy.optimize import minimize

DEADLINE_MARGIN = 5.0
MAX_RUN_TIME = 205.0


def construct():
    """
    Returns a numpy array of shape (3,) with coefficients [c0, c4, c8].
    The coefficient of H12 is implicitly determined by the constraint g(0) = 0.
    """
    t0 = time.time()
    
    # Hermite function indices
    basis_idx = [0, 4, 8, 12]
    size = 13  # up to n=12
    
    # Build X^2 and P^2 matrices for n=0..12
    X2 = np.zeros((size, size))
    P2 = np.zeros((size, size))
    
    for n in range(size):
        diag_val = n + 0.5
        X2[n, n] = diag_val
        P2[n, n] = diag_val
        
        if n >= 2:
            coeff = np.sqrt(n * (n - 1) / 4.0)
            X2[n, n-2] = coeff
            X2[n-2, n] = coeff
            P2[n, n-2] = coeff
            P2[n-2, n] = coeff
            
        if n < size - 2:
            coeff = np.sqrt((n + 1) * (n + 2) / 4.0)
            X2[n, n+2] = coeff
            X2[n+2, n] = coeff
            P2[n, n+2] = coeff
            P2[n+2, n] = coeff

    # Extract submatrices for basis {0, 4, 8, 12}
    X2_sub = np.zeros((4, 4))
    P2_sub = np.zeros((4, 4))
    for i, ni in enumerate(basis_idx):
        for j, nj in enumerate(basis_idx):
            X2_sub[i, j] = X2[ni, nj]
            P2_sub[i, j] = P2[ni, nj]
    
    # Compute psi_n(0) for n=0,4,8,12
    psi_vals = np.zeros(size)
    psi_vals[0] = np.pi ** (-0.25)
    for n in range(2, size, 2):
        psi_vals[n] = -np.sqrt((n-1)/n) * psi_vals[n-2]
    
    h0_vec = np.array([psi_vals[0], psi_vals[4], psi_vals[8], psi_vals[12]])
    
    def get_c12(c0, c4, c8):
        return -(c0 * h0_vec[0] + c4 * h0_vec[1] + c8 * h0_vec[2]) / h0_vec[3]
    
    def objective(c_free):
        c0, c4, c8 = c_free
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-12:
            return np.inf
            
        x2_val = np.dot(c, np.dot(X2_sub, c))
        p2_val = np.dot(c, np.dot(P2_sub, c))
        
        x2_val = max(x2_val, 1e-12)
        p2_val = max(p2_val, 1e-12)
        
        return np.sqrt(x2_val * p2_val) / norm_sq
    
    # Seed value
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(c_seed)
    best_c = c_seed.copy()
    
    # Local optimization from seed
    try:
        res = minimize(objective, c_seed, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_score:
            best_score = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() - t0 > MAX_RUN_TIME:
            break
            
        # Generate random initial point
        if i == 0:
            c_try = c_seed + rng.normal(0, 0.05, 3)
        else:
            # Mix of local and global
            if i % 3 == 0:
                c_try = rng.uniform(-1, 1, 3)
            else:
                c_try = c_seed + rng.normal(0, 0.1, 3)
        
        val = objective(c_try)
        if val < best_score:
            best_score = val
            best_c = c_try.copy()
            
            # Refine
            try:
                res2 = minimize(objective, c_try, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_c = res2.x.copy()
            except Exception:
                pass
    
    return np.array(best_c)


if __name__ == "__main__":
    print(construct())
