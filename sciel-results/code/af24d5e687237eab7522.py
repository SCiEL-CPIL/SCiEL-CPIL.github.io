"""
Strategy: Analytical reduction of the 3-variable problem to a 1-variable problem
using the structure of the Hermitian matrices, followed by high-precision 1D optimization.

The objective is J(c) = sqrt(c^T X2 c * c^T P2 c) / (c^T c), where c12 is determined by c0, c4, c8 via g(0)=0.
The matrices X2 and P2 in the Hermite basis (degrees 0,4,8,12) have a specific structure.
By parameterizing the free variables and using the constraint, we can reduce the search space.
A key insight from the inspirations is that the problem might be better tackled by
exploiting the symmetry or by using a global optimizer like Differential Evolution
on a reduced parameter space, or by finding the exact minimum analytically if possible.

However, a more robust "different" strategy is to use **Differential Evolution** directly on the 3 free variables,
but with a much tighter bound on the search space and higher population, leveraging the fact that
the objective is smooth and the global minimum is likely not too far from the seed.
The current program uses Nelder-Mead which is local. Differential Evolution is global.

Let's refine the objective function calculation to be as fast and accurate as possible.
The matrices X2 and P2 for the subspace {0, 4, 8, 12} can be precomputed.
X2[i,j] = <psi_i, x^2 psi_j>.
P2[i,j] = <psi_i, -d^2/dx^2 psi_j> (since P = -i d/dx, P^2 = -d^2/dx^2).

Actually, the standard uncertainty principle is ||x f|| ||f'||.
||f||^2 = c^T c (since psi are orthonormal).
||x f||^2 = c^T X2 c.
||f'||^2 = c^T D2 c, where D2 = -d^2/dx^2.
Note that <psi_n, -d^2/dx^2 psi_m> = (n+1/2) delta_nm + off-diagonals.
Wait, the eigenvalues of -d^2/dx^2 for psi_n are (n+1/2)?
Let's check: H_0 = 1, psi_0 ~ e^{-x^2/2}. -d^2/dx^2 (e^{-x^2/2}) = (2x^2 - 1) e^{-x^2/2}.
This is not an eigenfunction.
The eigenfunctions of the harmonic oscillator Hamiltonian H = -1/2 d^2/dx^2 + 1/2 x^2 are psi_n with eigenvalues n+1/2.
So -d^2/dx^2 = H - 1/2 x^2 + 1/2? No.
H = P^2/2 + X^2/2. So P^2 = 2H - X^2.
Eigenvalues of P^2 on psi_n: 2(n+1/2) - <psi_n, X^2 psi_n>.
<psi_n, X^2 psi_n> = n + 1/2.
So eigenvalues of P^2 are n+1/2.
So P2 and X2 have the same diagonal entries n+1/2.
The off-diagonals are related by the commutation relations.
Actually, in the Hermite basis, X and P have similar tridiagonal (actually pentadiagonal for squares) structures.
X^2_{n, n+2} = sqrt((n+1)(n+2)/4).
P^2_{n, n+2} = sqrt((n+1)(n+2)/4) as well?
Let's check the sign.
x psi_n = sqrt(n/2) psi_{n-1} + sqrt((n+1)/2) psi_{n+1}.
d/dx psi_n = -sqrt(n/2) psi_{n-1} + sqrt((n+1)/2) psi_{n+1}.
So P = -i d/dx.
P^2 = -d^2/dx^2.
d^2/dx^2 psi_n = d/dx (-sqrt(n/2) psi_{n-1} + sqrt((n+1)/2) psi_{n+1})
= -sqrt(n/2) (-sqrt((n-1)/2) psi_{n-2} + sqrt(n/2) psi_n) + sqrt((n+1)/2) (-sqrt((n+1)/2) psi_n + sqrt((n+2)/2) psi_{n+2})
= sqrt(n(n-1)/4) psi_{n-2} - (n/2) psi_n - (n+1)/2 psi_n + sqrt((n+1)(n+2)/4) psi_{n+2}
= sqrt(n(n-1)/4) psi_{n-2} - (n+1/2) psi_n + sqrt((n+1)(n+2)/4) psi_{n+2}.
So -d^2/dx^2 psi_n = -sqrt(n(n-1)/4) psi_{n-2} + (n+1/2) psi_n - sqrt((n+1)(n+2)/4) psi_{n+2}.
So P2 has diagonal n+1/2 and off-diagonals with MINUS signs compared to X2?
X2_{n, n+2} = + sqrt((n+1)(n+2)/4).
P2_{n, n+2} = - sqrt((n+1)(n+2)/4).
This is a crucial difference! The previous program assumed they were the same.

Let's implement this corrected matrix structure and use Differential Evolution.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

DEADLINE_MARGIN = 5.0
MAX_RUN_TIME = 205.0

def construct():
    t0 = time.time()
    
    # Basis indices
    basis_idx = [0, 4, 8, 12]
    size = 13  # up to 12
    
    # Precompute full X2 and P2 matrices for n=0..12
    X2_full = np.zeros((size, size))
    P2_full = np.zeros((size, size))
    
    for n in range(size):
        diag = n + 0.5
        X2_full[n, n] = diag
        P2_full[n, n] = diag
        
        if n >= 2:
            coeff = np.sqrt(n * (n - 1) / 4.0)
            X2_full[n, n-2] = coeff
            X2_full[n-2, n] = coeff
            P2_full[n, n-2] = -coeff  # Sign difference!
            P2_full[n-2, n] = -coeff
            
        if n < size - 2:
            coeff = np.sqrt((n + 1) * (n + 2) / 4.0)
            X2_full[n, n+2] = coeff
            X2_full[n+2, n] = coeff
            P2_full[n, n+2] = -coeff
            P2_full[n+2, n] = -coeff

    # Extract submatrices
    X2 = np.zeros((4, 4))
    P2 = np.zeros((4, 4))
    for i, ni in enumerate(basis_idx):
        for j, nj in enumerate(basis_idx):
            X2[i, j] = X2_full[ni, nj]
            P2[i, j] = P2_full[ni, nj]
            
    # Compute psi_n(0)
    psi_vals = np.zeros(size)
    psi_vals[0] = np.pi ** (-0.25)
    for n in range(2, size, 2):
        psi_vals[n] = -np.sqrt((n-1)/n) * psi_vals[n-2]
        
    h0 = np.array([psi_vals[0], psi_vals[4], psi_vals[8], psi_vals[12]])
    
    def objective(c_free):
        c0, c4, c8 = c_free
        # c12 = -(c0*h0[0] + c4*h0[1] + c8*h0[2]) / h0[3]
        c12 = -(c0 * h0[0] + c4 * h0[1] + c8 * h0[2]) / h0[3]
        c = np.array([c0, c4, c8, c12])
        
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-15:
            return 1e15
            
        x2_val = np.dot(c, X2.dot(c))
        p2_val = np.dot(c, P2.dot(c))
        
        if x2_val < 0 or p2_val < 0:
            return 1e15 # Should not happen for positive definite matrices
            
        return np.sqrt(x2_val * p2_val) / norm_sq

    # Seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(c_seed)
    best_c = c_seed.copy()
    
    # Differential Evolution
    # Bounds: The coefficients are likely small. Let's try [-5, 5].
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    try:
        res = differential_evolution(objective, bounds, maxiter=1000, popsize=20, seed=42, tol=1e-12)
        if res.fun < best_score:
            best_score = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
        
    # Refine with Nelder-Mead
    try:
        from scipy.optimize import minimize
        res2 = minimize(objective, best_c, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res2.fun < best_score:
            best_score = res2.fun
            best_c = res2.x.copy()
    except Exception:
        pass
        
    return np.array(best_c)

if __name__ == "__main__":
    print(construct())
