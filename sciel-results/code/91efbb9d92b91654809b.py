"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is fixed to impose g(0) = 0.
    
    We use the eigenfunction property of the Fourier transform on Hermite functions:
    The Fourier transform of H_n(x) * exp(-x^2/2) is (-i)^n * H_n(x) * exp(-x^2/2)
    (up to normalization conventions).
    
    The uncertainty product bound involves moments of |g|^2 and |G|^2 in position and frequency space.
    Since the Fourier transform maps Hermite functions to themselves (up to phase), we can compute
    all necessary quantities analytically as polynomial functions of the coefficients.
    
    For the standard convention where F[f](k) = (1/sqrt(2*pi)) * int f(x) exp(-ikx) dx,
    the Hermite functions psi_n(x) = (2^n n! sqrt(pi))^(-1/2) H_n(x) exp(-x^2/2) satisfy
    F[psi_n] = (-i)^n psi_n.
    
    For g(x) = sum c_n H_n(x) exp(-x^2/2), we have:
    |g(x)|^2 = (sum c_n H_n(x))^2 * exp(-x^2)
    |G(k)|^2 = (sum c_n H_n(k))^2 * exp(-k^2)  [since |(-i)^n|^2 = 1]
    
    The uncertainty product A(f)*A(F(f)) involves:
    A(f) = sqrt(int x^2 |f(x)|^2 dx / int |f(x)|^2 dx)
    A(F(f)) = sqrt(int k^2 |F(f)(k)|^2 dk / int |F(f)(k)|^2 dk)
    
    Due to the symmetry, both integrals have the same form. Let's define:
    N = int (sum c_n H_n(x))^2 exp(-x^2) dx
    M = int x^2 (sum c_n H_n(x))^2 exp(-x^2) dx
    
    Then A(f)*A(F(f)) = M/N (since both position and frequency give the same ratio)
    
    Actually, let me be more careful. The product is:
    A(f) * A(F(f)) = sqrt(M/N) * sqrt(M/N) = M/N
    
    where N = int |g|^2 dx and M = int x^2 |g|^2 dx.
    
    We need to compute N and M as functions of c0, c4, c8 (with c12 determined by g(0)=0).
    
    g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    H0(0) = 1
    H4(0) = -12
    H8(0) = 1680
    H12(0) = -103950
    
    So g(0) = c0 - 12*c4 + 1680*c8 - 103950*c12 = 0
    => c12 = (c0 - 12*c4 + 1680*c8) / 103950
    
    Now we need to compute:
    N = int (sum c_n H_n(x))^2 exp(-x^2) dx
    M = int x^2 (sum c_n H_n(x))^2 exp(-x^2) dx
    
    Using the orthogonality of Hermite polynomials with weight exp(-x^2):
    int H_m(x) H_n(x) exp(-x^2) dx = 2^n n! sqrt(pi) * delta_{mn}
    
    So N = sum_{n,m} c_n c_m int H_n H_m exp(-x^2) dx = sum_n c_n^2 * 2^n n! * sqrt(pi)
    
    For M, we use the recurrence relation: x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)
    So x^2 H_n(x) = (1/4) H_{n+2}(x) + (n + 1/2) H_n(x) + (n(n-1)/4) H_{n-2}(x)
    
    Therefore:
    int x^2 H_n(x) H_m(x) exp(-x^2) dx = 
      (1/4) int H_{n+2} H_m exp(-x^2) dx + (n+1/2) int H_n H_m exp(-x^2) dx + (n(n-1)/4) int H_{n-2} H_m exp(-x^2) dx
    = (1/4) * 2^{n+2} (n+2)! sqrt(pi) delta_{m,n+2} + (n+1/2) * 2^n n! sqrt(pi) delta_{mn} + (n(n-1)/4) * 2^{n-2} (n-2)! sqrt(pi) delta_{m,n-2}
    
    So M = sum_n c_n^2 * [ (1/4) * 2^{n+2} (n+2)! sqrt(pi) + (n+1/2) * 2^n n! sqrt(pi) + (n(n-1)/4) * 2^{n-2} (n-2)! sqrt(pi) ]
    (only the diagonal terms survive due to orthogonality, and we sum over n=m)
    
    Wait, that's not quite right. Let me redo:
    M = sum_{n,m} c_n c_m int x^2 H_n H_m exp(-x^2) dx
    
    The integral is non-zero only when m = n, m = n+2, or m = n-2.
    Due to symmetry, the off-diagonal terms (m = n+2 and m = n-2) will cancel or combine.
    
    Actually, let's just compute it directly:
    int x^2 H_n(x) H_m(x) exp(-x^2) dx is non-zero only if m = n, n+2, or n-2.
    
    For m = n: coefficient is (n+1/2) * 2^n n! sqrt(pi)
    For m = n+2: coefficient is (1/4) * 2^{n+2} (n+2)! sqrt(pi) = 2^n (n+2)! sqrt(pi)
    For m = n-2: coefficient is (n(n-1)/4) * 2^{n-2} (n-2)! sqrt(pi) = n(n-1) * 2^{n-4} (n-2)! sqrt(pi)
    
    But wait, when we sum over all n,m, the term with m=n+2 in the n-th row is the same as the term with m=n-2 in the (n+2)-th row.
    
    Let me just compute M numerically for given coefficients using the formula.
    """
    
    start_time = time.time()
    deadline = start_time + 180  # Safety margin: finish well before 210s
    
    # Hermite polynomial values at 0
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -103950.0
    
    def c12_from_others(c0, c4, c8):
        """Compute c12 such that g(0) = 0"""
        return (c0 + c4 * H4_0 + c8 * H8_0) / (-H12_0)
    
    def compute_bound(c0, c4, c8):
        """
        Compute the uncertainty product bound M/N for given coefficients.
        """
        c12 = c12_from_others(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        degrees = np.array([0, 4, 8, 12])
        
        # Compute N = sum_n c_n^2 * 2^n * n! * sqrt(pi)
        sqrt_pi = np.sqrt(np.pi)
        N = 0.0
        M = 0.0
        
        for i, (c, n) in enumerate(zip(coeffs, degrees)):
            if c == 0:
                continue
            # Diagonal term for N
            norm_n = 2.0**n * float(np.math.factorial(n)) * sqrt_pi
            N += c**2 * norm_n
            
            # For M, diagonal term
            diag_M = (n + 0.5) * norm_n
            M += c**2 * diag_M
        
        # Off-diagonal terms for M
        # For pairs (n, m) where m = n+2 or m = n-2
        for i, (ci, ni) in enumerate(zip(coeffs, degrees)):
            if ci == 0:
                continue
            for j, (cj, nj) in enumerate(zip(coeffs, degrees)):
                if cj == 0 or i >= j:
                    continue
                # Only consider if |ni - nj| == 2
                if abs(ni - nj) == 2:
                    # Compute int x^2 H_ni H_nj exp(-x^2) dx
                    # This is non-zero only when |ni - nj| = 2
                    n_min = min(ni, nj)
                    n_max = max(ni, nj)
                    # The integral equals:
                    # If ni = nj + 2: (1/4) * 2^{nj+2} (nj+2)! sqrt(pi) = 2^{nj} (nj+2)! sqrt(pi)
                    # If nj = ni + 2: (ni(ni-1)/4) * 2^{ni-2} (ni-2)! sqrt(pi)
                    # Both should give the same value by symmetry
                    if ni > nj:
                        # ni = nj + 2
                        integral = 2.0**nj * float(np.math.factorial(nj + 2)) * sqrt_pi
                    else:
                        # nj = ni + 2
                        integral = ni * (ni - 1) * 2.0**(ni - 4) * float(np.math.factorial(ni - 2)) * sqrt_pi
                    
                    M += 2.0 * ci * cj * integral
        
        if N <= 0:
            return np.inf
        
        return M / N
    
    # Initial guess from prior work
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    x0 = np.array([c0_init, c4_init, c8_init])
    
    def objective(x):
        return compute_bound(x[0], x[1], x[2])
    
    # Try differential evolution first for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    best_result = None
    best_score = np.inf
    
    # Quick local optimization from initial point
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        if best_result is not None:
            return best_result
        return x0
    
    # Try differential evolution for global search
    try:
        de_result = differential_evolution(objective, bounds, seed=42, 
                                            maxiter=100, tol=1e-8, 
                                            popsize=10, polish=True)
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_result is not None:
            return best_result
        return x0
    
    # Try multiple random restarts
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        x_rand = np.random.uniform(-5, 5, 3)
        try:
            res = minimize(objective, x_rand, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    return x0

# EVOLVE-BLOCK-END
