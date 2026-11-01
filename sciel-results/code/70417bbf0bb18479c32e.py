"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Constants
N = 50  # Number of Hermite functions to use in the basis


def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight is fixed so that g(0) = 0.
    
    Strategy: Discretize the position and momentum variance operators
    on the Hermite-Gaussian basis and find the optimal coefficients
    that minimize the upper bound on A(f)*A(F).
    """
    start_time = time.time()
    deadline = 210.0 - 30.0  # 30 second safety margin
    
    # We work with a finite-dimensional subspace spanned by Hermite functions H_k for k = 0 to N-1
    # The test function is g(x) = sum_k a_k * H_k(x) * exp(-x^2/2)
    # where only even k are considered (H0, H4, H8, H12, ...)
    
    # For this problem, we focus on the specific structure where:
    # g(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # with a12 determined by g(0) = 0
    
    # Let's compute the variance operators in the Hermite basis
    
    # Position operator x in Hermite basis: <m|x|n> is non-zero only for |m-n|=1
    # <m|x|n> = sqrt((n+1)/2) if m=n+1, sqrt(n/2) if m=n-1
    
    # Momentum operator p = -i*d/dx, similar structure
    
    # For variance: <x^2> and <p^2>
    # <m|x^2|n> has diagonal and off-diagonal terms
    
    # Diagonal: <n|x^2|n> = n + 0.5
    # Off-diagonal: <m|x^2|n> non-zero for |m-n|=2
    
    # Let's build the matrices for x^2 and p^2 in the even subspace
    # We'll use the first N even Hermite functions
    
    # Actually, let's take a more direct approach.
    # The key insight is that for Fourier-invariant functions, we can compute
    # A(f) = sqrt(<x^2>) and A(F) = sqrt(<p^2>) (up to constants)
    
    # In the Hermite basis, for a state psi = sum_k c_k |k>,
    # <x^2> = sum_k c_k^2 (k + 0.5) + off-diagonal terms
    # <p^2> = sum_k c_k^2 (k + 0.5) + off-diagonal terms (similar structure)
    
    # Due to the symmetry, <x^2> = <p^2> for real Hermite functions
    
    # The product A(f)*A(F) = <x^2> (when they're equal)
    
    # So we want to minimize <x^2> subject to normalization and g(0)=0
    
    # Let's set up the optimization
    
    # The coefficients are a0, a4, a8, with a12 = -(a0 + a4/24 + a8/3024) / (1/362880)
    # Wait, H_n(0) = 0 for odd n, and H_{2k}(0) = (-1)^k * (2k)! / k!
    
    # H0(0) = 1
    # H4(0) = -3 (since 4! / 2! = 12, (-1)^2 = 1, so H4(0) = -3? Let me recalculate)
    # Actually, H_n(0) = 0 if n odd, H_{2k}(0) = (-1)^k * (2k)! / k!
    # H0(0) = 1
    # H4(0) = (-1)^2 * 4! / 2! = 24/2 = 12... but with the sign convention
    # Let's use the probabilists' Hermite polynomials or physicists'?
    
    # For physicists' Hermite: H0=1, H1=2x, H2=4x^2-2, H3=8x^3-12x, H4=16x^4-48x^2+12
    # So H4(0) = 12
    # H8(0) = ? Let me compute: H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H8(0) = 1680
    
    # Wait, I need to be more careful. Let me use scipy to get these values.
    
    from scipy.special import eval_hermite
    
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = ?
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a12 = -(a0 + 12*a4 + 1680*a8) / H12(0)
    
    # Now, for the variance calculation:
    # In the Hermite basis, <x^2> for psi = sum_k a_k * H_k(x) * exp(-x^2/2) / sqrt(2^k k! sqrt(pi))
    # The normalized Hermite functions are phi_k(x) = (1/sqrt(2^k k! sqrt(pi))) * H_k(x) * exp(-x^2/2)
    
    # <phi_m | x^2 | phi_n> = sqrt((n+1)/2)*sqrt((n+2)/2)*delta_{m,n+2} + (n+0.5)*delta_{m,n} + sqrt(n/2)*sqrt((n-1)/2)*delta_{m,n-2}
    
    # For our case, we only have even k: k = 0, 4, 8, 12
    
    # Let's define the state in terms of the normalized basis
    # c_k = a_k / sqrt(2^k k! * sqrt(pi)) * sqrt(sqrt(pi)) = a_k / sqrt(2^k k!)
    # Wait, the normalization: integral |H_k(x) exp(-x^2/2)|^2 dx = 2^k k! sqrt(pi)
    # So the normalized function is phi_k = H_k(x) exp(-x^2/2) / sqrt(2^k k! sqrt(pi))
    # The coefficient in the normalized basis is c_k = a_k * sqrt(sqrt(pi)) / sqrt(2^k k!)
    # But since we're computing ratios, the sqrt(sqrt(pi)) factor cancels
    
    # Let's just work with the unnormalized coefficients and compute <x^2> properly
    
    def compute_x2(a0, a4, a8):
        """Compute <x^2> for the given coefficients."""
        if time.time() > deadline:
            return None
        
        # Compute a12 from the constraint g(0) = 0
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        # The state is psi(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
        
        # <x^2> = integral x^2 * psi(x)^2 dx / integral psi(x)^2 dx
        
        # In the Hermite basis, the matrix elements of x^2 are:
        # <k|x^2|m> = sqrt((m+1)/2)*sqrt((m+2)/2)*delta_{k,m+2} + (m+0.5)*delta_{k,m} + sqrt(m/2)*sqrt((m-1)/2)*delta_{k,m-2}
        
        # For k, m in {0, 4, 8, 12}:
        # Only diagonal terms and terms with |k-m|=2 contribute
        
        # Since we only have even k with gaps of 4, the only non-zero off-diagonal terms
        # would be if we had consecutive even k's, but we don't (0, 4, 8, 12 are spaced by 4)
        
        # So <x^2> = sum_k c_k^2 * (k + 0.5) / sum_k c_k^2
        
        # where c_k = a_k / sqrt(2^k k!) (up to a common factor)
        
        def norm_factor(k):
            return 2.0 ** k * float(np.math.factorial(k))
        
        c0_sq = a0 ** 2 / norm_factor(0)
        c4_sq = a4 ** 2 / norm_factor(4)
        c8_sq = a8 ** 2 / norm_factor(8)
        c12_sq = a12 ** 2 / norm_factor(12)
        
        total_norm = c0_sq + c4_sq + c8_sq + c12_sq
        
        if total_norm <= 0:
            return None
        
        x2 = (c0_sq * 0.5 + c4_sq * 4.5 + c8_sq * 8.5 + c12_sq * 12.5) / total_norm
        
        return x2
    
    # We want to minimize <x^2> (which equals A(f)*A(F) up to constants)
    
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        
        a0, a4, a8 = coeffs
        x2 = compute_x2(a0, a4, a8)
        if x2 is None:
            return 1e10
        return x2
    
    # Initial guess: the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple starting points to avoid local minima
    best_result = None
    best_obj = float('inf')
    
    # Starting points to try
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-1.5, 0.15, 0.0015]),
        np.array([-0.8, 0.08, 0.0008]),
        np.array([-1.2, 0.12, 0.0012]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fall back to the initial guess
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    return best_result
# EVOLVE-BLOCK-END
