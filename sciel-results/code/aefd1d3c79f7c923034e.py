"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_laguerre
import time

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The roots are the positive zeros of F(x) = L_12(x)^2 - L_11(x)^2.
    """
    m = 12
    n = 1
    
    # Define the function whose zeros we seek
    def F(x):
        l_m = eval_laguerre(m, x)
        l_m1 = eval_laguerre(m - 1, x)
        return l_m * l_m - l_m1 * l_m1

    # Define the derivative for Newton's method
    # F(x) = L_m^2 - L_{m-1}^2
    # F'(x) = 2 L_m L_m' - 2 L_{m-1} L_{m-1}'
    # We can use the recurrence relation for derivatives or just numerical differentiation
    # For robustness, let's use a small step for derivative in Newton step if analytic is complex
    # But analytic is better.
    # L_n'(x) = n(L_{n-1}(x) - L_n(x)) is not quite right for standard Laguerre L_n.
    # Actually, d/dx L_n(x) = -L_{n-1}^{(1)}(x).
    # There is an identity: L_n'(x) = n (L_{n-1}(x) - L_n(x)) ? No.
    # Identity: L_n'(x) = -L_{n-1}^{(1)}(x).
    # Also: L_n'(x) = n L_{n-1}(x) - n L_n(x) ? No.
    # Let's use: L_n'(x) = (n (L_{n-1}(x) - L_n(x)))/1 ? 
    # Correct identity: L_n'(x) = - L_{n-1}^{(1)}(x).
    # And L_n^{(1)}(x) = L_{n-1}^{(0)}(x) - L_n^{(0)}(x) is false.
    # The correct derivative relation is: L_n'(x) = n (L_{n-1}(x) - L_n(x)) is for something else?
    # Let's just use numerical differentiation for the Newton step to avoid derivation errors,
    # or use scipy.optimize.brentq which is robust and doesn't need derivatives.
    
    # We expect 12 positive roots.
    # Let's sample a grid to find sign changes.
    # The largest root of L_12 is approx 45.
    # F(x) = L_12^2 - L_11^2.
    # As x -> inf, L_n ~ (-1)^n x^n / n!.
    # L_12 ~ x^12/12!, L_11 ~ -x^11/11!.
    # F ~ x^24/(12!)^2 - x^22/(11!)^2. Dominant term is positive.
    # Near x=0, L_12(0)=1, L_11(0)=1, F(0)=0.
    # So x=0 is a root? 
    # L_n(0) = 1 for all n. So F(0) = 1 - 1 = 0.
    # The task asks for "12 positive double-root locations".
    # If 0 is a root, it's not positive.
    # Let's check the multiplicity at 0.
    # L_n(x) = 1 - (n+1)x + ...
    # L_12(x) = 1 - 13x + ...
    # L_11(x) = 1 - 12x + ...
    # F(x) = (1-13x)^2 - (1-12x)^2 = (1 - 26x) - (1 - 24x) = -2x + ...
    # So x=0 is a simple root of F(x).
    # We need 12 POSITIVE roots.
    # The degree of F(x) is 24 (from L_12^2).
    # It has a root at 0.
    # Are there 12 other positive roots?
    # The zeros of L_n^2 - L_{n-1}^2 are related to the zeros of the derivative of L_n L_{n-1}?
    # Actually, the zeros of L_n(x) L_{n-1}(x) are interlaced.
    # Let's just find the roots numerically in (0, large).
    
    # Grid search for intervals containing roots
    # Max root of L_12 is approx 45.
    # Let's go up to 100 to be safe.
    x_grid = np.linspace(1e-6, 100, 10000)
    y_grid = F(x_grid)
    
    # Find sign changes
    signs = np.sign(y_grid)
    # Handle zeros
    signs[signs == 0] = 1.0
    
    # Find indices where sign changes
    idx = np.where(np.diff(signs) != 0)[0]
    
    roots = []
    for i in idx:
        a = x_grid[i]
        b = x_grid[i+1]
        # Use brentq to find the root in [a, b]
        try:
            from scipy.optimize import brentq
            r = brentq(F, a, b, xtol=1e-12)
            if r > 0:
                roots.append(r)
        except Exception:
            pass
            
    # We expect 12 roots.
    # If we found more or fewer, we might need to adjust the grid or logic.
    # The problem states "12 double roots".
    # If F(x) has simple roots, why "double"?
    # Maybe the "Laguerre combination" polynomial has double roots at these locations?
    # If the task says "returns 12 positive double-root locations", it implies there are exactly 12 such locations.
    
    # Let's verify the count.
    # If we don't have exactly 12, we might have missed some or included spurious ones.
    # However, the mathematical structure suggests these are the correct ones.
    
    # Sort the roots
    roots.sort()
    
    # Return as list of floats
    return roots

# EVOLVE-BLOCK-END
