"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Global deadline guard
_deadline = None

def integrate(f, t_span, y0):
    global _deadline
    
    if _deadline is None:
        _deadline = time.time() + 310.0  # Safety margin before 330s limit
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # The problem is a fixed 10x10 2D heat equation.
    # y' = A y, where A is the 5-point Laplacian matrix (scaled).
    # We can detect the structure with very few calls.
    # Alternatively, we can just assume it's linear and time-invariant and reconstruct A.
    # But reconstruction of a 100x100 matrix takes 100 calls.
    # Can we do better?
    # The heat equation matrix is sparse and has a specific structure.
    # However, we don't know the exact scaling or boundary conditions for sure.
    # But the task says "fixed 10x10 two-dimensional heat equation".
    # Usually this implies standard Dirichlet or Neumann boundaries and a standard stencil.
    
    # Strategy:
    # 1. Check if f is linear and time-invariant.
    # 2. If so, reconstruct A using N calls (100 calls).
    # 3. Compute expm(A * dt) and apply to y0.
    # Total calls: ~105.
    # Score: -log10(eps) - 0.5 * log10(105) ~ 15 - 2.5 = 12.5.
    # This is much better than the current score of 1.07.
    # Why is current score 1.07?
    # Current code reconstructs A in a loop of N calls.
    # Wait, the current code DOES this.
    # Let's check the current code's performance.
    # It uses `expm(A * dt)`.
    # Maybe `expm` is slow or inaccurate?
    # Or maybe the linearity check fails?
    # Or maybe the generic solver is used?
    
    # Let's optimize the reconstruction.
    # Instead of N calls, can we exploit sparsity?
    # If we assume it's a standard 5-point Laplacian, we only need to find the coefficients.
    # But assuming structure is risky if the "fixed" problem has different parameters.
    # However, 100 calls is very cheap. The main cost is `expm` on a 100x100 matrix.
    # `expm` for 100x100 is fast (< 0.1s).
    
    # Let's try to improve accuracy or reduce calls further.
    # Can we reduce calls?
    # If we use random vectors, we still need N independent vectors to span the space.
    # So N calls is the minimum for a black-box linear operator reconstruction.
    
    # Is there a way to avoid full reconstruction?
    # If we know it's a heat equation, we might guess A.
    # But if the guess is wrong, error is high.
    # Reconstructing is safer.
    
    # Let's verify the current implementation's issues.
    # The current code checks linearity with 2 random vectors.
    # Then it reconstructs A using N basis vectors.
    # This should work.
    
    # One potential issue: `expm` might accumulate error or be slow for larger N?
    # N=100 is small.
    
    # Let's try to improve the fallback or the detection.
    # What if the system is not exactly linear due to floating point noise in f?
    # The checks use `allclose` with tight tolerances.
    
    # Let's try a different approach:
    # Instead of `expm`, use eigen-decomposition?
    # `eigvals, eigvecs = np.linalg.eig(A)`
    # `expm = eigvecs @ diag(exp(eigvals * dt)) @ inv(eigvecs)`
    # This might be more stable or faster?
    # `expm` uses Padé approximation which is robust.
    
    # Let's look at the score formula again:
    # Score = -log10(relative_error) - 0.5 * log10(calls)
    # If relative error is 1e-15, -log10(1e-15) = 15.
    # If calls = 105, -0.5 * log10(105) ~ -2.5.
    # Score ~ 12.5.
    
    # Why is the current score 1.07?
    # 1.07 = -log10(err) - 0.5 * log10(calls)
    # If calls = 105, -0.5 * log10(105) ~ -2.5.
    # So -log10(err) must be ~ 3.57.
    # err ~ 10^-3.57 ~ 2.7e-4.
    # This suggests the error is NOT machine precision.
    # Why?
    # Maybe `expm` is not accurate enough? Unlikely.
    # Maybe the reconstruction of A is inaccurate?
    # Maybe the linearity check failed and it fell back to `_generic_solve`?
    # `_generic_solve` uses RK4 with 100 steps.
    # RK4 error is O(h^4).
    # h = 0.05 / 100 = 5e-4.
    # Error ~ (5e-4)^4 = 6.25e-15.
    # This should be very accurate.
    # Calls = 100 * 4 = 400.
    # Score = -log10(1e-15) - 0.5 * log10(400) ~ 15 - 2.8 = 12.2.
    
    # So why is the score 1.07?
    # Maybe the error is larger?
    # Or maybe the calls are higher?
    # Or maybe the reference solution is not what I think?
    
    # Let's check the constraints.
    # "Scored by -log10(relative terminal error) - 0.5*log10(calls to f)"
    # "Against a SciPy reference solution"
    
    # Perhaps the `expm` approach has a subtle bug?
    # `A[:, j] = f(t_probe1, e_j)`
    # This assumes f(t, y) returns a vector.
    # If f returns a scalar, this fails.
    # But heat equation is a system of ODEs.
    
    # Let's try to make the reconstruction more robust.
    # And maybe reduce calls by using a smarter basis?
    # No, N calls is optimal for black-box.
    
    # What if we use `scipy.sparse.linalg.expm_multiply`?
    # It computes `expm(A * dt) @ y0` without forming the full matrix exponential.
    # This is more efficient and potentially more accurate.
    # `from scipy.sparse.linalg import expm_multiply`
    # `y_end = expm_multiply(dt * A, y0)`
    # This avoids forming the 100x100 matrix E.
    
    # Let's try this.
    
    # Also, let's ensure we don't waste calls on failed probes.
    # We can do the reconstruction directly and check consistency.
    
    # New Plan:
    # 1. Reconstruct A using N calls.
    # 2. Check if A is consistent with y0 (i.e., A @ y0 == f(t0, y0)).
    # 3. If consistent, use `expm_multiply` to get y_end.
    # 4. If not, fallback to RK4.
    
    # This saves the initial probe calls (2-5 calls).
    # Total calls: N (reconstruction) + 1 (check) = N+1.
    # For N=100, 101 calls.
    
    # Let's implement this.
    
    # Check time invariance first?
    # If f(t, y) depends on t, A is not constant.
    # But heat equation is time-invariant.
    # We can check f(t0, y0) vs f(t1, y0).
    # If they differ, fallback.
    
    # Let's do:
    # 1. f_y0 = f(t0, y0)
    # 2. f_y0_t1 = f(t1, y0)
    # 3. If not allclose, fallback.
    # 4. Reconstruct A using N calls at t0.
    # 5. Check A @ y0 == f_y0.
    # 6. Use expm_multiply.
    
    # Total calls: 2 + N = 102.
    
    # Let's code this.
    
    f_y0 = np.array(f(t0, y0))
    f_y0_t1 = np.array(f(t1, y0))
    
    if not np.allclose(f_y0, f_y0_t1, rtol=1e-9, atol=1e-12):
        return _generic_solve(f, t_span, y0)
        
    # Reconstruct A
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Check consistency
    if not np.allclose(A @ y0, f_y0, rtol=1e-9, atol=1e-12):
        return _generic_solve(f, t_span, y0)
        
    # Use expm_multiply
    try:
        from scipy.sparse.linalg import expm_multiply
        y_end = expm_multiply(dt * A, y0)
    except Exception:
        # Fallback to dense expm
        try:
            E = expm(dt * A)
            y_end = E @ y0
        except Exception:
            return _generic_solve(f, t_span, y0)
            
    if not np.all(np.isfinite(y_end)):
        return _generic_solve(f, t_span, y0)
        
    return y_end


def _generic_solve(f, t_span, y0):
    """Fallback generic solver if structure detection fails."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Use RK4 with adaptive or fixed small step.
    steps = 100
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
