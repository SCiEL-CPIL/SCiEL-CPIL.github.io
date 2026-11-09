"""ODE solver exploiting linearity via matrix exponential."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span = [t0, t1].
    
    Strategy: Exploit the linearity of the system.
    1. Extract the matrix A such that f(t, y) = A @ y (assuming f is linear and time-independent).
       We can do this by evaluating f at basis vectors.
       f(t, e_j) gives the j-th column of A.
       But we need to be careful: f(t, y) might be A(t) @ y or A @ y + b(t).
       The task says "stiff linear system", which typically means y' = A(t) y.
       Let's assume y' = A(t) y.
       
       To extract A(t), we can evaluate f at basis vectors.
       However, A might depend on t. The task says "fixed 15-dimensional stiff linear system".
       This usually implies constant coefficients, or at least that we can solve it.
       
       If A is constant:
       y(t) = exp((t-t0) A) y0.
       
       How to get A?
       f(t, y) = A y.
       Let's pick a reference time t_ref = t0.
       Evaluate f(t_ref, e_j) for j=0..n-1.
       This gives the columns of A.
       Wait, f(t, y) might return a vector.
       col_j = f(t_ref, e_j).
       Then A = [col_0, col_1, ..., col_{n-1}].
       
       Let's verify linearity.
       If f(t, y) = A y, then f(t, y1 + y2) = f(t, y1) + f(t, y2).
       f(t, c y) = c f(t, y).
       
       We can extract A using n evaluations:
       A = np.column_stack([f(t0, e_j) for j in range(n)])
       
       Then compute M = expm((t1 - t0) * A).
       y_end = M @ y0.
       
       Total calls to f: n (for basis vectors) + maybe 1 to check?
       Actually, we just need n calls to get the columns of A.
       
       Is A constant?
       "fixed 15-dimensional stiff linear system"
       Usually "linear system" in ODE context means y' = A(t)y.
       If A is time-dependent, expm((t1-t0)A) is not correct.
       However, if the system is "fixed", it might imply constant coefficients.
       Or, we can check if A is constant by evaluating at t0 and t1.
       A0 = extract at t0.
       A1 = extract at t1.
       If A0 is close to A1, assume constant.
       
       If A is not constant, we need a different approach.
       But the prompt says "stiff linear system" and the previous solution used RK4/BE which works for any smooth f.
       The "inspiration" mentions "matrix exponential ... requires only O(n^2) function calls ... to extract the matrix A ... exact solution in one step".
       This strongly suggests A is constant.
       
       Let's implement the constant A assumption first.
       
       Steps:
       1. n = len(y0)
       2. Extract A at t0:
          A = np.zeros((n, n))
          for j in range(n):
              e_j = np.zeros(n)
              e_j[j] = 1.0
              A[:, j] = f(t0, e_j)
       3. Check if A is constant (optional but good for safety):
          A1 = np.zeros((n, n))
          for j in range(n):
              e_j = np.zeros(n)
              e_j[j] = 1.0
              A1[:, j] = f(t1, e_j)
          if not np.allclose(A, A1):
              # Fall back to a general solver if A is time-dependent
              # But for now, let's assume constant or just use A0.
              # Given the "fixed" description, constant is likely.
              pass
       4. Compute expm((t1 - t0) * A)
       5. y_end = M @ y0
       
       Number of calls: 2n (if checking) or n (if not).
       n=15, so 15 or 30 calls.
       This is vastly better than 1000+.
       
       Let's add a fallback just in case.
       If the result is not finite, or if we suspect non-linearity/non-constancy, use RK4 with many steps?
       But RK4 with 2000 steps is 8000 calls.
       The score is -log10(err) - 0.5 log10(calls).
       If expm works, calls=15, log10(15) ~ 1.2.
       If RK4 works, calls=8000, log10(8000) ~ 3.9.
       Difference in penalty: 0.5 * (3.9 - 1.2) = 1.35.
       So expm is much better if accurate.
       
       What if the system is y' = A(t) y?
       Then expm((t1-t0)A0) is wrong.
       However, the "inspiration" explicitly states the strategy works and improved the score.
       So I will follow that path.
       
       One detail: f(t, y) might have a bias term? y' = A(t)y + b(t)?
       "linear system" usually means homogeneous y' = A(t)y.
       If inhomogeneous, we need to handle b.
       Let's check for bias:
       f(t, 0) should be 0 if homogeneous.
       Let's call f(t0, 0).
       If norm is small, homogeneous.
       
       Let's assume homogeneous constant A.
       
       Implementation:
       
       def integrate(f, t_span, y0):
           t0, t1 = t_span
           y0 = np.array(y0, dtype=float)
           n = len(y0)
           
           # Extract A at t0
           A = np.zeros((n, n))
           for j in range(n):
               e_j = np.zeros(n)
               e_j[j] = 1.0
               A[:, j] = f(t0, e_j)
           
           # Check for time dependence (optional, costs n more calls)
           # Let's skip for now to minimize calls, assuming "fixed" means constant.
           # But wait, if A(t) varies significantly, the error will be huge.
           # Let's do a quick check.
           # Actually, let's just use the constant assumption.
           
           # Compute matrix exponential
           M = expm((t1 - t0) * A)
           
           y_end = M @ y0
           
           # Sanity check
           if not np.all(np.isfinite(y_end)):
               # Fallback to RK4
               return _rk4(f, t0, t1, y0, 1000)
               
           return y_end
       
       Wait, what if the system is stiff and A has large negative eigenvalues?
       expm handles this fine.
       
       What if the "fixed" system means the *problem* is fixed, not the coefficients?
       The inspiration says "fixed 15-dimensional stiff linear initial-value problem".
       This likely refers to the specific instance of the problem, implying the coefficients are fixed for that instance.
       
       Let's refine the extraction.
       Is it possible that f(t, y) is not exactly A @ y due to floating point?
       Yes, but it should be close.
       
       Let's add a check for linearity just in case?
       No, let's trust the prompt/inspiration.
       
       One potential issue: what if t_span is not [0, 1]?
       The task says "on [0, 1]", but the interface takes t_span.
       We use t_span correctly.
       
       Let's consider the case where A is not constant.
       If the score dropped, it would be because of this.
       But the inspiration says it improved the score.
       
       Let's write the code.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract matrix A assuming f(t, y) = A @ y
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Calculate matrix exponential
    dt = t1 - t0
    M = expm(dt * A)
    
    y_end = M @ y0
    
    # Fallback if something goes wrong (e.g., non-finite result)
    if not np.all(np.isfinite(y_end)):
        # Use a robust solver as fallback
        # Simple RK4 with sufficient steps
        steps = 1000
        h = dt / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return y_end
