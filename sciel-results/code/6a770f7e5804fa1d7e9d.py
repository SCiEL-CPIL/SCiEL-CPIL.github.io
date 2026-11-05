"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time
from scipy.optimize import minimize
import sys

# EVOLVE-BLOCK-START

def _value(h):
    """Compute the evaluator's value: abs(2 * N * max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    # Full autoconvolution
    ac = np.convolve(h, h, mode='full')
    peak = np.max(ac)
    return abs(2 * N * peak / (s ** 2))

def _qp_objective(h):
    """
    Objective for QP: Minimize ||h*h||_2^2 subject to sum(h)=1.
    This is equivalent to minimizing sum(ac^2).
    Since the autoconvolution is symmetric, we can compute it efficiently.
    """
    # Use FFT for faster autoconvolution calculation if needed, 
    # but for N=400, direct is fine and less prone to numerical noise in gradients.
    # However, for SLSQP gradient, we need analytic or numerical gradient.
    # Let's use numerical gradient via finite differences or provide analytic?
    # Providing analytic gradient for sum(ac^2) is complex. 
    # SLSQP can use numerical gradient, which is slow for 400 vars but acceptable if maxiter is low.
    # Alternatively, we can use the fact that min ||h*h||_2^2 is a convex QP.
    
    # Let's stick to the objective value for now and let SLSQP handle gradients.
    # To speed up, we can restrict the search space or use a smarter initialization.
    
    ac = np.convolve(h, h, mode='full')
    return np.sum(ac * ac)

def _qp_grad(h):
    """
    Analytic gradient of sum((h*h)^2) with respect to h.
    Let c = h*h. We want grad sum(c_i^2) = 2 * sum(c_i * dc_i/dh).
    c_k = sum_j h_j h_{k-j}.
    dc_k/dh_m = h_{k-m} + h_{m-k} (if indices valid).
    This is essentially the derivative of the quadratic form h^T A h.
    Actually, sum((h*h)^2) is a quartic form.
    Let's use the property: min ||h*h||_2^2 is equivalent to min h^T (C^T C) h ? No.
    
    Let's use a simpler approach: 
    The problem min ||h*h||_2^2 s.t. sum(h)=1 is a convex optimization problem.
    We can solve it using projected gradient or SLSQP.
    Given the time limit, SLSQP with numerical gradients might be too slow for 400 variables.
    
    Alternative: 
    We know that the optimal h for min ||h*h||_2^2 is often related to the eigenvector of the matrix representing the convolution operator.
    Specifically, if we view the autoconvolution as a linear operator L(h) = h*h, we are minimizing ||L(h)||_2^2.
    
    Let's try a different structural change: 
    Use the fact that we want to minimize the peak of the autocorrelation.
    The QP formulation min sum(ac^2) is a good proxy.
    
    To make it faster, we can reduce the dimensionality by assuming symmetry h[i] = h[N-1-i].
    This reduces variables to 200.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Initial guess: uniform distribution (seed)
    h_best = np.ones(N) / N  # Normalize sum to 1 to simplify constraint
    # The value is scale invariant, so sum(h)=1 is a valid normalization.
    val_best = _value(h_best)
    
    # Strategy: Solve the QP min sum((h*h)^2) s.t. sum(h)=1
    # We will use SLSQP. To handle the 400 variables, we will use symmetry.
    # h is symmetric: h[i] = h[N-1-i].
    # Variables: x[0..199] where h[0:200] = x, h[200:400] = x[::-1].
    
    def unpack(x):
        h = np.zeros(N)
        h[:200] = x
        h[200:] = x[::-1]
        return h

    def objective_sym(x):
        h = unpack(x)
        ac = np.convolve(h, h, mode='full')
        return np.sum(ac * ac)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0} # sum(h) = 1 => sum(x) = N/2
    ]
    
    # Bounds: allow negative values? The task says "possibly negative step heights".
    # However, for min ||h*h||_2, positive h is usually optimal for the "flat" spectrum.
    # Let's try bounds (-1, 1) to be safe, but start near positive.
    bounds = [(-1, 1)] * 200
    
    # Initial guess for symmetric part
    x0 = np.ones(200) * (N / 2.0) / 200 # sum(x) = N/2
    
    try:
        # Use SLSQP
        res = minimize(
            objective_sym, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        
        if res.success:
            h_candidate = unpack(res.x)
            val_candidate = _value(h_candidate)
            if val_candidate < val_best:
                h_best = h_candidate
                val_best = val_candidate
        else:
            # If SLSQP fails, try a simpler gradient descent on the full vector
            # with a few iterations
            pass
            
    except Exception as e:
        print(f"SLSQP failed: {e}", file=sys.stderr)
    
    # If we have time, try a local refinement on the full 400-dimensional vector
    # starting from the symmetric solution, breaking symmetry if it helps.
    if time.time() < deadline - 10:
        try:
            def objective_full(h):
                ac = np.convolve(h, h, mode='full')
                return np.sum(ac * ac)
            
            res2 = minimize(
                objective_full, 
                h_best, 
                method='SLSQP', 
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}],
                options={'maxiter': 20, 'ftol': 1e-8}
            )
            
            if res2.success:
                val_candidate = _value(res2.x)
                if val_candidate < val_best:
                    h_best = res2.x
                    val_best = val_candidate
        except Exception as e:
            print(f"Refinement failed: {e}", file=sys.stderr)

    return h_best

# EVOLVE-BLOCK-END
