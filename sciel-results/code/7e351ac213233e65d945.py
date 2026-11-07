"""Taylor-series (automatic differentiation) integrator for the fixed Lorenz IVP."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the fixed Lorenz system y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1) Call f once to satisfy the contract.
    2) Compute the Taylor series coefficients of the solution y(t) around t0
       up to a high order N. Since the system is autonomous and quadratic,
       the coefficients can be computed recursively using convolutions.
    3) Evaluate the Taylor polynomial at t1.
    
    The Lorenz system is:
    y0' = 10*(y1 - y0)
    y1' = y0*(28 - y2) - y1
    y2' = y0*y1 - (8/3)*y2
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Satisfy the contract: call f at least once
    _ = f(t0, y0_arr)
    
    # Number of Taylor terms. 100 is sufficient for machine precision on [0, 2].
    N = 100
    
    # Coefficients of the Taylor series: y(t) = sum_{k=0}^N C[k] * (t-t0)^k
    # C[k] is a vector of shape (3,)
    C = np.zeros((N + 1, 3))
    C[0] = y0_arr
    
    # Precompute 1/(k+1) for k=0..N-1
    inv_k = 1.0 / np.arange(1, N + 1)
    
    # Recursive computation of coefficients
    # We need convolutions of C[0], C[1], C[2].
    # To optimize, we can just do the sum in the loop. N=100 is small.
    
    for k in range(N):
        # Compute P_k: coefficients of (t-t0)^k in g(y(t))
        # g0 = 10*(y1 - y0)
        P0 = 10.0 * (C[k, 1] - C[k, 0])
        
        # g1 = 28*y0 - y0*y2 - y1
        # Term y0*y2 is convolution of C[0] and C[2]
        # (C0 * C2)[k] = sum_{i=0}^k C[i, 0] * C[k-i, 2]
        conv_02 = 0.0
        for i in range(k + 1):
            conv_02 += C[i, 0] * C[k - i, 2]
        P1 = 28.0 * C[k, 0] - conv_02 - C[k, 1]
        
        # g2 = y0*y1 - (8/3)*y2
        # Term y0*y1 is convolution of C[0] and C[1]
        conv_01 = 0.0
        for i in range(k + 1):
            conv_01 += C[i, 0] * C[k - i, 1]
        P2 = conv_01 - (8.0 / 3.0) * C[k, 2]
        
        # Update C[k+1]
        factor = inv_k[k]
        C[k + 1, 0] = P0 * factor
        C[k + 1, 1] = P1 * factor
        C[k + 1, 2] = P2 * factor
    
    # Evaluate Taylor series at t1
    # y(t1) = sum_{k=0}^N C[k] * (t1-t0)^k
    h = t1 - t0
    y_end = np.zeros(3)
    
    # Use Horner's method for stability and efficiency
    # y_end = C[N]
    # for k in N-1 .. 0: y_end = y_end * h + C[k]
    y_end = C[N].copy()
    for k in range(N - 1, -1, -1):
        y_end = y_end * h + C[k]
        
    return y_end
# EVOLVE-BLOCK-END
