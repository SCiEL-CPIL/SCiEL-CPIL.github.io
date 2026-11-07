"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a cubic B-spline basis with 10 parameters.
    
    Cubic B-splines provide a smooth, local basis that is often better conditioned
    for optimization than high-degree global polynomials. This helps BFGS converge
    more reliably to a good local minimum for a variety of smooth target functions.
    """
    n = len(x)
    if n == 0:
        return np.array([])
        
    # Define knot vector for 10 cubic B-splines
    # We need N+4 = 14 knots for N=10 cubic splines.
    # Domain is [0.1, 6]. We extend the knot range slightly to ensure coverage.
    # u = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7]
    u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0])
    
    # Initialize basis matrix
    # M[i, j] is the value of the j-th basis function at x[i]
    M = np.zeros((n, 10), dtype=np.float64)
    
    # Cox-de Boor recursion for cubic B-splines (order 4)
    # B_{i,0}(x)
    for i in range(len(u) - 1):
        if u[i] < u[i+1]: # Only if interval is non-zero
            mask = (x >= u[i]) & (x < u[i+1])
            M[mask, i] = 1.0
            
    # Note: The standard indexing for B-splines in Cox-de Boor is a bit tricky with the number of basis functions.
    # Let's use a more robust iterative evaluation for each basis function.
    # Actually, for N=10, we can just compute the 10 basis functions explicitly.
    # Let's restart the calculation with a clearer method.
    
    M = np.zeros((n, 10), dtype=np.float64)
    
    # Helper to compute B_{i,k}(x)
    # We will compute all B_{i,0}, then B_{i,1}, then B_{i,2}, then B_{i,3}.
    # We need to be careful with the number of terms.
    # For order 4 (cubic), we start with k=0.
    # Number of B_{i,0} terms = len(u) - 1 = 13.
    # Number of B_{i,1} terms = len(u) - 2 = 12.
    # Number of B_{i,2} terms = len(u) - 3 = 11.
    # Number of B_{i,3} terms = len(u) - 4 = 10.
    # So we want the 10 terms of B_{i,3}.
    
    # k=0
    B_curr = np.zeros((n, len(u) - 1), dtype=np.float64)
    for i in range(len(u) - 1):
        if u[i] < u[i+1]:
            mask = (x >= u[i]) & (x < u[i+1])
            B_curr[mask, i] = 1.0
        # Handle the last interval if x == u[-1]? Usually B-splines are defined on [u0, u_{n-1}].
        # If x == u[-1], the last interval [u_{n-2}, u_{n-1}] usually includes the right endpoint in some definitions.
        # Let's assume standard half-open intervals [u_i, u_{i+1}) except for the last one which is [u_{n-2}, u_{n-1}].
        if i == len(u) - 2:
             mask = (x >= u[i]) & (x <= u[i+1])
             B_curr[mask, i] = 1.0

    # k=1
    B_next = np.zeros((n, len(u) - 2), dtype=np.float64)
    for i in range(len(u) - 2):
        denom1 = u[i+1] - u[i]
        denom2 = u[i+2] - u[i+1]
        if denom1 > 0:
            B_next[:, i] += (x - u[i]) / denom1 * B_curr[:, i]
        if denom2 > 0:
            B_next[:, i] += (u[i+2] - x) / denom2 * B_curr[:, i+1]
    B_curr = B_next
    
    # k=2
    B_next = np.zeros((n, len(u) - 3), dtype=np.float64)
    for i in range(len(u) - 3):
        denom1 = u[i+2] - u[i]
        denom2 = u[i+3] - u[i+1]
        if denom1 > 0:
            B_next[:, i] += (x - u[i]) / denom1 * B_curr[:, i]
        if denom2 > 0:
            B_next[:, i] += (u[i+3] - x) / denom2 * B_curr[:, i+1]
    B_curr = B_next
    
    # k=3
    B_next = np.zeros((n, len(u) - 4), dtype=np.float64)
    for i in range(len(u) - 4):
        denom1 = u[i+3] - u[i]
        denom2 = u[i+4] - u[i+1]
        if denom1 > 0:
            B_next[:, i] += (x - u[i]) / denom1 * B_curr[:, i]
        if denom2 > 0:
            B_next[:, i] += (u[i+4] - x) / denom2 * B_curr[:, i+1]
            
    M = B_next
    
    # Check for non-finite values
    if not np.all(np.isfinite(M)):
        return np.zeros(n)
        
    return M @ params
# EVOLVE-BLOCK-END
