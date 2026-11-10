"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a low-rank separable basis.
    f(x, y) ≈ sum_{k=0}^{4} params[k] * P_k(x) * params[5+k] * P_k(y)
    
    This models f(x,y) as a sum of 5 rank-1 terms:
    f(x,y) ≈ (sum_{k=0}^{4} a_k P_k(x)) * (sum_{k=0}^{4} b_k P_k(y)) is NOT what this does.
    Actually, the params are:
    params[0:5] -> coefficients for x-basis
    params[5:10] -> coefficients for y-basis
    
    Wait, a standard low rank expansion is sum_{r=1}^R u_r(x) v_r(y).
    With 10 params, we can do R=2, with 4 params for u's (2 vectors of 2 coeffs? No, basis is fixed).
    
    Let's use the strategy suggested: 
    f(x,y) ≈ sum_{i=0}^3 a_i(x) b_i(y)
    where a_i and b_i are fixed basis functions? 
    If a_i(x) = P_i(x) and b_i(y) = P_i(y), then:
    f(x,y) = sum_{i=0}^3 c_i P_i(x) P_i(y).
    This uses 4 parameters. We have 10.
    
    The prompt says: "Implement a separable quadratic basis using the form f(x,y) ≈ sum_{i=0}^3 a_i(x) b_i(y), where a_i and b_i are fixed, orthonormalized polynomial bases"
    
    If the basis is fixed, the only free parameters are the coefficients in the sum.
    If we use i=0..3, we have 4 coefficients.
    We have 10 params.
    
    Perhaps the strategy implies:
    f(x,y) = (sum_{j=0}^4 u_j P_j(x)) * (sum_{k=0}^4 v_k P_k(y)) ?
    This is rank-1. It uses 10 parameters (5 for u, 5 for v).
    This is a very strong structural constraint (rank 1).
    
    Alternatively, a sum of two rank-1 terms?
    f(x,y) = (sum_{j=0}^1 u_j P_j(x)) (sum_{k=0}^1 v_k P_k(y)) + (sum_{j=0}^1 w_j P_j(x)) (sum_{k=0}^1 z_k P_k(y))
    This uses 8 parameters.
    
    Let's stick to the most direct interpretation of "separable ... to explicitly model low-rank structure".
    A rank-1 model is the most extreme low-rank structure.
    f(x,y) = U(x) * V(y)
    where U(x) = sum_{j=0}^4 params[j] * P_j(x)
    and V(y) = sum_{k=0}^4 params[5+k] * P_k(y)
    
    This uses all 10 parameters.
    Legendre polynomials P_j(t) on [-1, 1]:
    P_0(t) = 1
    P_1(t) = t
    P_2(t) = 0.5 * (3t^2 - 1)
    P_3(t) = 0.5 * (5t^3 - 3t)
    P_4(t) = 0.375 * (35t^4 - 30t^2 + 3)
    
    This formulation is non-linear in the parameters (product of two linear combinations).
    BFGS can handle this.
    """
    # Extract parameters
    # First 5 for x-direction, next 5 for y-direction
    ux = params[0:5]
    vy = params[5:10]
    
    # Compute Legendre polynomials for x
    # P0
    P0_x = np.ones_like(x)
    # P1
    P1_x = x
    # P2
    P2_x = 0.5 * (3 * x**2 - 1)
    # P3
    P3_x = 0.5 * (5 * x**3 - 3 * x)
    # P4
    P4_x = 0.375 * (35 * x**4 - 30 * x**2 + 3)
    
    # Compute Legendre polynomials for y
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = 0.5 * (3 * y**2 - 1)
    P3_y = 0.5 * (5 * y**3 - 3 * y)
    P4_y = 0.375 * (35 * y**4 - 30 * y**2 + 3)
    
    # Construct U(x) and V(y)
    U_x = ux[0] * P0_x + ux[1] * P1_x + ux[2] * P2_x + ux[3] * P3_x + ux[4] * P4_x
    V_y = vy[0] * P0_y + vy[1] * P1_y + vy[2] * P2_y + vy[3] * P3_y + vy[4] * P4_y
    
    # Rank-1 product
    return U_x * V_y
# EVOLVE-BLOCK-END
