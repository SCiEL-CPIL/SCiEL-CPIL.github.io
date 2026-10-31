import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[n] = (2n-1)!! / 2^n * sqrt(pi)
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# Physicist's Hermite polynomials: H_n(x)
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# Constraint g(0) = 0 => P(0) = 0
# P(0) = a0*3 + a4*3 + a8*105 + a12*945 = 0  (Wait, H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945)
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A_mat such that p = A_mat * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6) corresponding to x^0, x^2, ..., x^12

H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

# Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
# So q_k = p_k * (-i)^k? 
# Wait, the Fourier transform of x^{2k} is not just a scalar multiple.
# However, the problem states "Fourier-invariant Hermite-polynomial times Gaussian".
# For $g(x) = P(x) e^{-x^2/2}$, $\hat{g}(\xi) = \sqrt{2\pi} \sum c_k (-i)^k H_k(\xi) e^{-\xi^2/2}$.
# So the coefficients for the Fourier side are $q_k = c_k (-i)^k$.
# Since we are dealing with real variances, we look at $|\hat{g}|^2$.
# The integral $\int \xi^2 |\hat{g}|^2 d\xi$ will involve $q_k q_j$.
# Since $q_k$ are complex, we need to be careful.
# Actually, $H_n$ are real. $\hat{g}$ is complex.
# $|\hat{g}|^2 = \hat{g} \hat{g}^*$.
# The variance calculation in Inspiration 2 used `sign_vec = [1, -1, 1, -1, ...]`.
# This corresponds to $(-i)^k$ squared? No, $(-i)^k (-i)^{-k} = 1$.
# But the cross terms?
# Let's stick to the Inspiration 2 formulation which was successful.
# It used `q = p * sign_vec`. This implies the Fourier coefficients are effectively $p_k \cdot \text{sign}_k$.
# This is correct for the *squared* modulus if we consider the basis functions $H_k e^{-x^2/2}$ are orthogonal and the Fourier transform maps $H_k$ to $(-i)^k H_k$.
# The integral of $\xi^2 H_i H_j$ is zero if $i \ne j$? No, $\xi^2$ couples $H_i$ to $H_{i\pm 2}$.
# The matrix formulation in Inspiration 2 handles this via $M_1$.

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound J = sqrt(Vx * Vxi) and its gradient.
    """
    p = _get_p(a)
    q = p * sign_vec
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            idx0 = i + j
            idx1 = i + j + 1
            I0 = I[idx0]
            I1 = I[idx1]
            
            Mx0 += pi * p[j] * I0
            Mx1 += pi * p[j] * I1
            
            Mxi0 += qi * q[j] * I0
            Mxi1 += qi * q[j] * I1
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    Mx0_calc = np.dot(p, Sx[0])
    Mx1_calc = np.dot(p, Sx[1])
    Mxi0_calc = np.dot(q, Sxi[0])
    Mxi1_calc = np.dot(q, Sxi[1])
    
    if Mx0_calc <= 1e-15 or Mxi0_calc <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx_calc = Mx1_calc / Mx0_calc
    vxi_calc = Mxi1_calc / Mxi0_calc
    
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0_calc - Mx1_calc*dMx[0, m]) / (Mx0_calc*Mx0_calc)
        dvxi[m] = (dMxi[1, m]*Mxi0_calc - Mxi1_calc*dMxi[0, m]) / (Mxi0_calc*Mxi0_calc)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi_calc*dvx[m] + vx_calc*dvxi[m])
        
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val if np.isfinite(val) else 1e18

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    if not np.isfinite(val):
        return 1e18, np.zeros(3)
    return val, grad

def construct():
    t_start = time.time()
    DEADLINE = 205.0  # Safety margin
    
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    best = seed.copy()
    best_val, _ = _compute_bound_and_grad(*seed)
    
    # Generate initial guesses
    # 1. The seed
    starts = [seed.copy()]
    
    # 2. Random starts on the sphere (scale invariance)
    rng = np.random.default_rng(42)
    for i in range(20):
        v = rng.normal(0, 1, 3)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm
            # Check if g(0) != 0 is handled by the linear constraint in A_mat, 
            # so any v is a valid direction in the 3D space of free coeffs.
            starts.append(v)
            
    # 3. Perturbations of the seed
    for i in range(10):
        scale = 0.1 / (i + 1)
        v = seed + rng.normal(0, scale, 3)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            starts.append(v / norm)
            
    # Optimize from each start
    for s in starts:
        if time.time() > DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 10000, 'gtol': 1e-15, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
            
    # Final high-precision polish
    if time.time() < DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 50000, 'gtol': 1e-18, 'ftol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    return np.array(best)
