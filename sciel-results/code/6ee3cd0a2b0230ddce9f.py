import numpy as np
import time

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

# Fourier transform of x^{2k} exp(-x^2/2) in the same basis:
# F[x^{2k} exp(-x^2/2)](xi) = i^{2k} * (2/sqrt(pi))^{1/2} * sum_{j=0}^{k} (-1)^j * (2k)!/(j! (2k-2j)!) * (sqrt(2))^{2k-2j} * xi^{2j} exp(-xi^2/2)
# Actually, let's use the standard result: if f(x) = x^n exp(-x^2/2), then
# F[f](xi) = i^n (2/sqrt(pi))^{1/2} ... this gets complicated.
# 
# Simpler: use the fact that the Fourier transform of the n-th Hermite-Gaussian is (-i)^n times itself (up to normalization).
# For our power-basis representation, we can compute the FT matrix numerically or use known formulas.
#
# Actually, the cleanest approach: the Fourier transform maps x^{2k} exp(-x^2/2) to a polynomial in xi of degree 2k times exp(-xi^2/2).
# The transformation matrix T where F[x^{2k} e^{-x^2/2}] = sum_j T[k,j] xi^{2j} e^{-xi^2/2} can be computed.
#
# For even powers, the FT is real and even. We can compute T[k,j] by noting:
# F[x^n e^{-x^2/2}](xi) = i^n (d/dx)^n [e^{-x^2/2}] evaluated appropriately... 
#
# Let me just use the known result that for the Fourier transform F[f](xi) = (1/sqrt(2*pi)) int f(x) e^{-ixi x} dx:
# F[x^{2k} e^{-x^2/2}](xi) = (1/sqrt(2*pi)) * (-i)^{2k} * (2k)! / (2^{2k}) * sum ... 
#
# Actually, the simplest correct approach: the Fourier transform of e^{-x^2/2} is e^{-xi^2/2} (up to a constant).
# And multiplication by x in position space corresponds to i*d/dxi in frequency space.
# So x^{2k} e^{-x^2/2} -> (i d/dxi)^{2k} e^{-xi^2/2} (up to constants).
#
# Let me compute the FT matrix numerically via the known identity:
# If we define the basis phi_k(x) = x^{2k} e^{-x^2/2}, then
# F[phi_k](xi) = sum_j T[k,j] phi_j(xi)
# where T can be computed from the Hermite polynomial expansion.
#
# For the purpose of this problem, since the test function is g(x) = P(x) e^{-x^2/2} where P is a polynomial,
# and the Fourier transform of such a function is also a polynomial times e^{-xi^2/2},
# we can compute the FT matrix T explicitly.
#
# Key fact: F[x^n e^{-x^2/2}](xi) = i^n (d^n/dxi^n) e^{-xi^2/2} * (1/sqrt(2*pi)) * (2*pi)^{1/2} ... 
# 
# Let me use: F[e^{-x^2/2}](xi) = e^{-xi^2/2} (with the convention F[f](xi) = (1/sqrt(2*pi)) int f(x) e^{-i x xi} dx)
# Then x f(x) -> i d/dxi F[f](xi)
# So x^{2k} e^{-x^2/2} -> (i d/dxi)^{2k} e^{-xi^2/2} = (-1)^k (d^2/dxi^2)^k e^{-xi^2/2}
#
# (d^2/dxi^2) e^{-xi^2/2} = (xi^2 - 1) e^{-xi^2/2}
# (d^2/dxi^2)^2 e^{-xi^2/2} = (d^2/dxi^2)[(xi^2-1) e^{-xi^2/2}] = (xi^4 - 4xi^2 + 2) e^{-xi^2/2}
# In general, (d^2/dxi^2)^k e^{-xi^2/2} = H_{2k}(xi/sqrt(2)) * (sqrt(2))^{-2k} * 2^k * k! * e^{-xi^2/2} ... 
#
# Actually, let me just compute T numerically by evaluating the FT at enough points or using the recurrence.
# 
# Simpler: use the fact that the FT matrix in the {x^{2k} e^{-x^2/2}} basis is:
# T[k,j] = coefficient of xi^{2j} in (i d/dxi)^{2k} e^{-xi^2/2} / e^{-xi^2/2}
#
# Let me compute this iteratively.
def _compute_ft_matrix(max_k=6):
    """Compute the Fourier transform matrix T where F[x^{2k} e^{-x^2/2}] = sum_j T[k,j] xi^{2j} e^{-xi^2/2}."""
    T = np.zeros((max_k+1, max_k+1))
    # Start: F[e^{-x^2/2}] = e^{-xi^2/2}, so T[0,0] = 1
    T[0, 0] = 1.0
    
    # x^{2k} e^{-x^2/2} -> (i d/dxi)^{2k} e^{-xi^2/2}
    # (i d/dxi)^2 = -d^2/dxi^2
    # So x^{2k} -> (-d^2/dxi^2)^k
    # 
    # Let's compute the polynomial Q_k(xi) such that (-d^2/dxi^2)^k e^{-xi^2/2} = Q_k(xi) e^{-xi^2/2}
    # Q_0(xi) = 1
    # Q_{k+1}(xi) = -d^2/dxi^2 [Q_k(xi) e^{-xi^2/2}] / e^{-xi^2/2}
    #             = -[Q_k'' - 2xi Q_k' + (xi^2 - 1) Q_k] ... wait
    # d/dxi [Q e^{-xi^2/2}] = (Q' - xi Q) e^{-xi^2/2}
    # d^2/dxi^2 [Q e^{-xi^2/2}] = (Q'' - xi Q' - Q' + xi^2 Q - xi Q) e^{-xi^2/2}
    #                            = (Q'' - 2xi Q' + (xi^2 - 1) Q) e^{-xi^2/2}
    # So -d^2/dxi^2 [Q e^{-xi^2/2}] = (-Q'' + 2xi Q' - (xi^2 - 1) Q) e^{-xi^2/2}
    # Q_{k+1} = -Q_k'' + 2xi Q_k' - (xi^2 - 1) Q_k
    
    # Represent Q_k as a polynomial in xi^2: Q_k(xi) = sum_j c[j] xi^{2j}
    # Q_0 = 1
    coeffs = [1.0] + [0.0] * max_k  # coeffs[j] is coefficient of xi^{2j}
    T[0, :] = coeffs
    
    for k in range(1, max_k+1):
        # Q_{k-1} is stored in coeffs
        # Q_k = -Q_{k-1}'' + 2xi Q_{k-1}' - (xi^2 - 1) Q_{k-1}
        # Q_{k-1}(xi) = sum_j c[j] xi^{2j}
        # Q_{k-1}'(xi) = sum_j 2j c[j] xi^{2j-1}
        # Q_{k-1}''(xi) = sum_j 2j(2j-1) c[j] xi^{2j-2}
        # 
        # -Q'' = -sum_j 2j(2j-1) c[j] xi^{2j-2}
        # 2xi Q' = 2 sum_j 2j c[j] xi^{2j} = sum_j 4j c[j] xi^{2j}
        # -(xi^2-1) Q = -sum_j c[j] xi^{2j+2} + sum_j c[j] xi^{2j}
        # 
        # Collect coefficients of xi^{2m}:
        # From -Q'': term with xi^{2m} comes from j=m+1: -2(m+1)(2(m+1)-1) c[m+1] = -2(m+1)(2m+1) c[m+1]
        # From 2xiQ': term with xi^{2m} comes from j=m: 4m c[m]
        # From -(xi^2-1)Q: term with xi^{2m} comes from j=m (from +Q) and j=m-1 (from -xi^2 Q): c[m] - c[m-1]
        # 
        # Total coefficient of xi^{2m} in Q_k:
        #   -2(m+1)(2m+1) c[m+1] + 4m c[m] + c[m] - c[m-1]
        #   = -2(m+1)(2m+1) c[m+1] + (4m+1) c[m] - c[m-1]
        
        new_coeffs = [0.0] * (max_k+1)
        for m in range(max_k+1):
            val = 0.0
            # -2(m+1)(2m+1) c[m+1]
            if m+1 <= max_k:
                val += -2.0*(m+1)*(2.0*m+1.0) * coeffs[m+1]
            # (4m+1) c[m]
            val += (4.0*m+1.0) * coeffs[m]
            # -c[m-1]
            if m >= 1:
                val += -coeffs[m-1]
            new_coeffs[m] = val
        coeffs = new_coeffs
        T[k, :] = coeffs
    
    return T

FT_mat = _compute_ft_matrix(6)  # FT_mat[k,j]: F[x^{2k} e^{-x^2/2}] = sum_j FT_mat[k,j] xi^{2j} e^{-xi^2/2}

def _compute_bound(a):
    """
    Compute the uncertainty bound for given coefficients a = [a0, a4, a8].
    Returns the bound value.
    """
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k  (Fourier transform coefficient in the same basis)
    # Actually, the FT of sum_k p[k] x^{2k} e^{-x^2/2} is sum_k p[k] * (FT of x^{2k} e^{-x^2/2})
    # = sum_k p[k] sum_j FT_mat[k,j] xi^{2j} e^{-xi^2/2}
    # = sum_j (sum_k p[k] FT_mat[k,j]) xi^{2j} e^{-xi^2/2}
    # So q[j] = sum_k p[k] FT_mat[k,j]
    q = FT_mat.T @ p  # q[j] = sum_k p[k] * FT_mat[k,j]
    
    # Compute moments
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            Mx0 += pi * pj * I[i+j]
            Mx1 += pi * pj * I[i+j+1]
            Mxi0 += qi * qj * I[i+j]
            Mxi1 += qi * qj * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    # q = FT_mat.T @ p
    q = FT_mat.T @ p
    
    # Compute moments
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            Mx0 += pi * pj * I[i+j]
            Mx1 += pi * pj * I[i+j+1]
            Mxi0 += qi * qj * I[i+j]
            Mxi1 += qi * qj * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation using chain rule
    # p = A_mat @ a, q = FT_mat.T @ p = FT_mat.T @ A_mat @ a
    # dp/da_m = A_mat[:, m]
    # dq/da_m = FT_mat.T @ A_mat[:, m]
    
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
    
    dMx0 = np.zeros(3)
    dMx1 = np.zeros(3)
    dMxi0 = np.zeros(3)
    dMxi1 = np.zeros(3)
    
    FT_A = FT_mat.T @ A_mat  # (7, 3): dq/da_m = FT_A[:, m]
    
    for m in range(3):
        Am = A_mat[:, m]
        FAm = FT_A[:, m]
        dMx0[m] = 2.0 * np.dot(Am, Sx0)
        dMx1[m] = 2.0 * np.dot(Am, Sx1)
        dMxi0[m] = 2.0 * np.dot(FAm, Sxi0)
        dMxi1[m] = 2.0 * np.dot(FAm, Sxi1)
    
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    
    for m in range(3):
        dvx_m = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / (Mx0*Mx0)
        dvxi_m = (dMxi1[m]*Mxi0 - Mxi1*dMxi0[m]) / (Mxi0*Mxi0)
        grad[m] = factor * (vxi*dvx_m + vx*dvxi_m)
        
    return bound, grad

def _solve_via_eigenvalue():
    """
    Formulate the minimization as a generalized eigenvalue problem using the
    exact Fourier transform matrix.
    
    The uncertainty product squared is:
    (Mx1/Mx0) * (Mxi1/Mxi0)
    
    where:
    Mx0 = p^T Gx0 p, Mx1 = p^T Gx1 p
    Mxi0 = q^T Gxi0 q, Mxi1 = q^T Gxi1 q
    
    with p = A @ a, q = FT_mat.T @ A @ a =: B @ a
    
    So:
    Mx0 = a^T (A^T Gx0 A) a = a^T M0 a
    Mx1 = a^T (A^T Gx1 A) a = a^T M1 a
    Mxi0 = a^T (B^T Gxi0 B) a = a^T N0 a
    Mxi1 = a^T (B^T Gxi1 B) a = a^T N1 a
    
    We want to minimize F(a) = (a^T M1 a)/(a^T M0 a) * (a^T N1 a)/(a^T N0 a)
    
    At the optimum, the gradient vanishes. This leads to a secular equation
    that can be solved for the optimal Lagrange multiplier.
    
    Alternative approach: use the fact that at the minimum, there exists a
    scalar lambda such that the gradient condition holds. We can set up the
    system and solve it.
    """
    # Build the moment matrices in the p-space (7x7)
    Gx0 = np.zeros((7, 7))
    Gx1 = np.zeros((7, 7))
    Gxi0 = np.zeros((7, 7))
    Gxi1 = np.zeros((7, 7))
    
    for i in range(7):
        for j in range(7):
            Gx0[i, j] = I[i + j]
            Gx1[i, j] = I[i + j + 1]
            Gxi0[i, j] = I[i + j]
            Gxi1[i, j] = I[i + j + 1]
    
    # B = FT_mat.T @ A_mat
    B_mat = FT_mat.T @ A_mat
    
    # Transform to a-space (3x3)
    M0 = A_mat.T @ Gx0 @ A_mat
    M1 = A_mat.T @ Gx1 @ A_mat
    N0 = B_mat.T @ Gxi0 @ B_mat
    N1 = B_mat.T @ Gxi1 @ B_mat
    
    # Ensure symmetry
    M0 = (M0 + M0.T) / 2.0
    M1 = (M1 + M1.T) / 2.0
    N0 = (N0 + N0.T) / 2.0
    N1 = (N1 + N1.T) / 2.0
    
    # Minimize F(a) = (a^T M1 a)/(a^T M0 a) * (a^T N1 a)/(a^T N0 a)
    # 
    # Let u = a^T M1 a / a^T M0 a, v = a^T N1 a / a^T N0 a
    # We want to minimize u * v.
    #
    # At the optimum, for any direction da:
    # d(u*v)/da = 0
    # v * du/da + u * dv/da = 0
    #
    # This gives us a system of 3 equations (one per component of a).
    #
    # Alternatively, we can use an iterative approach:
    # 1. Fix v = v0, minimize u subject to a^T N1 a / a^T N0 a = v0
    # 2. Fix u = u0, minimize v subject to a^T M1 a / a^T M0 a = u0
    # 3. Iterate
    #
    # Or use the secular equation approach:
    # At the optimum, there exists lambda such that:
    # (v * M1 - u * M0) a = lambda * (v * N0 - u * N1) a ... this isn't quite right.
    #
    # Let me use a simpler iterative approach with the exact FT matrix.
    
    def _optimize_ratio(M_num, M_den, a_init):
        """Minimize (a^T M_num a) / (a^T M_den a) starting from a_init."""
        try:
            # Generalized eigenvalue: M_num a = lambda M_den a
            # Solve M_den^{-1} M_num a = lambda a
            M_den_inv_M_num = np.linalg.solve(M_den, M_num)
            # Make symmetric for eigvalsh
            M_sym = (M_den_inv_M_num + M_den_inv_M_num.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(M_sym)
            idx = np.argmin(eigvals)
            a_opt = eigvecs[:, idx]
            a_opt = a_opt / np.linalg.norm(a_opt)
            return a_opt, eigvals[idx]
        except np.linalg.LinAlgError:
            return a_init, None
    
    # Start with the seed
    a = np.array([-1.13, 0.04, 0.000308642])
    a = a / np.linalg.norm(a)
    
    for _ in range(200):
        # Optimize the position ratio: minimize a^T M1 a / a^T M0 a
        a_pos, _ = _optimize_ratio(M1, M0, a)
        # Optimize the momentum ratio: minimize a^T N1 a / a^T N0 a
        a_mom, _ = _optimize_ratio(N1, N0, a_pos)
        
        # Average the two to get a compromise
        a_new = a_pos + a_mom
        a_new = a_new / np.linalg.norm(a_new)
        
        # Check convergence
        if np.linalg.norm(a_new - a) < 1e-14:
            a = a_new
            break
        a = a_new
    
    # Evaluate the bound
    val = _compute_bound(a)
    return a, val

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5  # seconds
    
    # First, try the eigenvalue-based approach with exact FT matrix
    if time.time() < _DEADLINE - deadline_margin:
        try:
            a_eig, val_eig = _solve_via_eigenvalue()
            if np.isfinite(val_eig) and val_eig < best_val:
                best_val = val_eig
                best_params = a_eig.copy()
        except Exception:
            pass
    
    # Use direct optimization in a-space with L-BFGS-B
    def _obj_and_grad(a):
        val, grad = _compute_bound_and_grad(a)
        return val, grad
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starts = [seed_params.copy()]
    
    # Add the eigenvalue solution as a start
    if np.all(np.isfinite(best_params)):
        starts.append(best_params.copy())
    
    # Grid starts around seed
    for da0 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        for da4 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for da8 in [-0.0001, -0.00005, 0.0, 0.00005, 0.0001]:
                candidate = seed_params + np.array([da0, da4, da8])
                starts.append(candidate)
    
    # Random starts
    for _ in range(200):
        a0 = rng.uniform(-3, 1)
        a4 = rng.uniform(-0.2, 0.2)
        a8 = rng.uniform(-0.001, 0.002)
        starts.append(np.array([a0, a4, a8]))
    
    # Some structured starts with different sign patterns
    for sign0 in [-1, 1]:
        for sign4 in [-1, 1]:
            for sign8 in [-1, 1]:
                for mag0 in [0.5, 1.0, 2.0]:
                    for mag4 in [0.01, 0.05, 0.1]:
                        for mag8 in [0.0001, 0.0005, 0.001]:
                            starts.append(np.array([
                                sign0 * mag0,
                                sign4 * mag4,
                                sign8 * mag8
                            ]))
    
    # Shuffle starts (keep seed first)
    rng2 = np.random.default_rng(99)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    # Optimization loop
    for idx, start in enumerate(starts):
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            from scipy.optimize import minimize
            res = minimize(_obj_and_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < _DEADLINE - deadline_margin:
        try:
            from scipy.optimize import minimize
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead for local refinement
    if time.time() < _DEADLINE - deadline_margin:
        try:
            from scipy.optimize import minimize
            res = minimize(lambda a: _compute_bound(a), best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # One more round of L-BFGS-B from the Nelder-Mead result
    if time.time() < _DEADLINE - deadline_margin:
        try:
            from scipy.optimize import minimize
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
