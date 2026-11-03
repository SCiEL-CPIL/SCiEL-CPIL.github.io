import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
import time


def construct():
    t_start = time.time()
    deadline = 195.0  # safety margin

    # Hermite polynomial coefficients (physicist's convention)
    # H0(x) = 1
    # H4(x) = 12 - 48x^2 + 16x^4
    # H8(x) = 1680 - 32256x^2 + 23040x^4 - 4608x^6 + 256x^8
    # H12(x) = 665280 - 12355200x^2 + 13271040x^4 - 5806080x^6 + 1032192x^8 - 98304x^10 + 4096x^12
    
    # Coefficients indexed by power//2 (i.e., coefficient of x^{2m})
    H0_c = np.array([1.0])
    H4_c = np.array([12.0, -48.0, 16.0])
    H8_c = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_c = np.array([665280.0, -12355200.0, 13271040.0, -5806080.0,
                      1032192.0, -98304.0, 4096.0])
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12
    # = (2k-1)!! / (2π)^k
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Fourier transform: Q[m] = P[m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    def build_P(c0, c4, c8):
        """Build P array (coefficients of x^{2m} for m=0..6) given c0, c4, c8."""
        a12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        P = np.zeros(7)
        P[0] += c0 * H0_c[0]
        for m in range(3):
            P[m] += c4 * H4_c[m]
        for m in range(5):
            P[m] += c8 * H8_c[m]
        for m in range(7):
            P[m] += a12 * H12_c[m]
        return P

    def compute_norm_sq(P):
        """Compute ||g||² = sum_k E_int[k] * (sum_{i+j=k} P[i]*P[j])"""
        P2 = np.convolve(P, P)
        return np.dot(P2, E_int)

    def compute_norm_sq_hat(P):
        """Compute ||ĝ||²"""
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        return np.dot(Q2, E_int)

    def objective(c):
        P = build_P(c[0], c[1], c[2])
        n1 = compute_norm_sq(P)
        n2 = compute_norm_sq_hat(P)
        return n1 * n2

    # Build the quadratic form matrices for ||g||² and ||ĝ||²
    # P is linear in (c0, c4, c8), so ||g||² is a quadratic form in c
    # P[m] = c0 * H0_c[m] + c4 * H4_c[m] + c8 * H8_c[m] + a12 * H12_c[m]
    # where a12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    # Let's express P[m] = sum_i c_i * V[i][m] where c = [c0, c4, c8]
    # V[i][m] = Hermite_c[i][m] - (Hermite_0[i] / H12_0) * H12_c[m]
    
    V = np.zeros((3, 7))
    for i, (Hc, H0_val) in enumerate([(H0_c, H0_0), (H4_c, H4_0), (H8_c, H8_0)]):
        for m in range(len(Hc)):
            V[i, m] = Hc[m]
        for m in range(7):
            V[i, m] -= (H0_val / H12_0) * H12_c[m]
    
    # ||g||² = sum_{k=0}^{12} E_int[k] * (sum_{i+j=k} P[i]*P[j])
    # P = V @ c (where c = [c0, c4, c8])
    # P[i]*P[j] = (V[i] @ c) * (V[j] @ c) = c^T (V[i] V[j]^T) c
    # ||g||² = c^T M c where M = sum_k E_int[k] * sum_{i+j=k} V[i] V[j]^T
    
    M = np.zeros((3, 3))
    M_hat = np.zeros((3, 3))
    for k in range(13):
        for i in range(7):
            for j in range(7):
                if i + j == k:
                    M += E_int[k] * np.outer(V[:, i], V[:, j])
                    # For hat: Q[m] = P[m] * (-1)^m, so V_hat[i][m] = V[i][m] * (-1)^m
                    Vi_hat = V[:, i] * signs[i]
                    Vj_hat = V[:, j] * signs[j]
                    M_hat += E_int[k] * np.outer(Vi_hat, Vj_hat)
    
    # The objective is f(c) = (c^T M c) * (c^T M_hat c)
    # This is a homogeneous quartic form. Critical points satisfy:
    # ∇f = 0, i.e., M c * (c^T M_hat c) + M_hat c * (c^T M c) = 0
    # Let α = c^T M c, β = c^T M_hat c
    # M c * β + M_hat c * α = 0
    # This is a system of 3 equations. Since it's homogeneous, we can normalize.
    # Divide by α (assuming α ≠ 0):
    # M c * (β/α) + M_hat c = 0
    # Let λ = β/α = (c^T M_hat c) / (c^T M c)
    # Then: M c * λ + M_hat c = 0, i.e., (λ M + M_hat) c = 0
    # This is a generalized eigenvalue problem: M_hat c = -λ M c
    # Or: (M_hat + λ M) c = 0
    # Non-trivial solution exists when det(M_hat + λ M) = 0
    # This is a cubic in λ (since M and M_hat are 3x3)
    
    # det(M_hat + λ M) = 0
    # Expand: this is a polynomial in λ of degree 3
    # Let's compute it explicitly
    
    # A(λ) = M_hat + λ M
    # det(A) = a3 λ^3 + a2 λ^2 + a1 λ + a0
    
    # Compute using the formula for det of 3x3 matrix
    def det3(mat):
        return (mat[0,0]*(mat[1,1]*mat[2,2]-mat[1,2]*mat[2,1])
                - mat[0,1]*(mat[1,0]*mat[2,2]-mat[1,2]*mat[2,0])
                + mat[0,2]*(mat[1,0]*mat[2,1]-mat[1,1]*mat[2,0]))
    
    # det(M_hat + λ M) as polynomial in λ
    # Coefficient of λ^3: det(M)
    a3 = det3(M)
    # Coefficient of λ^2: sum of 2x2 minors with one row from M_hat and rest from M
    # More precisely, use the expansion
    # Actually, let me just evaluate at several points and interpolate, or use sympy-like approach
    
    # Better: compute the characteristic polynomial directly
    # det(M_hat + λ M) = λ^3 det(M) + λ^2 * (sum of cofactors) + ...
    # Let me compute by evaluating at λ = 0, 1, 2, 3 and interpolating
    
    vals = []
    for lam in [0.0, 1.0, 2.0, 3.0]:
        A = M_hat + lam * M
        vals.append(det3(A))
    
    # Fit cubic: p(λ) = a3 λ^3 + a2 λ^2 + a1 λ + a0
    # Using the 4 values
    lam_pts = np.array([0.0, 1.0, 2.0, 3.0])
    coeffs_poly = np.polyfit(lam_pts, vals, 3)
    # coeffs_poly = [a3, a2, a1, a0] in descending power order
    
    # Find roots of the cubic
    lambda_roots = np.roots(coeffs_poly)
    
    best_c = None
    best_val = np.inf
    
    for lam in lambda_roots:
        if not np.isreal(lam):
            continue
        lam_r = lam.real
        if abs(lam.imag) > 1e-10:
            continue
        
        # Solve (M_hat + lam_r * M) c = 0
        A = M_hat + lam_r * M
        # Find null vector
        try:
            U, S, Vt = np.linalg.svd(A)
            c_null = Vt[-1]  # last row of Vt is the right singular vector for smallest singular value
            # Normalize
            norm_c = np.linalg.norm(c_null)
            if norm_c < 1e-15:
                continue
            c_null = c_null / norm_c
            
            # Check both signs
            for sign in [1.0, -1.0]:
                c_test = sign * c_null
                val = objective(c_test)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_c = c_test.copy()
        except Exception:
            continue
    
    # Also try the seed point
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    val_seed = objective(c_seed)
    if np.isfinite(val_seed) and val_seed < best_val:
        best_val = val_seed
        best_c = c_seed.copy()
    
    # If we found a good candidate, polish it
    if best_c is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_c,
                method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-18, 'disp': False}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Additional Nelder-Mead polish with very tight tolerance
    if best_c is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Try multiple starting points from the algebraic solutions + seed
    if time.time() < deadline - 20:
        start_points = []
        if best_c is not None:
            start_points.append(best_c)
        start_points.append(c_seed)
        
        # Add perturbed versions
        for base in start_points[:]:
            for scale in [0.5, 1.5, 0.1, 2.0]:
                start_points.append(base * scale)
        
        for sp in start_points:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(
                    objective,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_c = result.x.copy()
            except Exception:
                continue
    
    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                objective,
                best_c,
                method='BFGS',
                options={'maxiter': 10000, 'gtol': 1e-20, 'disp': False}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    if best_c is None:
        return c_seed
    
    return best_c
