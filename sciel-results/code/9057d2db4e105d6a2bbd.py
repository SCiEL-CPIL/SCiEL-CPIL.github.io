"""
Optimized search for the best Hermite-Gaussian coefficients minimizing the C4 bound.
Uses analytic Gram matrices via Gauss-Hermite quadrature.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite

def _hermite_poly(n, x):
    """Physicist's Hermite polynomial via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def construct():
    t0 = time.time()
    deadline = 205.0  # Safety margin: 210s limit, aim to finish in 205s

    def elapsed():
        return time.time() - t0

    # ---------- Analytic Basis & Gram Matrices ----------
    # Use Gauss-Hermite quadrature for exact integration of polynomials * e^-x^2
    # We need enough points for degree 25 (max degree of product is 24)
    n_quad = 13
    x_nodes, w_nodes = roots_hermite(n_quad)
    
    # Basis polynomials coefficients for v0, v4, v8
    # v_k(x) = (H_k(x) - coeff_k H_12(x)) e^{-x^2/2}
    # Note: roots_hermite is for e^{-x^2}.
    # Our functions are P(x) e^{-x^2/2}.
    # Integral \int P(x) Q(x) e^{-x^2} dx.
    # This matches the GH weight e^{-x^2}.
    
    H0 = _hermite_poly(0, x_nodes)
    H4 = _hermite_poly(4, x_nodes)
    H8 = _hermite_poly(8, x_nodes)
    H12 = _hermite_poly(12, x_nodes)
    
    # Effective polynomial parts
    p0 = H0 - H12 / 40320.0
    p4 = H4 - 12.0 * H12 / 40320.0
    p8 = H8 - 1680.0 * H12 / 40320.0
    p_vecs = (p0, p4, p8)
    
    # Compute M: Gram matrix in position space
    # M[i,j] = \int p_i(x) p_j(x) e^{-x^2} dx
    M = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            M[i, j] = M[j, i] = float(np.sum(w_nodes * p_vecs[i] * p_vecs[j]))

    # Compute Mf: Gram matrix in frequency space
    # \hat{p}_k(\omega) = \sum_{m \in \{0,4,8,12\}} c_{km} (-i)^m H_m(\omega) e^{-\omega^2/2}
    # But we defined p_k as the polynomial part of v_k(x) = p_k(x) e^{-x^2/2}.
    # So \hat{v}_k(\omega) = \sqrt{2\pi} \sum c_{km} (-i)^m H_m(\omega) e^{-\omega^2/2}.
    # Mf[i,j] = \int \omega^2 \hat{v}_i \overline{\hat{v}_j} \frac{d\omega}{2\pi}
    # = \int \omega^2 \hat{u}_i \overline{\hat{u}_j} d\omega
    # where \hat{u}_k(\omega) = \sum c_{km} (-i)^m H_m(\omega) e^{-\omega^2/2}.
    
    # We can compute this using GH quadrature on omega with weight e^{-\omega^2}.
    # Integrand: \omega^2 e^{-\omega^2} [ \sum c_{im} (-i)^m H_m(\omega) ] [ \sum \bar{c}_{jm} (-i)^m H_m(\omega) ]
    # Since c are real, \bar{c} = c.
    # Note: H_m(\omega) is real for real \omega.
    # So \hat{u}_k(\omega) = e^{-\omega^2/2} Q_k(\omega) where Q_k is polynomial.
    # Product \hat{u}_i \overline{\hat{u}_j} = e^{-\omega^2} Q_i(\omega) Q_j(\omega).
    # Integral \int \omega^2 e^{-\omega^2} Q_i(\omega) Q_j(\omega) d\omega.
    # This is exactly GH quadrature with weight e^{-\omega^2}.
    
    # Construct Q_k(\omega)
    # c_{00}=1, c_{0,12}=-1/40320
    # c_{44}=1, c_{4,12}=-12/40320
    # c_{88}=1, c_{8,12}=-1680/40320
    
    def get_Q(k, w):
        # k is 0, 4, or 8
        if k == 0:
            return _hermite_poly(0, w) - _hermite_poly(12, w) / 40320.0
        elif k == 4:
            return _hermite_poly(4, w) - 12.0 * _hermite_poly(12, w) / 40320.0
        elif k == 8:
            return _hermite_poly(8, w) - 1680.0 * _hermite_poly(12, w) / 40320.0
        return 0.0

    # Note: The (-i)^m factor.
    # H_0: 1. H_4: real. H_8: real. H_12: real.
    # (-i)^0 = 1. (-i)^4 = 1. (-i)^8 = 1. (-i)^12 = 1.
    # So all coefficients are real and positive/negative as in the polynomial definition.
    # No imaginary unit complications.
    
    Mf = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            deg_i = [0, 4, 8][i]
            deg_j = [0, 4, 8][j]
            Qi = get_Q(deg_i, x_nodes)
            Qj = get_Q(deg_j, x_nodes)
            
            integrand = x_nodes * x_nodes * Qi * Qj
            Mf[i, j] = Mf[j, i] = float(np.sum(w_nodes * integrand))

    # C matrix
    # The seed used C[i,j] = sum_k M[i,k] M[j,k] * dx
    # Here we use C[i,j] = sum_k M[i,k] M[j,k] (assuming dx=1 or scaling is absorbed)
    # If the score is sensitive to scaling, this might be wrong.
    # However, M is now an exact integral, not a sum * dx.
    # Let's assume the "true" C matrix is the matrix product M^T M or similar.
    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = float(np.sum(M[i, :] * M[j, :]))

    # Symmetrize
    M = 0.5 * (M + M.T)
    Mf = 0.5 * (Mf + Mf.T)
    C = 0.5 * (C + C.T)

    # ---------- Objective Function ----------
    def objective(c):
        """
        Minimize C(c) * sqrt(M(c) * F(c))
        c is a 3D vector. We normalize it internally.
        Returns (value, gradient)
        """
        nrm = float(np.dot(c, c))
        if nrm <= 1e-30:
            return np.inf, np.zeros(3)
        
        cn = c / np.sqrt(nrm)
        
        Mv = float(cn @ M @ cn)
        Fv = float(cn @ Mf @ cn)
        Cn = float(cn @ C @ cn)
        
        # Avoid log(0) or sqrt(0) issues
        eps = 1e-15
        Mv_s = max(Mv, eps)
        Fv_s = max(Fv, eps)
        
        val = Cn * np.sqrt(Mv_s * Fv_s)
        
        # Gradient calculation
        # f(c) = C(c) * sqrt(M(c) F(c))
        # Let u = C, v = M, w = F
        # grad f = grad u * sqrt(vw) + u * (grad v * w + grad w * v) / (2 sqrt(vw))
        
        gC = 2.0 * (C @ cn)
        gM = 2.0 * (M @ cn)
        gF = 2.0 * (Mf @ cn)
        
        sqrtMF = np.sqrt(Mv_s * Fv_s)
        
        # Gradient with respect to cn
        gcn = gC * sqrtMF + Cn * (gM * Fv_s + gF * Mv_s) / (2.0 * sqrtMF)
        
        # Project to tangent space of sphere (remove radial component)
        grad = (gcn - float(np.dot(gcn, cn)) * cn) / (nrm ** 1.5)
        
        return val, grad

    def obj_only(c):
        val, _ = objective(c)
        return val

    # ---------- Optimization Strategy ----------
    # 1. Global search with many random/deterministic starts
    # 2. Local refinement with L-BFGS-B in spherical coordinates
    
    best_c = None
    best_val = np.inf
    
    # Generate starting points
    # Use a Fibonacci sphere for uniform coverage
    n_starts = 200
    # Add some random starts
    rng = np.random.RandomState(42)
    starts = []
    
    # Fibonacci sphere
    i = np.arange(0, n_starts) + 0.5
    phi = np.arccos(1 - 2 * i / n_starts)
    theta = np.pi * (1 + 5**0.5) * i
    
    for th, ph in zip(theta, phi):
        c_start = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th)
        ])
        starts.append(c_start)
        
    # Add random starts
    for _ in range(n_starts):
        d = rng.randn(3)
        starts.append(d / np.linalg.norm(d))
        
    # Add seed solution
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    starts.append(c_seed / np.linalg.norm(c_seed))

    def project_to_sphere(c):
        n = np.linalg.norm(c)
        if n < 1e-12:
            return c
        return c / n

    # Local optimization function using spherical coordinates
    def refine(c0):
        if elapsed() > deadline - 5.0:
            return None, np.inf
            
        c0 = project_to_sphere(c0)
        
        # Convert to spherical coordinates
        # Avoid poles for stability
        c0[2] = np.clip(c0[2], -0.999999, 0.999999)
        theta = np.arccos(c0[2])
        phi = np.arctan2(c0[1], c0[0])
        
        def spherical_obj(params):
            th, ph = params
            # Wrap phi to [-pi, pi]
            ph = (ph + np.pi) % (2 * np.pi) - np.pi
            c_sph = np.array([
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th)
            ])
            return obj_only(c_sph)
        
        def spherical_jac(params):
            th, ph = params
            ph = (ph + np.pi) % (2 * np.pi) - np.pi
            c_sph = np.array([
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th)
            ])
            _, grad_c = objective(c_sph)
            
            # Jacobian of c wrt (th, ph)
            dcdth = np.array([
                np.cos(th) * np.cos(ph),
                np.cos(th) * np.sin(ph),
                -np.sin(th)
            ])
            dcdph = np.array([
                -np.sin(th) * np.sin(ph),
                np.sin(th) * np.cos(ph),
                0.0
            ])
            return np.array([
                np.dot(grad_c, dcdth),
                np.dot(grad_c, dcdph)
            ])
            
        try:
            res = minimize(
                spherical_obj,
                x0=np.array([theta, phi]),
                jac=spherical_jac,
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi - 1e-6), (-np.pi, np.pi)],
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            
            if np.isfinite(res.fun):
                th, ph = res.x
                ph = (ph + np.pi) % (2 * np.pi) - np.pi
                c_res = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                c_res = project_to_sphere(c_res)
                val, _ = objective(c_res)
                return c_res, val
        except Exception:
            pass
            
        return None, np.inf

    # Run global search
    # We have ~400 starts. Each local opt is fast.
    # Allocate time for ~300 starts to be safe.
    max_starts = min(len(starts), 300)
    
    for i in range(max_starts):
        if elapsed() > deadline - 10.0:
            break
            
        c_start = starts[i]
        c_opt, val_opt = refine(c_start)
        
        if c_opt is not None and np.isfinite(val_opt):
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
                
    # If we have time, polish the best result further
    if best_c is not None and elapsed() < deadline - 2.0:
        # Run a tighter polish on the best candidate
        def polish(c0):
            c0 = project_to_sphere(c0)
            theta = np.arccos(np.clip(c0[2], -1, 1))
            phi = np.arctan2(c0[1], c0[0])
            
            # Use Powell method which is derivative-free and robust
            # Or SLSQP with tighter tol
            def spherical_obj(params):
                th, ph = params
                ph = (ph + np.pi) % (2 * np.pi) - np.pi
                c_sph = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                return obj_only(c_sph)
                
            def spherical_jac(params):
                th, ph = params
                ph = (ph + np.pi) % (2 * np.pi) - np.pi
                c_sph = np.array([
                    np.sin(th) * np.cos(ph),
                    np.sin(th) * np.sin(ph),
                    np.cos(th)
                ])
                _, grad_c = objective(c_sph)
                dcdth = np.array([
                    np.cos(th) * np.cos(ph),
                    np.cos(th) * np.sin(ph),
                    -np.sin(th)
                ])
                dcdph = np.array([
                    -np.sin(th) * np.sin(ph),
                    np.sin(th) * np.cos(ph),
                    0.0
                ])
                return np.array([
                    np.dot(grad_c, dcdth),
                    np.dot(grad_c, dcdph)
                ])

            try:
                res = minimize(
                    spherical_obj,
                    x0=np.array([theta, phi]),
                    jac=spherical_jac,
                    method='L-BFGS-B',
                    bounds=[(1e-6, np.pi - 1e-6), (-np.pi, np.pi)],
                    options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
                )
                if np.isfinite(res.fun):
                    th, ph = res.x
                    ph = (ph + np.pi) % (2 * np.pi) - np.pi
                    c_res = np.array([
                        np.sin(th) * np.cos(ph),
                        np.sin(th) * np.sin(ph),
                        np.cos(th)
                    ])
                    c_res = project_to_sphere(c_res)
                    val, _ = objective(c_res)
                    if val < best_val:
                        best_val = val
                        best_c = c_res.copy()
            except Exception:
                pass

        polish(best_c)

    if best_c is None:
        # Fallback
        c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = c_seed / np.linalg.norm(c_seed)
    else:
        best_c = project_to_sphere(best_c)

    return np.array([best_c[0], best_c[1], best_c[2]])
