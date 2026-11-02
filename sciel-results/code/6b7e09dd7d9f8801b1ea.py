import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    # 210s limit. Use 190s deadline to be safe.
    deadline = 190.0 

    # Pre-compute Gauss-Hermite nodes and weights
    # Degree of g is 12, g^2 is degree 24.
    # N=64 is sufficient for high precision.
    N_points = 64
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(N_points)
    
    # Pre-compute powers of x_nodes for efficiency
    x = x_nodes
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    x3 = x2 * x
    x5 = x4 * x
    x7 = x6 * x
    x9 = x8 * x
    x11 = x10 * x

    # Pre-compute Hermite polynomials and their derivatives at nodes
    H0 = np.ones_like(x)
    H4 = x4 - 6*x2 + 3
    H8 = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12 = x12 - 66*x10 + 1260*x8 - 9240*x6 + 23100*x4 - 27720*x2 + 10395
    
    dH0 = np.zeros_like(x)
    dH4 = 4*x3 - 12*x
    dH8 = 8*x7 - 168*x5 + 840*x3 - 840*x
    dH12 = 12*x11 - 660*x9 + 10080*x7 - 55440*x5 + 92400*x3 - 55440*x

    # Coefficients for H12 are fixed by g(0)=0 => c0 + 3c4 + 105c8 + 10395c12 = 0
    # c12 = -(c0 + 3c4 + 105c8) / 10395
    c12_const = np.array([-1.0, -3.0, -105.0]) / 10395.0

    def eval_g_and_deriv(coeffs):
        """
        Evaluates g(x) and g'(x) at the Gauss-Hermite nodes.
        coeffs: (c0, c4, c8)
        Returns: g_vals, gp_vals
        """
        c0, c4, c8 = coeffs
        c12 = np.dot(coeffs, c12_const)
        
        g = c0*H0 + c4*H4 + c8*H8 + c12*H12
        gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
        
        return g, gp

    def compute_score_and_grad(coeffs):
        """
        Computes the score (upper bound) and its analytic gradient.
        """
        if time.time() > deadline - 2:
            return np.inf, np.zeros(3)
            
        try:
            c0, c4, c8 = coeffs
            c12 = np.dot(coeffs, c12_const)
            
            # Evaluate g and gp
            g = c0*H0 + c4*H4 + c8*H8 + c12*H12
            gp = c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12
            
            # Integrals
            # I0 = sum(w * g^2)
            I0 = np.sum(w_nodes * g**2)
            if I0 <= 1e-30 or not np.isfinite(I0):
                return np.inf, np.zeros(3)
                
            # Ix2 = sum(w * x^2 * g^2)
            Ix2 = np.sum(w_nodes * x2 * g**2)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf, np.zeros(3)
                
            # Ideriv = sum(w * (gp - x*g)^2)
            deriv_term = gp - x * g
            Ideriv = np.sum(w_nodes * deriv_term**2)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf, np.zeros(3)
                
            # Score
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0
            score_sq = A_f_sq * A_gf_sq
            
            if score_sq <= 0 or not np.isfinite(score_sq):
                return np.inf, np.zeros(3)
                
            score = np.sqrt(score_sq)
            
            # Analytic Gradient
            # We minimize J = sqrt(K) where K = Ix2 * Ideriv / (4*pi^2 * I0^2)
            # dJ/dc = (1/(2*J)) * dK/dc
            # K = C * Ix2 * Ideriv * I0^-2, C = 1/(4*pi^2)
            # dK/dc = C * ( (dIx2/dc) * Ideriv * I0^-2 + Ix2 * (dIderiv/dc) * I0^-2 - Ix2 * Ideriv * 2 * I0^-3 * (dI0/dc) )
            
            # Gradients of I0, Ix2, Ideriv with respect to coeffs (c0, c4, c8)
            # I0 = sum(w * g^2) => dI0/dc_j = 2 * sum(w * g * dH_j)
            # Ix2 = sum(w * x^2 * g^2) => dIx2/dc_j = 2 * sum(w * x^2 * g * dH_j)
            # Ideriv = sum(w * (gp - x*g)^2)
            # Let h = gp - x*g. dh/dc_j = d(gp_j) - x*(dH_j)
            # dIderiv/dc_j = 2 * sum(w * h * dh/dc_j)
            
            # Pre-compute weighted sums for efficiency
            # Note: c12 depends on c0, c4, c8.
            # dH_j in the gradient sums refers to the basis function H_j (for j=0,4,8) 
            # but we must account for the chain rule through c12?
            # Actually, it's easier to treat g = sum c_i H_i where c12 is a function of c0,c4,c8.
            # g = c0(H0 + c12_0 H12) + c4(H4 + c12_4 H12) + c8(H8 + c12_8 H12)
            # Let B0 = H0 + c12_const[0] H12
            # Let B4 = H4 + c12_const[1] H12
            # Let B8 = H8 + c12_const[2] H12
            # Then g = c0 B0 + c4 B4 + c8 B8
            # Similarly for gp:
            # dB0 = dH0 + c12_const[0] dH12
            # dB4 = dH4 + c12_const[1] dH12
            # dB8 = dH8 + c12_const[2] dH12
            
            B0 = H0 + c12_const[0] * H12
            B4 = H4 + c12_const[1] * H12
            B8 = H8 + c12_const[2] * H12
            
            dB0 = dH0 + c12_const[0] * dH12
            dB4 = dH4 + c12_const[1] * dH12
            dB8 = dH8 + c12_const[2] * dH12
            
            # Gradients for I0
            dI0_0 = 2 * np.sum(w_nodes * g * B0)
            dI0_4 = 2 * np.sum(w_nodes * g * B4)
            dI0_8 = 2 * np.sum(w_nodes * g * B8)
            
            # Gradients for Ix2
            dIx2_0 = 2 * np.sum(w_nodes * x2 * g * B0)
            dIx2_4 = 2 * np.sum(w_nodes * x2 * g * B4)
            dIx2_8 = 2 * np.sum(w_nodes * x2 * g * B8)
            
            # Gradients for Ideriv
            # h = gp - x*g
            # dh/dc0 = dB0 - x*B0
            # dh/dc4 = dB4 - x*B4
            # dh/dc8 = dB8 - x*B8
            dh_0 = dB0 - x * B0
            dh_4 = dB4 - x * B4
            dh_8 = dB8 - x * B8
            
            dIderiv_0 = 2 * np.sum(w_nodes * deriv_term * dh_0)
            dIderiv_4 = 2 * np.sum(w_nodes * deriv_term * dh_4)
            dIderiv_8 = 2 * np.sum(w_nodes * deriv_term * dh_8)
            
            C = 1.0 / (4.0 * np.pi**2)
            
            # dK/dc_j = C * ( dIx2_j * Ideriv / I0^2 + Ix2 * dIderiv_j / I0^2 - 2 * Ix2 * Ideriv * dI0_j / I0^3 )
            inv_I0_sq = 1.0 / (I0 * I0)
            inv_I0_cu = 1.0 / (I0 * I0 * I0)
            
            term1 = Ideriv * inv_I0_sq
            term2 = Ix2 * inv_I0_sq
            term3 = -2.0 * Ix2 * Ideriv * inv_I0_cu
            
            dK_0 = C * (dIx2_0 * term1 + dIderiv_0 * term2 + dI0_0 * term3)
            dK_4 = C * (dIx2_4 * term1 + dIderiv_4 * term2 + dI0_4 * term3)
            dK_8 = C * (dIx2_8 * term1 + dIderiv_8 * term2 + dI0_8 * term3)
            
            grad_score = 0.5 / score * np.array([dK_0, dK_4, dK_8])
            
            return score, grad_score
            
        except Exception:
            return np.inf, np.zeros(3)

    # Initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score, _ = compute_score_and_grad(x0)
    best_coeffs = x0.copy()
    
    # 1. Global Search: Differential Evolution
    # DE is robust and doesn't require gradients.
    # We use a moderate number of iterations to find a good basin.
    if time.time() < deadline - 30:
        try:
            # Bounds: The coefficients shouldn't be too large or the function blows up.
            # c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003
            bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
            
            res_de = opt.differential_evolution(
                lambda x: compute_score_and_grad(x)[0],
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=False, # We will polish manually with analytic grad
                updating='immediate'
            )
            
            if res_de.fun < best_score and np.isfinite(res_de.fun):
                best_score = res_de.fun
                best_coeffs = res_de.x.copy()
        except Exception:
            pass

    # 2. Local Refinement: L-BFGS-B with Analytic Gradient
    # This will converge much faster and more accurately than numerical methods.
    def objective_wrapper(x):
        val, grad = compute_score_and_grad(x)
        if not np.isfinite(val):
            return 1e10, np.zeros(3)
        return val, grad

    if time.time() < deadline - 5:
        try:
            res_lbfgs = opt.minimize(
                objective_wrapper,
                best_coeffs,
                method='L-BFGS-B',
                jac=True, # We provide the gradient
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
            )
            if res_lbfgs.fun < best_score and np.isfinite(res_lbfgs.fun):
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.copy()
        except Exception:
            pass

    # 3. Additional Random Restarts with Global + Local
    np.random.seed(43)
    for i in range(20):
        if time.time() > deadline - 10:
            break
            
        # Generate a random point in the neighborhood or global space
        # Mix of local perturbations and global jumps
        if i < 10:
            # Local perturbation
            noise = np.random.normal(0, 0.1, 3)
            x_try = best_coeffs + noise
        else:
            # Global random
            x_try = np.array([np.random.uniform(-5, 5), np.random.uniform(-1, 1), np.random.uniform(-0.1, 0.1)])
            
        # Quick local polish
        try:
            res_r = opt.minimize(
                objective_wrapper,
                x_try,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_r.fun < best_score and np.isfinite(res_r.fun):
                best_score = res_r.fun
                best_coeffs = res_r.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
