import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180 # 180s limit, keep margin
    
    # Hermite polynomials values
    # We need H0, H4, H8, H12
    # We can compute them using recurrence or explicit formulas
    # H0 = 1
    # H2 = 4x^2 - 2
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13824x^2 + 4032
    # H12 = ...
    
    # Better to use scipy.special.eval_hermite
    from scipy.special import eval_hermite
    
    # Gauss-Hermite quadrature
    # We need to integrate f(x) e^{-x^2}
    # roots_hermite(n) returns nodes x_i and weights w_i for \int f(x) e^{-x^2} dx
    n_quad = 100
    x_nodes, w_nodes = roots_hermite(n_quad)
    
    # Precompute H values at nodes
    H0_nodes = np.ones(n_quad)
    H4_nodes = eval_hermite(4, x_nodes)
    H8_nodes = eval_hermite(8, x_nodes)
    H12_nodes = eval_hermite(12, x_nodes)
    
    # Weights for the integrals
    # Integral I = \sum w_i f(x_i)
    
    # H(0) values for constraint
    H0_0 = 1.0
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
        P = c0 * H0_nodes + c4 * H4_nodes + c8 * H8_nodes + c12 * H12_nodes
        
        # ||g||_2^2 = \int P^2 e^{-x^2} dx
        norm2_sq = np.sum(w_nodes * P**2)
        
        # ||g||_4^4 = \int P^4 e^{-2x^2} dx
        # Note: roots_hermite is for e^{-x^2}.
        # We need \int P^4 e^{-2x^2} dx.
        # This requires a different quadrature or transformation.
        # Let y = sqrt(2) x. Then x = y/sqrt(2), dx = dy/sqrt(2).
        # \int P(x)^4 e^{-2x^2} dx = \int P(y/sqrt(2))^4 e^{-y^2} \frac{dy}{\sqrt{2}}
        # We can just use a fine grid for this term, or compute moments.
        # Given time constraints, a numerical integration with a fixed grid might be safer/faster to implement correctly.
        # Let's use a simple grid for the 4th power term to avoid quadrature mismatch issues.
        
        # Actually, let's stick to Gauss-Hermite for e^{-x^2} terms.
        # For e^{-2x^2}, we can use scaled nodes.
        # Or just compute using the same nodes? No, that's inaccurate.
        
        # Let's use a fixed grid for all integrals to be safe and simple.
        # Grid from -10 to 10.
        # This is slower but robust.
        
        # Let's re-evaluate.
        # We can compute \int P^4 e^{-2x^2} dx using Gauss-Hermite with scaled variables.
        # But let's just use a simple numpy grid for the objective function.
        # 1000 points is fast enough.
        
        # To avoid recomputing grid every time, we can define it outside.
        # But P depends on coeffs.
        
        # Let's use the quadrature nodes for the e^{-x^2} part and a separate logic for e^{-2x^2}.
        # Actually, \int P^4 e^{-2x^2} dx can be approximated by \sum w_i' P(x_i')^4 where x_i' are nodes for e^{-2x^2}.
        # Nodes for e^{-2x^2}: scale roots_hermite by 1/sqrt(2). Weights scale by 1/sqrt(2).
        
        x_nodes_2, w_nodes_2 = roots_hermite(n_quad)
        x_nodes_2 = x_nodes_2 / np.sqrt(2)
        w_nodes_2 = w_nodes_2 / np.sqrt(2)
        
        P2 = c0 * eval_hermite(0, x_nodes_2) + c4 * eval_hermite(4, x_nodes_2) + c8 * eval_hermite(8, x_nodes_2) + c12 * eval_hermite(12, x_nodes_2)
        norm4_4 = np.sum(w_nodes_2 * P2**4)
        
        # Fourier side
        # \hat{g}(k) = \sqrt{2\pi} e^{-k^2/2} \sum c_n i^n H_n(k)
        # |\hat{g}(k)|^4 = (2\pi)^2 e^{-4k^2} |\hat{P}(k)|^4
        # \|\hat{g}\|_4^4 = (2\pi)^2 \int e^{-4k^2} |\hat{P}(k)|^4 dk
        
        # Use nodes for e^{-4k^2}. Scale by 1/2.
        x_nodes_4, w_nodes_4 = roots_hermite(n_quad)
        x_nodes_4 = x_nodes_4 / 2.0
        w_nodes_4 = w_nodes_4 / 2.0
        
        # H_n(k) values
        H0_k = np.ones(n_quad)
        H4_k = eval_hermite(4, x_nodes_4)
        H8_k = eval_hermite(8, x_nodes_4)
        H12_k = eval_hermite(12, x_nodes_4)
        
        # \hat{P}(k) = c0 H0 + c4 i^4 H4 + c8 i^8 H8 + c12 i^12 H12
        # i^4 = 1, i^8 = 1, i^12 = -1
        P_hat = c0 * H0_k + c4 * 1 * H4_k + c8 * 1 * H8_k + c12 * (-1) * H12_k
        
        norm4_hat_4 = (2 * np.pi)**2 * np.sum(w_nodes_4 * np.abs(P_hat)**4)
        
        # Ratio R = (||g||_4 * ||\hat{g}||_4) / ||g||_2^2
        # R = (norm4_4^(1/4) * norm4_hat_4^(1/4)) / norm2_sq
        
        if norm2_sq < 1e-10:
            return 1e10
            
        R = (norm4_4**0.25 * norm4_hat_4**0.25) / norm2_sq
        return R

    # Initial guess: Seed values
    # Seed: c0=-1.13, c4=0.04, c8=0.0003
    # We can scale them. Since R is scale invariant, we can set c0=1.
    # But let's start with the seed values to be close.
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Optimize
    # Use Nelder-Mead or L-BFGS-B. L-BFGS-B is faster for smooth functions.
    # We need to ensure it doesn't take too long.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        best_coeffs = res.x
        best_val = res.fun
    except Exception:
        best_coeffs = x0
        best_val = objective(x0)
        
    # Check time
    if time.time() > deadline - 10:
         # Return best found so far
         pass
    else:
        # Maybe try a few random restarts?
        # The landscape might have local minima.
        # Let's do 5 random restarts if time permits.
        best_val_global = best_val
        best_coeffs_global = best_coeffs
        
        for i in range(5):
            if time.time() > deadline - 20:
                break
            # Random start near seed
            noise = np.random.normal(0, 0.1, 3)
            x_start = x0 + noise
            try:
                res_r = minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res_r.fun < best_val_global:
                    best_val_global = res_r.fun
                    best_coeffs_global = res_r.x
            except Exception:
                pass
        
        best_coeffs = best_coeffs_global
        best_val = best_val_global

    return best_coeffs
