import numpy as np
import math
import sys
from scipy.optimize import minimize

N = 150

def construct():
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_on_sphere(n, rng):
        v = rng.normal(0, 1, size=(n, 3))
        norms = np.linalg.norm(v, axis=1)
        v /= norms[:, None]
        return v
        
    def compute_energy_flat(flat):
        # flat is shape (N*3,)
        P = flat.reshape(N, 3)
        # Renormalize to handle numerical drift during optimization
        norms = np.linalg.norm(P, axis=1)
        P = P / norms[:, None]
        
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        
        # log(1/d) = -log(d)
        # Using math.fsum is requested in the task description for the final score, 
        # but for optimization speed, np.sum is usually fine. 
        # However, to be safe and consistent with "minimize", we compute the raw energy.
        # The task says "Score: ... sum ... with math.fsum". 
        # For the optimizer, we want a fast, differentiable function.
        # The difference between np.sum and math.fsum is negligible for optimization steps 
        # unless we are extremely close to the minimum, but np.sum is much faster.
        # We will use np.sum for the gradient steps.
        return -np.sum(np.log(d_iu))

    def compute_grad_flat(flat):
        P = flat.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1)
        P = P / norms[:, None]
        
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        # Euclidean gradient
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        # Project to tangent space
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        
        return G_riemann.flatten()

    def riemannian_exp_flat(P, V_flat):
        # P is (N, 3), V_flat is (N*3,)
        V = V_flat.reshape(N, 3)
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        # Renormalize
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new.flatten()

    def riemannian_retract_flat(P, V_flat):
        # Retraction: P + V, then normalize
        V = V_flat.reshape(N, 3)
        P_new = P + V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new.flatten()

    # We will use a custom L-BFGS implementation on the sphere or use scipy's minimize with a custom metric?
    # scipy.optimize.minimize does not natively support Riemannian manifolds.
    # We can use the "L-BFGS-B" method on the flat coordinates, but we must handle the constraint ||x_i||=1.
    # A better approach for Riemannian optimization is to use a library like Geoopt, but it's not available.
    # We can implement a simple Riemannian L-BFGS.
    
    class RiemannianLBFGS:
        def __init__(self, x0, m=10):
            self.x = x0
            self.m = m
            self.s = [] # directions
            self.y = [] # gradient differences
            self.rho = [] # 1/(y^T s)
            self.g = self._grad(self.x)
            self.E = self._func(self.x)
            
        def _func(self, x):
            return compute_energy_flat(x)
            
        def _grad(self, x):
            return compute_grad_flat(x)
            
        def _exp(self, x, d):
            # x is flat, d is flat
            P = x.reshape(N, 3)
            # Normalize P just in case
            norms = np.linalg.norm(P, axis=1)
            P = P / norms[:, None]
            return riemannian_exp_flat(P, d)
            
        def _retract(self, x, d):
            P = x.reshape(N, 3)
            norms = np.linalg.norm(P, axis=1)
            P = P / norms[:, None]
            return riemannian_retract_flat(P, d)

        def _compute_alpha(self, g):
            # Two-loop recursion for L-BFGS
            # g is current gradient
            q = g.copy()
            for i in range(len(self.s) - 1, -1, -1):
                alpha_i = self.rho[i] * np.dot(self.s[i], q)
                q -= alpha_i * self.y[i]
            
            # Initial Hessian approximation H0 = 1
            r = q
            
            for i in range(len(self.s)):
                beta_i = self.rho[i] * np.dot(self.y[i], r)
                r += (alpha_i - beta_i) * self.s[i]
                
            return r

        def step(self):
            # Line search
            d = self._compute_alpha(self.g)
            # Ensure we are moving in a descent direction
            if np.dot(self.g, d) > 0:
                d = -self.g
                # Reset history if direction is bad?
                # Or just use steepest descent for this step
                pass
            
            # Backtracking line search
            alpha = 1.0
            c1 = 1e-4
            E_cur = self.E
            g_cur = self.g
            slope = np.dot(g_cur, d)
            
            if slope >= 0:
                # Should not happen if d is descent direction
                return False
            
            for _ in range(20):
                x_new = self._exp(self.x, alpha * d)
                E_new = self._func(x_new)
                
                if E_new <= E_cur + c1 * alpha * slope:
                    # Accept step
                    s = alpha * d
                    g_new = self._grad(x_new)
                    y = g_new - g_cur
                    
                    # Update history
                    sy = np.dot(s, y)
                    if sy > 1e-10:
                        self.s.append(s)
                        self.y.append(y)
                        self.rho.append(1.0 / sy)
                        if len(self.s) > self.m:
                            self.s.pop(0)
                            self.y.pop(0)
                            self.rho.pop(0)
                    
                    self.x = x_new
                    self.g = g_new
                    self.E = E_new
                    return True
                else:
                    alpha *= 0.5
            return False

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    # 1. Fibonacci spiral
    initializations.append(fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci
    P_fib = fibonacci_spiral(N)
    noise = rng.normal(0, 0.1, size=(N, 3))
    P_pert = P_fib + noise
    P_pert /= np.linalg.norm(P_pert, axis=1)[:, None]
    initializations.append(P_pert)
    
    # 3. Random initializations
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))

    for P_init in initializations:
        P = P_init.copy()
        P_flat = P.flatten()
        
        # Check validity
        if not np.all(np.isfinite(P_flat)):
            continue
            
        E_init = compute_energy_flat(P_flat)
        if E_init < best_energy:
            best_energy = E_init
            P_best = P.copy()
            
        # Run Riemannian L-BFGS
        opt = RiemannianLBFGS(P_flat)
        
        max_iters = 1000
        for i in range(max_iters):
            if not opt.step():
                break
                
            if opt.E < best_energy:
                best_energy = opt.E
                P_best = opt.x.reshape(N, 3)
                
            # Check convergence
            if np.linalg.norm(opt.g) < 1e-12:
                break

    # Final cleanup: ensure points are on sphere
    if P_best is not None:
        norms = np.linalg.norm(P_best, axis=1)
        P_best /= norms[:, None]
        
    return P_best
