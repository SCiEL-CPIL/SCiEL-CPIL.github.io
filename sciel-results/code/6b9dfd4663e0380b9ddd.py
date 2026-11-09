# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's Method with restarts.
    """
    x0 = np.array(x0, dtype=float)
    
    def eval_F(x):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return None, np.inf
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, np.inf

    def compute_jacobian(x):
        h = 1e-5
        J = np.zeros((2, 2))
        Fx, _ = eval_F(x)
        if Fx is None:
            return None
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus, _ = eval_F(x + e)
            Fx_minus, _ = eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J

    def broyden_step(x, J, Fx):
        """
        Perform one Broyden step with line search.
        Returns new x, new J, new Fx, new norm, success_flag
        """
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, add small regularization or return failure
            return x, J, Fx, np.linalg.norm(Fx), False

        # Line search
        alpha = 1.0
        current_norm = np.linalg.norm(Fx)
        x_new = x.copy()
        Fx_new = None
        norm_new = current_norm
        
        for _ in range(20):
            x_new = x + alpha * dx
            Fx_new, norm_new = eval_F(x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue
            if norm_new < current_norm:
                break
            alpha *= 0.5
            
        if Fx_new is None or norm_new >= current_norm:
            return x, J, Fx, current_norm, False

        # Update Jacobian (Broyden's Good Method)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-15:
            J_new = J + np.outer(y - J @ s, s) / s_norm_sq
        else:
            J_new = J
            
        return x_new, J_new, Fx_new, norm_new, True

    def run_broyden(x_start, max_iter=50):
        x = x_start.copy()
        Fx, norm_F = eval_F(x)
        if Fx is None:
            return x, norm_F, False
        if norm_F <= 1e-8:
            return x, norm_F, True
            
        J = compute_jacobian(x)
        if J is None:
            return x, norm_F, False
            
        for _ in range(max_iter):
            x, J, Fx, norm_F, success = broyden_step(x, J, Fx)
            if not success:
                break
            if norm_F <= 1e-8:
                return x, norm_F, True
                
        return x, norm_F, norm_F <= 1e-8

    # Strategy: Try Broyden from x0, then from random points if it fails.
    # We track the best point found to return if all fail.
    best_x = x0
    best_norm = np.inf
    
    # 1. Try from x0
    x_sol, norm_sol, success = run_broyden(x0)
    if norm_sol < best_norm:
        best_x = x_sol
        best_norm = norm_sol
    if success:
        return x_sol

    # 2. If failed, try a few random restarts
    # Generate random points in a expanding sphere around x0
    rng = np.random.default_rng(42) # Fixed seed for reproducibility? No, task says "hidden", but random is fine.
    # Actually, deterministic is better for stability. Let's use fixed offsets.
    offsets = [
        [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1],
        [1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0],
        [10.0, 0], [-10.0, 0], [0, 10.0], [0, -10.0],
        [0.1, 0.1], [-0.1, -0.1], [1.0, 1.0], [-1.0, -1.0]
    ]
    
    for off in offsets:
        x_start = x0 + np.array(off)
        x_sol, norm_sol, success = run_broyden(x_start)
        if norm_sol < best_norm:
            best_x = x_sol
            best_norm = norm_sol
        if success:
            return x_sol
            
    # 3. If all failed, return best point
    return best_x
# EVOLVE-BLOCK-END
