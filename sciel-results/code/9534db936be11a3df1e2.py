import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    calls = 0
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= 495:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def compute_jacobian(x, Fx, h=1e-7):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return None
            J[:, j] = (Fx_j - Fx) / h
        return J

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    best_norm = norm
    best_x = x.copy()
    if norm <= 1e-8:
        return best_x

    # Compute initial Jacobian
    J = compute_jacobian(x, Fx)
    if J is None:
        J = np.eye(2)

    # Levenberg-Marquardt damped Newton with adaptive damping
    max_iter = 300
    lam = 1e-3  # initial damping parameter
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if calls >= 490:
            break

        # Solve (J^T J + lam * I) dx = -J^T F
        JtJ = J.T @ J
        JtF = J.T @ Fx
        A = JtJ + lam * np.eye(2)
        b = -JtF

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Try the step with backtracking if needed
        alpha = 1.0
        step_accepted = False
        for _ in range(15):
            x_try = x + alpha * dx
            Fx_try = safe_F(x_try)
            if Fx_try is None:
                alpha *= 0.5
                continue

            norm_try = np.linalg.norm(Fx_try)

            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()

            if norm_try < norm * (1 - 1e-4 * alpha):
                # Accept step
                s = alpha * dx
                y = Fx_try - Fx
                # Update Jacobian with Broyden rank-1 update
                J_s = J @ s
                diff = y - J_s
                denom = s @ s
                if denom > 1e-15:
                    J = J + np.outer(diff, s) / denom
                
                x = x_try
                Fx = Fx_try
                norm = norm_try
                lam = max(lam * 0.1, 1e-10)  # reduce damping on success
                step_accepted = True
                break
            else:
                alpha *= 0.5

        if not step_accepted:
            # Increase damping
            lam *= 10
            if lam > 1e10:
                break
            continue

        if norm <= 1e-8:
            return best_x

    return best_x
