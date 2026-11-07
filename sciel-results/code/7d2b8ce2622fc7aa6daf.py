"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J

    def norm_f(x):
        Fx = F(x)
        return np.linalg.norm(Fx), Fx

    # Initial evaluation
    Fx = F(x)
    nrm = np.linalg.norm(Fx)
    best_norm = nrm
    best_x = x.copy()

    if nrm <= 1e-8:
        return x

    # Initialize Jacobian
    J = compute_jacobian(x, Fx)

    # Broyden's method with restarts
    max_iter = 200
    for it in range(max_iter):
        if time_left() < 5.0:
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Restart with new Jacobian
            J = compute_jacobian(x, Fx)
            continue

        # Damping: limit step size if too large
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm

        x_new = x + dx
        Fx_new = F(x_new)
        nrm_new = np.linalg.norm(Fx_new)

        if nrm_new < best_norm:
            best_norm = nrm_new
            best_x = x_new.copy()

        if nrm_new <= 1e-8:
            return x_new

        # Broyden rank-1 update
        dF = Fx_new - Fx
        dx_actual = x_new - x
        J = J + (dF - J @ dx_actual) @ dx_actual.T / (dx_actual @ dx_actual)

        x = x_new
        Fx = Fx_new

        # Periodic re-computation of Jacobian for robustness
        if it % 10 == 9:
            J = compute_jacobian(x, Fx)

    # Final polish: Newton steps with tighter tolerance
    for _ in range(20):
        if time_left() < 2.0:
            break
        Fx = F(x)
        nrm = np.linalg.norm(Fx)
        if nrm <= 1e-12:
            break
        J = compute_jacobian(x, Fx)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        x = x + dx

    Fx_final = F(x)
    if np.linalg.norm(Fx_final) < best_norm:
        best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
