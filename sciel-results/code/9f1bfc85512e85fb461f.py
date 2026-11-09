"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fd_jacobian(F, x, h=1e-7):
    """Compute finite-difference Jacobian of F at x."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def _damped_newton(F, x, max_iter=40, tol=1e-12, deadline=None):
    """Damped Newton's method with backtracking line search."""
    x = np.array(x, dtype=float)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        Fx = F(x)
        nrm = np.linalg.norm(Fx)
        if nrm <= tol:
            return x, nrm
        J = _fd_jacobian(F, x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        # Backtracking line search
        step = 1.0
        for _ in range(30):
            x_new = x + step * dx
            Fx_new = F(x_new)
            nrm_new = np.linalg.norm(Fx_new)
            if nrm_new < nrm:
                x = x_new
                break
            step *= 0.5
        else:
            break
    return x, np.linalg.norm(F(x))


def _nelder_mead(F, x0, max_iter=200, tol=1e-12, deadline=None):
    """Nelder-Mead minimization of ||F(x)||^2."""
    def objective(x):
        return np.sum(F(x) ** 2)
    
    x = np.array(x0, dtype=float)
    # Initialize simplex
    x0_arr = x.copy()
    simplex = [x0_arr.copy()]
    for i in range(2):
        xi = x0_arr.copy()
        xi[i] += 0.1 * (1.0 + abs(x0_arr[i]))
        simplex.append(xi)
    
    # Sort by objective
    simplex.sort(key=lambda v: objective(v))
    best = simplex[0].copy()
    best_val = objective(best)
    
    alpha, gamma, rho = 0.5, 2.0, 0.5  # reflection, expansion, contraction
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if best_val <= tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < objective(simplex[0]):
            # Try expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        elif fr < objective(simplex[-2]):
            simplex[-1] = xr
        else:
            # Contraction
            if fr < objective(simplex[-1]):
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < max(fr, objective(simplex[-1])):
                simplex[-1] = xc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
        
        # Re-sort
        simplex.sort(key=lambda v: objective(v))
        if objective(simplex[0]) < best_val:
            best = simplex[0].copy()
            best_val = objective(simplex[0])
    
    return best, np.sqrt(best_val)


def find_root_system(F, x0):
    """Find root of F: R^2 -> R^2 using hybrid Newton + Nelder-Mead."""
    x = np.array(x0, dtype=float)
    deadline = time.time() + 100.0  # Safety margin before 150s limit
    
    # Check initial point
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Phase 1: Damped Newton from x0
    x_newton, nrm_newton = _damped_newton(F, x, max_iter=30, deadline=deadline)
    if nrm_newton <= 1e-8:
        return x_newton
    
    # Phase 2: Nelder-Mead from x0 and from Newton result
    best_x, best_nrm = x_newton, nrm_newton
    
    x_nm1, nrm_nm1 = _nelder_mead(F, x, max_iter=100, deadline=deadline)
    if nrm_nm1 < best_nrm:
        best_x, best_nrm = x_nm1, nrm_nm1
    
    if best_nrm <= 1e-8:
        return best_x
    
    x_nm2, nrm_nm2 = _nelder_mead(F, x_newton, max_iter=100, deadline=deadline)
    if nrm_nm2 < best_nrm:
        best_x, best_nrm = x_nm2, nrm_nm2
    
    if best_nrm <= 1e-8:
        return best_x
    
    # Phase 3: Polish with Newton from best point
    x_polish, nrm_polish = _damped_newton(F, best_x, max_iter=20, deadline=deadline)
    if nrm_polish < best_nrm:
        best_x, best_nrm = x_polish, nrm_polish
    
    return best_x
# EVOLVE-BLOCK-END
