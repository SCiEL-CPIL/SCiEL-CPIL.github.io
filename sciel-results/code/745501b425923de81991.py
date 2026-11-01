"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(math.log(d) for d in dist[iu])
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _hessian(pts):
    """
    Compute the Euclidean Hessian of the logarithmic energy.
    H is a (3N, 3N) matrix.
    """
    n = pts.shape[0]
    H = np.zeros((3*n, 3*n))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    
    # We need d^4 for the second term
    dist4 = dist2 * dist2
    
    # Iterate over pairs
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal block H_ii is sum of interactions with all k != i
                # We will accumulate this
                pass
            
            d2 = dist2[i, j]
            d4 = dist4[i, j]
            diff_ij = diff[i, j]
            
            # Interaction term between i and j
            # H_ij = I/d2 - 2 * outer(diff_ij, diff_ij) / d4
            # Note: The energy is -sum log(d).
            # d/dd log(d) = 1/d. d2/dd2 log(d) = -1/d2.
            # Let's verify the sign.
            # E = - sum log(d_ij)
            # dE/dx_i = - sum_j (1/d_ij) * (1/d_ij) * (x_i - x_j) = - sum_j (x_i - x_j)/d_ij^2. Correct.
            # d2E/dx_i dx_j (i!=j) = - [ d/dx_j ( (x_i - x_j)/d_ij^2 ) ]
            # = - [ -I/d_ij^2 + (x_i - x_j) * d/dx_j (1/d_ij^2) ]
            # d/dx_j (1/d_ij^2) = -2 (x_j - x_i) / d_ij^4 = 2 (x_i - x_j) / d_ij^4
            # So term is - [ -I/d2 + (x_i - x_j) (2 (x_i - x_j) / d4) ]
            # = I/d2 - 2 (x_i - x_j)(x_i - x_j)^T / d4.
            
            block = np.eye(3) / d2 - 2.0 * np.outer(diff_ij, diff_ij) / d4
            
            H[3*i:3*i+3, 3*j:3*j+3] += block
            if i != j:
                H[3*j:3*j+3, 3*i:3*i+3] += block
            else:
                # Diagonal block gets the contribution from all k != i
                # We are accumulating into H[3*i:3*i+3, 3*i:3*i+3]
                # Wait, the loop structure above adds block to H[i,j] and H[j,i].
                # For i=j, we add block to H[i,i].
                # But H_ii should be sum_{k!=i} H_{ik}.
                # The loop iterates i, j.
                # If i=j, we add the interaction of i with i? No, dist2 is 1.0 on diagonal, but we should skip self-interaction in the sum for the Hessian diagonal?
                # Actually, the Hessian diagonal $H_{ii}$ is the sum of the second derivatives with respect to $x_i$ from all terms involving $x_i$.
                # The term $E_{ij}$ contributes to $H_{ii}$ and $H_{jj}$ and $H_{ij}$.
                # My loop adds `block` to `H[i,j]`.
                # If `i==j`, `block` is computed with `d2=1.0` (dummy). This is wrong.
                # We must skip `i==j` in the pair loop and handle the diagonal accumulation separately.
                pass

    # Let's rewrite the Hessian calculation more carefully.
    H = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(i+1, n):
            d2 = dist2[i, j]
            d4 = d2 * d2
            diff_ij = diff[i, j]
            
            block = np.eye(3) / d2 - 2.0 * np.outer(diff_ij, diff_ij) / d4
            
            H[3*i:3*i+3, 3*j:3*j+3] += block
            H[3*j:3*j+3, 3*i:3*i+3] += block
            H[3*i:3*i+3, 3*i:3*i+3] += block
            H[3*j:3*j+3, 3*j:3*j+3] += block
            
    return H

def _sphere_energy_and_grad(x_flat):
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, direction, step_size):
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-18,
                'gtol': 1e-16,
                'maxfun': 30000,
                'maxls': 50
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x, best_f

def _newton_polish(pts, deadline, max_iter=50):
    """
    Projected Newton method for high-precision polishing.
    """
    n = pts.shape[0]
    f, g = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Compute Hessian
        H = _hessian(pts)
        
        # Solve H d_amb = -g
        # Add small regularization for stability
        H_reg = H + 1e-10 * np.eye(3*n)
        
        try:
            d_amb = np.linalg.solve(H_reg, -g.flatten())
        except np.linalg.LinAlgError:
            break
            
        d_amb = d_amb.reshape(n, 3)
        
        # Project to tangent space
        proj = np.einsum('ij,ij->i', d_amb, pts)
        d_tan = d_amb - proj[:, np.newaxis] * pts
        
        # Line search
        step_size = 1.0
        found = False
        for _ in range(20):
            new_pts = _retract(pts, d_tan, step_size)
            new_f, _ = _energy_and_grad(new_pts)
            if new_f < f:
                pts = new_pts
                f = new_f
                found = True
                break
            step_size *= 0.5
            
        if not found:
            break
            
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
            
        # Update gradient
        f, g = _energy_and_grad(pts)
        
    return pts, f

def _md_step(pts, deadline, n_steps=50, temp_init=0.01, temp_decay=0.9):
    """
    Simple Langevin Dynamics step to escape local minima.
    """
    f, g = _energy_and_grad(pts)
    temp = temp_init
    rng = np.random.default_rng(42)
    
    for _ in range(n_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Force is -grad
        force = -g
        
        # Velocity update (simple Euler)
        # v = v + force * dt + noise
        # x = x + v * dt
        # We don't track velocity, just do a random walk biased by force
        
        noise = rng.standard_normal(pts.shape) * temp
        step = force * 0.01 + noise
        
        new_pts = _retract(pts, step, 1.0)
        
        new_f, new_g = _energy_and_grad(new_pts)
        
        # Metropolis accept/reject (simplified: always accept if better, else accept with prob exp(-(new_f-f)/temp))
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
        else:
            delta_f = new_f - f
            if delta_f < 100 * temp: # Avoid overflow
                prob = math.exp(-delta_f / temp)
                if rng.random() < prob:
                    pts = new_pts
                    f = new_f
                    g = new_g
                    
        temp *= temp_decay
        
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-14, deadline=None, m=30):
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(60):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
            if new_f <= f + 1e-6 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-14:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # 2. Random points
    for seed_offset in range(5):
        rng_local = np.random.default_rng(42 + seed_offset)
        random_pts = rng_local.standard_normal((N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / np.maximum(norms, 1e-15)
        initializations.append(random_pts)
        
    # 3. Perturbed Fibonacci
    rng_pert = np.random.default_rng(123)
    for pert_scale in [0.01, 0.05, 0.1]:
        perturbed = fib_pts + rng_pert.standard_normal((N, 3)) * pert_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        initializations.append(perturbed)
        
    # 4. Spherical harmonic-like init (simplified)
    rng_sh = np.random.default_rng(99)
    sh_pts = rng_sh.standard_normal((N, 3))
    sh_norms = np.linalg.norm(sh_pts, axis=1, keepdims=True)
    sh_pts = sh_pts / np.maximum(sh_norms, 1e-15)
    initializations.append(sh_pts)
    
    # Sort by energy
    init_energies = []
    for init_pts in initializations:
        e, _ = _energy_and_grad(init_pts)
        init_energies.append((e, init_pts))
    init_energies.sort(key=lambda x: x[0])
    
    best_x = None
    best_f = float('inf')
    
    n_inits = len(init_energies)
    
    for idx, (init_e, init_pts) in enumerate(init_energies):
        if time.time() > deadline - 20.0:
            break
            
        # 1. MD Step to escape local minima
        pts_md, f_md = _md_step(init_pts, deadline, n_steps=20, temp_init=0.05)
        
        # 2. L-BFGS
        x0 = pts_md.flatten()
        x_opt, f_opt = _lbfgs_sphere(x0, deadline, maxiter=2000)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()
            
        if idx > 5 and f_opt < -1083.0:
             break

    if best_x is None:
        best_x = init_energies[0][1].flatten()
        best_f, _ = _sphere_energy_and_grad(best_x)
        
    # Final Polish
    if time.time() < deadline - 10.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # Newton Polish
        pts, f = _newton_polish(pts, deadline, max_iter=50)
        
        # Custom L-BFGS for final convergence
        if time.time() < deadline - 2.0:
            pts, f = _lbfgs_sphere_custom(pts, max_iter=200, tol=1e-15, deadline=deadline - 1.0)
            
        best_x = pts.flatten()
        best_f = f
        
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
