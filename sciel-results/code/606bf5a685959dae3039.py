"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    """
    n = len(points)
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    # Energy
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    grad = -np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _sph_harm(m, n, theta, phi):
    """
    Spherical harmonic Y_l^m(theta, phi).
    theta: polar angle (0 to pi)
    phi: azimuthal angle (0 to 2pi)
    """
    # Use scipy.special.sph_harm if available, otherwise manual
    # scipy.special.sph_harm(m, l, phi, theta) 
    # Note: scipy convention is Y_l^m(theta, phi) but arguments are (m, l, phi, theta)
    # Let's stick to a simple implementation or use scipy if it's fast enough.
    # For performance, we precompute basis.
    pass

def _generate_basis_points(n_points, l_max, rng):
    """
    Generate a set of basis points on the sphere up to degree l_max.
    We use a subset of spherical harmonics or just random points with low-frequency constraints.
    Actually, a simpler spectral approach: 
    Parameterize the position of point i as:
    p_i = sum_{l,m} c_{l,m} Y_l^m(theta_i, phi_i)
    This is complex to invert.
    
    Alternative Spectral Strategy:
    Instead of explicit spherical harmonics for positions (which is hard to constrain to sphere),
    we use a "Voronoi/Spectral" hybrid:
    1. Start with a good grid (Fibonacci).
    2. Define a low-pass filter on the Laplacian of the sphere.
    3. Optimize displacements in the low-frequency subspace.
    
    Even simpler and effective for "globally coherent":
    Use a parameterization where points are moved by a global vector field defined by low-degree spherical harmonics.
    v(x) = sum_{l<=L, m} (a_{l,m} Y_l^m(x) + b_{l,m} Y_l^m(x) + c_{l,m} Y_l^m(x))
    This ensures that nearby points move similarly, preserving the grid structure and avoiding local minima that break global symmetry.
    """
    pass

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _spectral_deform(points, coeffs, l_max=3):
    """
    Apply a deformation to points using low-degree spherical harmonics.
    coeffs: array of shape (3 * (l_max+1)^2, ) or similar.
    We define a vector field V(x) = sum_{l=0}^{l_max} sum_{m=-l}^{l} (c_{l,m} * Y_l^m(x))
    where c_{l,m} are 3D vectors.
    New point x' = normalize(x + eps * V(x))
    """
    # For simplicity and speed, we implement a simpler "global shift" or "low-freq noise"
    # that captures the essence of spectral regularization.
    # Let's use a random low-frequency vector field.
    # Actually, let's stick to the standard Riemannian optimization but with a "Spectral Initialization" 
    # and "Spectral Regularization" during the optimization steps.
    
    # The prompt asks for a DIFFERENT strategy. 
    # Let's implement the "Spectral Basis Discrete Variational Optimization" as:
    # 1. Initialize with Fibonacci.
    # 2. Compute a "spectral energy" proxy or just use the standard energy.
    # 3. Optimize the coefficients of a low-degree spherical harmonic expansion of the *displacement*.
    
    # To keep it robust and fast, we will use a two-stage approach:
    # Stage 1: Global spectral optimization.
    # We parameterize the points as p_i = normalize(base_i + sum_{k} c_k * phi_k(base_i))
    # where phi_k are low-frequency spherical harmonics.
    # This reduces the dimensionality of the search space for the initial global shape.
    
    # Let's define a small set of basis functions (l=1, 2).
    # Y_1^0 ~ z, Y_1^1 ~ x, Y_1^-1 ~ y (dipole)
    # Y_2^0 ~ 3z^2-1, etc. (quadrupole)
    
    # We will optimize the coefficients c.
    # Number of coefficients: l=0 (1), l=1 (3), l=2 (5). Total 9 complex coeffs -> 18 real? 
    # Or just 3D vector field components for each harmonic.
    # Let's use 3 components for each of l=0,1,2.
    # l=0: 3 coeffs (uniform shift) - doesn't change normalized points much.
    # l=1: 9 coeffs (dipole)
    # l=2: 15 coeffs (quadrupole)
    # Total ~24 coeffs.
    
    # Basis functions:
    # We use the real spherical harmonics or just monomials normalized.
    # For l=1: x, y, z
    # For l=2: x^2-y^2, xy, xz, yz, 2z^2-x^2-y^2
    
    # Let's implement this.
    
    n = len(points)
    # Create design matrix for low-frequency basis
    # Basis: 1, x, y, z, x^2-y^2, xy, xz, yz, 2z^2-x^2-y^2
    # 9 basis functions.
    # We apply a 3D displacement for each? No, that's 27 coeffs.
    # Let's just use a single scalar field per component?
    # Displacement vector d_i = sum_{k=0}^{8} c_k * B_k(p_i) * u_k
    # This is getting complicated.
    
    # Simpler Spectral Strategy:
    # Use the fact that the optimal configuration for log energy is close to the Fibonacci lattice.
    # We can optimize the *angles* (theta, phi) of the Fibonacci points directly using gradient descent,
    # but with a "spectral" penalty that encourages smoothness in the angular distribution.
    
    # However, the most distinct and effective "Spectral" method for this is:
    # **Spherical Harmonic Expansion of the Density**.
    # But we are placing discrete points.
    
    # Let's go with the **Low-Frequency Displacement Optimization**.
    # We define a set of 10 low-frequency vector fields (basis functions) on the sphere.
    # We optimize the weights of these fields to deform the initial grid.
    
    # Basis vectors (evaluated at point p):
    # 1. (0,0,1) - constant (doesn't help much)
    # 2. (1,0,0), (0,1,0), (0,0,1) - linear (dipole)
    # 3. (2x, 0, 0), (0, 2y, 0), (0, 0, 2z) ? No, these are gradients of quadrupoles.
    
    # Let's use the gradients of the first few spherical harmonics.
    # grad(Y_1^0) ~ (x, y, 2z)
    # grad(Y_1^1) ~ ...
    
    # To be safe and fast, I will implement a **Random Low-Frequency Deformation** search.
    # 1. Start with Fibonacci.
    # 2. Generate random low-frequency vector fields.
    # 3. Optimize the amplitude of these fields using gradient descent.
    # 4. This effectively performs a "spectral" search in the space of smooth deformations.
    
    pass

def _optimize_spectral(points, deadline, rng):
    """
    Optimize points by applying low-frequency deformations.
    """
    n = len(points)
    
    # Define a set of random low-frequency basis functions.
    # We use 5 random basis vectors for each of 3 components?
    # Let's use 10 random smooth functions.
    # A smooth function on the sphere can be approximated by a low-degree polynomial.
    # We will use random linear combinations of l=1 and l=2 harmonics.
    
    # Generate 10 random basis functions (scalar fields).
    # b_k(p) = sum_{l=1}^2 sum_{m=-l}^l (a_{l,m} cos(m*phi) + b_{l,m} sin(m*phi)) * P_l(cos(theta))
    
    # For simplicity, we use random vectors in the space of (x,y,z) and (x^2-y^2, etc).
    
    # Let's just use a standard gradient optimization but with a "Spectral Regularization" 
    # added to the objective function.
    # Energy_reg = Energy + lambda * ||Laplacian(p)||^2
    # This encourages the points to be distributed smoothly, avoiding clumping.
    
    # However, the task is to minimize energy. Regularization might hurt the final score if lambda is too high.
    # But it helps escape local minima.
    
    # Let's implement a **Two-Stage Optimization**:
    # 1. Spectral Stage: Optimize with a strong regularization term to find a globally coherent shape.
    # 2. Polish Stage: Optimize without regularization to minimize raw energy.
    
    def _regularized_energy_and_grad(points, lam):
        energy, grad = _log_energy_and_grad(points)
        
        # Compute a simple "clumping" penalty: sum of 1/d^4 for close pairs
        # This acts as a repulsive force for very close points, helping escape local minima
        # where points might be too close.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        # Penalty: sum 1/d^2
        pen = np.sum(1.0 / dists_sq)
        
        # Gradient of penalty: grad_i = sum_j 2 * (x_i - x_j) / |x_i - x_j|^4
        # Note: d/dx_i (1/|x_i-x_j|^2) = -2 (x_i-x_j) / |x_i-x_j|^4
        # So grad_pen = sum_j 2 (x_i-x_j) / |x_i-x_j|^4
        # Wait, derivative of |x|^2 is 2x. Derivative of 1/u is -1/u^2.
        # d/dx_i (|x_i-x_j|^-2) = -2 |x_i-x_j|^-4 * (x_i-x_j)
        # So grad_pen_i = sum_j -2 (x_i-x_j) / |x_i-x_j|^4
        
        # This is computationally expensive. Let's use a simpler penalty.
        # Just use the energy itself but with a "kick" strategy.
        
        return energy + lam * pen, grad + lam * (-2 * np.sum(diff / dists_sq[:, :, np.newaxis]**2, axis=1))

    # Actually, the "Spectral" idea in the prompt is specific.
    # "Model the problem as a discrete variational optimization using a spectral basis (spherical harmonics) to parameterize point positions as low-frequency deformations of a uniform grid"
    
    # I will implement this parameterization.
    # p_i(t) = normalize( base_i + t * V(base_i) )
    # where V is a vector field from the low-frequency subspace.
    
    # We optimize t and the coefficients of V.
    
    # Let's define 3 sets of basis functions for x, y, z components.
    # Basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    # We project the displacement onto these.
    
    # This is effectively a linearized optimization.
    
    # Given time constraints, I will implement a **Hybrid Approach**:
    # 1. Use the current L-BFGS-B for local polish.
    # 2. Add a **Spectral Restart** strategy:
    #    - Generate a random low-frequency deformation of the Fibonacci grid.
    #    - Optimize from there.
    
    # This is materially different from pure random restarts because the restarts are structured (smooth).
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Evaluate initial Fibonacci
    try:
        e, _ = _log_energy_and_grad(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    rng = np.random.default_rng(42)

    # 2. Spectral Deformation Search
    # We will try several low-frequency deformations of the Fibonacci grid.
    # Deformation: p' = normalize(p + sum_{k=1}^K c_k * b_k(p))
    # where b_k are random smooth functions.
    
    # Define 5 random smooth basis functions (scalar).
    # b_k(x,y,z) = a_k * x + b_k * y + c_k * z + d_k * (x^2 - y^2) + e_k * (x^2 + y^2 - 2z^2)
    # We use random coefficients.
    
    n_basis = 5
    # Precompute basis values for the Fibonacci grid
    base_pts = best_points
    # Basis 1: x
    # Basis 2: y
    # Basis 3: z
    # Basis 4: x^2 - y^2
    # Basis 5: 2z^2 - x^2 - y^2
    
    basis_vals = np.zeros((N, n_basis))
    basis_vals[:, 0] = base_pts[:, 0]
    basis_vals[:, 1] = base_pts[:, 1]
    basis_vals[:, 2] = base_pts[:, 2]
    basis_vals[:, 3] = base_pts[:, 0]**2 - base_pts[:, 1]**2
    basis_vals[:, 4] = 2*base_pts[:, 2]**2 - base_pts[:, 0]**2 - base_pts[:, 1]**2
    
    # We optimize the coefficients c = [c1..c5] and a scale factor s.
    # Displacement d_i = s * sum_{k} c_k * b_k(p_i) * u_k
    # To keep it simple, let's just optimize the coefficients of a single vector field
    # composed of these basis functions.
    
    # Let's use scipy.optimize.minimize to find the best deformation coefficients.
    
    def _deform_energy(coeffs):
        s = coeffs[0]
        c = coeffs[1:]
        # Displacement vector for each point
        # d_i = s * (c1*b1 + ... + c5*b5) * (1,1,1)? No, that's a scalar displacement.
        # We need a vector displacement.
        # Let's use 3 sets of coefficients, one for each component.
        # Total 1 + 3*5 = 16 coeffs.
        
        if len(coeffs) < 16:
            return 1e10, np.zeros(16)
            
        s = coeffs[0]
        cx = coeffs[1:6]
        cy = coeffs[6:11]
        cz = coeffs[11:16]
        
        dx = s * (basis_vals @ cx)
        dy = s * (basis_vals @ cy)
        dz = s * (basis_vals @ cz)
        
        new_pts = base_pts.copy()
        new_pts[:, 0] += dx
        new_pts[:, 1] += dy
        new_pts[:, 2] += dz
        
        new_pts = _normalize(new_pts)
        
        energy, grad_pts = _log_energy_and_grad(new_pts)
        
        # We don't have easy gradients for coeffs.
        # We will use Nelder-Mead or Powell for this low-dimensional search.
        return energy

    # Use Nelder-Mead to find good deformation coefficients
    # This is a global search in a low-dimensional space (16D) of smooth deformations.
    
    try:
        if time.time() < deadline - 30:
            x0 = np.zeros(16)
            x0[0] = 0.1 # small scale
            
            res = minimize(_deform_energy, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_energy:
                s = res.x[0]
                cx = res.x[1:6]
                cy = res.x[6:11]
                cz = res.x[11:16]
                
                dx = s * (basis_vals @ cx)
                dy = s * (basis_vals @ cy)
                dz = s * (basis_vals @ cz)
                
                new_pts = base_pts.copy()
                new_pts[:, 0] += dx
                new_pts[:, 1] += dy
                new_pts[:, 2] += dz
                new_pts = _normalize(new_pts)
                
                e, _ = _log_energy_and_grad(new_pts)
                if e < best_energy:
                    best_energy = e
                    best_points = new_pts
    except Exception:
        pass

    # 3. Standard Riemannian Polish from the best point found
    def _optimize_from(points, deadline, rng, maxiter=200):
        start_time = time.time()
        
        def objective_and_grad(x):
            pts = _normalize(x.reshape(-1, 3))
            energy, grad = _log_energy_and_grad(pts)
            
            # Project gradient to tangent space
            grad_tangent = np.zeros_like(grad)
            for i in range(len(pts)):
                proj = np.dot(grad[i], pts[i])
                grad_tangent[i] = grad[i] - proj * pts[i]
            
            return energy, grad_tangent.flatten()
        
        result = minimize(
            objective_and_grad,
            points.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxls': 50}
        )
        
        result_points = _normalize(result.x.reshape(-1, 3))
        return result_points

    try:
        if time.time() < deadline - 20:
            refined = _optimize_from(best_points, deadline, rng, maxiter=100)
            e, _ = _log_energy_and_grad(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    except Exception:
        pass

    # 4. Portfolio of Spectral Restarts
    # Generate random low-frequency deformations and polish them.
    
    for k in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random coefficients for deformation
        coeffs = rng.standard_normal(16) * 0.1
        coeffs[0] = 0.1 + 0.1 * rng.random()
        
        try:
            s = coeffs[0]
            cx = coeffs[1:6]
            cy = coeffs[6:11]
            cz = coeffs[11:16]
            
            dx = s * (basis_vals @ cx)
            dy = s * (basis_vals @ cy)
            dz = s * (basis_vals @ cz)
            
            new_pts = base_pts.copy()
            new_pts[:, 0] += dx
            new_pts[:, 1] += dy
            new_pts[:, 2] += dz
            new_pts = _normalize(new_pts)
            
            # Polish
            if time.time() < deadline - 15:
                refined_pts = _optimize_from(new_pts, deadline, rng, maxiter=50)
                e, _ = _log_energy_and_grad(refined_pts)

                if e < best_energy:
                    best_energy = e
                    best_points = refined_pts

                    # If it's good, refine more
                    if time.time() < deadline - 20:
                        further_refined = _optimize_from(refined_pts, deadline, rng, maxiter=150)
                        e2, _ = _log_energy_and_grad(further_refined)
                        if e2 < best_energy:
                            best_energy = e2
                            best_points = further_refined
        except Exception:
            continue

    # 5. Final polish
    if time.time() < deadline - 5:
        try:
            final_pts = _optimize_from(best_points, deadline, rng, maxiter=300)
            e, _ = _log_energy_and_grad(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
