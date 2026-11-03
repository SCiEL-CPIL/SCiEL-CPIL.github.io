import numpy as np
import time
import math
from scipy.special import lpmv

N = 150
DEADLINE = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    # 1. Generate a high-resolution grid of points on the sphere for density representation
    # Using a Fibonacci spiral for uniform coverage of the grid itself
    M = 20000  # Number of grid points
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # 2. Compute Spherical Harmonic basis up to L_max
    # We use real spherical harmonics for real-valued density
    L_max = 40  # Resolution of the density representation
    # Precompute Y_lm(θ, φ) for the grid
    # θ = arccos(z), φ = atan2(y, x)
    theta = np.arccos(np.clip(z_grid, -1.0, 1.0))
    phi = np.arctan2(grid_points[:, 1], grid_points[:, 0])
    
    # Store Y_lm values: shape (M, (L_max+1)^2)
    Y_lm = np.zeros((M, (L_max + 1) ** 2))
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # Compute Y_lm(theta, phi)
            # P_l^m(cos(theta)) * e^{i m phi}
            # For real harmonics:
            if m == 0:
                P_lm = lpmv(0, l, z_grid)
                Y_val = np.sqrt((2 * l + 1) / (4 * np.pi)) * P_lm
                Y_lm[:, idx] = Y_val
            elif m > 0:
                P_lm = lpmv(m, l, z_grid)
                # Normalization factor: sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
                # lpmv includes the Condon-Shortley phase and factorial ratio in scipy? 
                # scipy.special.lpmv(m, l, x) returns P_l^m(x) with Condon-Shortley phase.
                # We need to multiply by sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
                # Actually, standard normalized Y_lm uses:
                # Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos theta) * e^{i m phi}
                # For m>0, real part: sqrt(2) * ... * cos(m phi)
                # For m<0, real part: sqrt(2) * ... * sin(|m| phi) (with sign adjustments)
                
                # Let's use a simpler, more robust approach: 
                # Compute complex Y_lm and take real/imag parts appropriately, or just use real basis directly.
                # For efficiency and stability, let's stick to complex basis and optimize real coefficients 
                # by pairing m and -m. 
                # However, for simplicity in this implementation, I will use the complex basis 
                # and optimize the real and imaginary parts separately, ensuring the density is real.
                
                # Actually, let's just compute the complex Y_lm and store real and imaginary parts.
                # But we need a real density. 
                # Let's use the real spherical harmonics directly.
                
                # Real spherical harmonics:
                # Y_l^m (m>0) = sqrt(2) * (-1)^m * sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos theta) * cos(m phi)
                # Y_l^{-m} (m>0) = sqrt(2) * sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos theta) * sin(m phi)
                
                # scipy lpmv gives P_l^m with Condon-Shortley (-1)^m included.
                # So for m>0:
                # Y_l^m (real) = sqrt(2) * sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m * cos(m phi)
                # Note: The (-1)^m is already in lpmv. The standard real harmonic definition 
                # sometimes absorbs this. Let's check consistency.
                # Standard: Y_l^m = N_l^m P_l^m(cos theta) e^{i m phi}
                # Real: Y_l^m (m>0) = sqrt(2) N_l^m P_l^m cos(m phi)
                # Imag: Y_l^m (m>0) = sqrt(2) N_l^m P_l^m sin(m phi)
                
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.cos(m * phi)
                Y_lm[:, idx] = Y_real
            else:
                m_pos = -m
                P_lm = lpmv(m_pos, l, z_grid)
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m_pos) / math.factorial(l + m_pos))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.sin(m_pos * phi)
                Y_lm[:, idx] = Y_real
            idx += 1
            
    # 3. Initialize density coefficients
    # Start with a slightly perturbed uniform density (only l=0 term non-zero)
    # Add small random noise to break symmetry
    coeffs = np.zeros((L_max + 1) ** 2)
    coeffs[0] = 1.0  # Uniform component
    # Add small random high-frequency noise to break symmetry and help optimization
    coeffs[1:] += np.random.normal(0, 0.01, size=(L_max + 1) ** 2 - 1)
    
    # 4. Gradient Flow Optimization
    # Energy E = sum_{i<j} log(1/|x_i - x_j|) approximated by density interaction
    # For a continuous density rho, the energy is E[rho] = 1/2 ∫∫ rho(x) rho(y) log(1/|x-y|) dx dy
    # Gradient with respect to coefficients c_k:
    # ∂E/∂c_k = ∫ Y_k(x) (∫ Y(x) log(1/|x-y|) rho(y) dy) dx
    # This is expensive to compute directly. 
    # Alternative: Compute the potential V(x) = ∫ Y(x) log(1/|x-y|) rho(y) dy on the grid.
    # Then the gradient of the energy with respect to the density value at grid point i is V(x_i).
    # Project this gradient onto the spherical harmonic basis to get the coefficient gradient.
    
    # We will use a simple gradient descent on the coefficients.
    # Step size needs to be small.
    
    # Precompute the interaction matrix? No, M=20000 is too big for MxM.
    # We can compute the potential on the grid efficiently using FFT? 
    # The kernel log(1/|x-y|) is not translation invariant on the sphere in a simple way, 
    # but it can be expanded in spherical harmonics:
    # log(1/|x-y|) = sum_{l=0}^∞ A_l P_l(cos gamma)
    # where A_l = 1/(l(l+1)) for l>=1? No, the expansion of log(1/|x-y|) is known.
    # Actually, it's easier to just compute the potential numerically on the grid using 
    # a fast multipole method or just approximate the gradient.
    
    # Given the time constraint and complexity, let's use a simpler heuristic:
    # The gradient flow for the logarithmic energy on the sphere tends to equalize the density.
    # We can simulate this by iterating:
    # 1. Compute density on grid: rho = Y_lm @ coeffs
    # 2. Compute "force" or potential: The repulsive force from log kernel.
    #    For small steps, the gradient of the energy with respect to moving a point x_i 
    #    is proportional to sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * log(1/|x_i - x_j|)? 
    #    No, that's for the discrete case.
    
    # Let's try a different structural approach: 
    # Use the discrete points but optimize them using a "soft" assignment to a high-res grid.
    # This is effectively what the density approach does.
    
    # Let's implement a direct gradient descent on the 150 points, but using a 
    # multi-scale approach: 
    # 1. Start with Fibonacci.
    # 2. Use L-BFGS-B for a few steps to get out of the immediate spiral structure.
    # 3. Then use a "smoothing" step: blur the point distribution, re-discretize, 
    #    and optimize again. This mimics the spectral flow.
    
    # Actually, the prompt asks for a spectral gradient flow. Let's stick to that.
    # I will compute the gradient of the energy with respect to the coefficients 
    # using the spherical harmonic expansion of the kernel.
    
    # Expansion: log(1/|x-y|) = sum_{l=0}^∞ c_l P_l(cos gamma)
    # The coefficients c_l are known. 
    # c_0 = 0 (log singularity is integrable, but the constant term is 0 in the expansion? 
    # Actually, the expansion is: 
    # log(1/|x-y|) = sum_{l=1}^∞ (1/(l(l+1))) P_l(cos gamma) * (2l+1)/2 ? No.
    # Let's look up the expansion.
    # The potential of a uniform charge on the sphere is constant. 
    # The expansion of log(1/|x-y|) in terms of Legendre polynomials is:
    # log(1/|x-y|) = sum_{l=0}^∞ A_l P_l(cos gamma)
    # A_l = 2 * ∫_0^1 t^{l+1} log(1/t) dt = 2 / (l+1)^2 ? 
    # Let's check. ∫ t^{l+1} log(1/t) dt = 1/(l+1)^2.
    # So A_l = 2/(l+1)^2 for l>=0? 
    # Wait, the expansion of 1/|x-y| is sum (2l+1) P_l. 
    # Integrating this gives log(1/|x-y|).
    # ∫ (2l+1) P_l dt = ...
    # Actually, there is a known series: 
    # log(1/|x-y|) = sum_{l=0}^∞ \frac{1}{l+1} P_l(cos gamma) ? No.
    
    # Let's use a numerical gradient on the coefficients instead.
    # E(coeffs) = 0.5 * sum_{i,j} rho_i rho_j log(1/|x_i - x_j|)
    # where rho_i = Y(x_i) @ coeffs.
    # This requires an MxM matrix for the log kernel, which is 20000x20000 = 400M elements. 
    # Too big for memory (3.2 GB).
    
    # Alternative: Use a smaller grid M=5000. 25M elements = 200MB. Feasible.
    # Let's reduce M to 5000.
    
    M = 5000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Recompute Y_lm for smaller grid
    Y_lm = np.zeros((M, (L_max + 1) ** 2))
    theta = np.arccos(np.clip(z_grid, -1.0, 1.0))
    phi = np.arctan2(grid_points[:, 1], grid_points[:, 0])
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            if m == 0:
                P_lm = lpmv(0, l, z_grid)
                Y_val = np.sqrt((2 * l + 1) / (4 * np.pi)) * P_lm
                Y_lm[:, idx] = Y_val
            elif m > 0:
                P_lm = lpmv(m, l, z_grid)
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.cos(m * phi)
                Y_lm[:, idx] = Y_real
            else:
                m_pos = -m
                P_lm = lpmv(m_pos, l, z_grid)
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m_pos) / math.factorial(l + m_pos))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.sin(m_pos * phi)
                Y_lm[:, idx] = Y_real
            idx += 1
            
    # Compute the log kernel matrix K on the grid
    # K[i, j] = log(1 / |x_i - x_j|)
    # Use broadcasting
    # dist = np.linalg.norm(grid_points[:, None, :] - grid_points[None, :, :], axis=2)
    # This is 5000x5000x3 = 75M floats = 600MB. Might be too much.
    # Let's compute in chunks or use a faster method.
    # Or just use M=2000. 4M elements = 32MB. Safe.
    
    M = 2000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    Y_lm = np.zeros((M, (L_max + 1) ** 2))
    theta = np.arccos(np.clip(z_grid, -1.0, 1.0))
    phi = np.arctan2(grid_points[:, 1], grid_points[:, 0])
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            if m == 0:
                P_lm = lpmv(0, l, z_grid)
                Y_val = np.sqrt((2 * l + 1) / (4 * np.pi)) * P_lm
                Y_lm[:, idx] = Y_val
            elif m > 0:
                P_lm = lpmv(m, l, z_grid)
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.cos(m * phi)
                Y_lm[:, idx] = Y_real
            else:
                m_pos = -m
                P_lm = lpmv(m_pos, l, z_grid)
                N_lm = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m_pos) / math.factorial(l + m_pos))
                Y_real = np.sqrt(2) * N_lm * P_lm * np.sin(m_pos * phi)
                Y_lm[:, idx] = Y_real
            idx += 1
            
    # Compute K
    diff = grid_points[:, None, :] - grid_points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid log(0)
    dist[dist < 1e-10] = 1e-10
    K = np.log(1.0 / dist)
    
    # Initialize coefficients
    coeffs = np.zeros((L_max + 1) ** 2)
    coeffs[0] = 1.0
    coeffs[1:] += np.random.normal(0, 0.01, size=(L_max + 1) ** 2 - 1)
    
    # Gradient descent
    lr = 0.01
    best_energy = np.inf
    best_coeffs = coeffs.copy()
    
    for step in range(500):
        if time.time() - start_time > DEADLINE:
            break
            
        # Density on grid
        rho = Y_lm @ coeffs
        
        # Potential V = K @ rho
        V = K @ rho
        
        # Gradient of E = 0.5 rho^T K rho with respect to coeffs
        # dE/dcoeffs = Y_lm^T V
        grad = Y_lm.T @ V
        
        # Update
        coeffs -= lr * grad
        
        # Normalize density to have unit integral? 
        # The l=0 mode should be kept constant to preserve total "mass"
        coeffs[0] = 1.0
        
        # Compute energy
        E = 0.5 * np.dot(rho, V)
        if E < best_energy:
            best_energy = E
            best_coeffs = coeffs.copy()
            
    # Extract 150 points from the best density
    rho_best = Y_lm @ best_coeffs
    # Find the 150 points with highest density
    # To avoid clustering, we can use a greedy selection:
    # Pick the max, remove its neighborhood, pick next max, etc.
    
    # Simpler: Just pick the top 150 indices. 
    # But they might be close. 
    # Let's use a simple repulsion-based selection from the grid points.
    
    selected_indices = []
    remaining = np.arange(M)
    for _ in range(N):
        if len(remaining) == 0:
            break
        # Pick the index with highest density in remaining
        max_idx_local = np.argmax(rho_best[remaining])
        max_idx_global = remaining[max_idx_local]
        selected_indices.append(max_idx_global)
        
        # Remove neighbors within a small angle
        # Distance on sphere
        center = grid_points[max_idx_global]
        dists = np.linalg.norm(grid_points[remaining] - center, axis=1)
        # Keep only those far enough
        mask = dists > 0.1  # ~5 degrees
        remaining = remaining[mask]
        
    points = grid_points[selected_indices]
    
    # Final polish with L-BFGS on the discrete points
    # This is crucial for the final score.
    from scipy.optimize import minimize
    
    def log_energy_and_grad(pts):
        pts = pts.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, None]
        
        # Compute distances
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        
        # Energy
        energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                energy += np.log(1.0 / dist[i, j])
        
        # Gradient
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # d/dx_i log(1/|x_i - x_j|) = - (x_i - x_j) / |x_i - x_j|^2 * log(1/|x_i - x_j|)?
                # No, d/dx log(1/|x-y|) = - (x-y)/|x-y|^2
                # Wait, the energy is sum log(1/d). 
                # d/dx_i log(1/|x_i - x_j|) = -1 * (x_i - x_j) / |x_i - x_j|^2
                # So grad_i = sum_{j != i} - (x_i - x_j) / |x_i - x_j|^2
                # But this is for the discrete energy. 
                # The gradient on the sphere is the projection of this to the tangent space.
                
        # This double loop is slow. Let's use vectorization.
        # grad = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # This is the standard Coulomb force, but without the log factor?
        # Wait, the derivative of log(1/d) is -1/d * (1/d^2) * d? No.
        # d/dx log(1/|x-y|) = - (x-y) / |x-y|^2.
        # So the force is the same as the Coulomb force for 1/d? No, 1/d force is (x-y)/|x-y|^3.
        # Here it's (x-y)/|x-y|^2.
        
        # Let's just use numerical gradient or a simpler approximation for the polish.
        # Given time, let's just return the grid-selected points.
        return energy, grad
        
    # Since the gradient computation above is incomplete/slow, and we are close to deadline,
    # we will skip the L-BFGS polish if time is tight, or do a very simple one.
    
    # Let's do a simple gradient descent on the points for 100 steps.
    for step in range(100):
        if time.time() - start_time > DEADLINE:
            break
        pts = points.copy()
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    continue
                # grad_i += - (x_i - x_j) / d^2
                grad[i] -= diff / (d * d)
                
        # Project to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
            
        # Update
        points -= 0.01 * grad
        # Normalize
        points /= np.linalg.norm(points, axis=1)[:, None]
        
    return points
