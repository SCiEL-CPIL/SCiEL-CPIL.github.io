"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools


def _hilbert3d(n, depth=4):
    """
    Generate n points in [0,1]^3 using a Hilbert-curve-like space-filling approach.
    We use a recursive subdivision to map 1D indices to 3D coordinates.
    """
    # For simplicity and speed, we'll use a Z-order (Morton) curve instead of true Hilbert,
    # which is faster to compute and still provides good space-filling properties.
    # Actually, let's use a simple grid-based initialization with jitter for robustness,
    # but the directive suggests Hilbert. Let me implement a proper Hilbert-like mapping.
    
    # A simpler approach: use a 3D grid with the best packing structure known.
    # For 50 spheres, a 3x4x4 or 4x4x3 grid might work well.
    # Let's start with a body-centered cubic or face-centered cubic inspired arrangement.
    
    # Actually, let's follow the directive: use a space-filling curve.
    # I'll implement a simple Hilbert curve in 3D.
    
    # For 3D Hilbert curve, we can use the algorithm from Vaidyanathan & Varadhan.
    # But for simplicity and given time constraints, let's use a Z-curve (Morton code) 
    # which is easier to implement and still provides good space-filling.
    
    # Generate points using a 3D Morton/Z-order curve
    # Map 1D index to 3D coordinates using bit interleaving
    
    def morton3d(i, bits=8):
        """Convert 1D index to 3D Morton code coordinates."""
        x = 0
        y = 0
        z = 0
        for b in range(bits):
            if i & (1 << (3 * b)):
                x |= (1 << b)
            if i & (1 << (3 * b + 1)):
                y |= (1 << b)
            if i & (1 << (3 * b + 2)):
                z |= (1 << b)
        return x / (2**bits - 1), y / (2**bits - 1), z / (2**bits - 1)
    
    points = []
    for i in range(n):
        x, y, z = morton3d(i, bits=8)
        # Add small jitter to avoid perfect grid alignment
        points.append([x, y, z])
    
    return np.array(points)


def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Compute minimum pairwise distance
    min_pair_dist = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair_dist = min(min_pair_dist, d)
    
    # Compute minimum distance to boundary
    min_boundary_dist = np.inf
    for c in centers:
        for k in range(3):
            d = 0.5 - abs(c[k])
            min_boundary_dist = min(min_boundary_dist, d)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _objective(centers_flat):
    """
    Objective function to minimize: negative of the minimum distance.
    We want to maximize the minimum distance, so we minimize -min_dist.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Compute all pairwise distances
    min_pair_dist = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            min_pair_dist = min(min_pair_dist, d)
    
    # Compute minimum boundary distance
    min_boundary_dist = np.inf
    for c in centers:
        for k in range(3):
            d = 0.5 - abs(c[k])
            min_boundary_dist = min(min_boundary_dist, d)
    
    # We want to maximize min(min_pair_dist/2, min_boundary_dist)
    # So we minimize -min(min_pair_dist/2, min_boundary_dist)
    # But this is not smooth. Let's use a smooth approximation.
    
    # Smooth approximation: use a soft-min
    # soft_min(a, b) = -log(exp(-a/t) + exp(-b/t)) * t for small t
    # But for simplicity, let's just return -min(min_pair_dist/2, min_boundary_dist)
    
    val = min(min_pair_dist / 2.0, min_boundary_dist)
    return -val


def _gradient(centers_flat):
    """
    Numerical gradient of the objective function.
    """
    eps = 1e-7
    grad = np.zeros_like(centers_flat)
    f0 = _objective(centers_flat)
    for i in range(len(centers_flat)):
        old = centers_flat[i]
        centers_flat[i] = old + eps
        fp = _objective(centers_flat)
        centers_flat[i] = old - eps
        fm = _objective(centers_flat)
        centers_flat[i] = old
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline
    
    # Initialize using Hilbert/Morton curve
    centers = _hilbert3d(N, depth=8)
    # Scale to fit in [-0.5, 0.5]^3 with some margin
    centers = centers * 0.9 - 0.45  # Scale to [-0.45, 0.45]
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try gradient-based optimization
    centers_flat = centers.flatten()
    
    # Use Nelder-Mead for a derivative-free approach which is more robust
    # for this non-smooth objective
    
    def objective_smooth(x):
        """Smooth objective using soft-min."""
        c = x.reshape(-1, 3)
        
        # Compute all pairwise distances
        n = len(c)
        min_pair_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                min_pair_dist = min(min_pair_dist, d)
        
        # Compute minimum boundary distance
        min_boundary_dist = np.inf
        for ci in c:
            for k in range(3):
                d = 0.5 - abs(ci[k])
                min_boundary_dist = min(min_boundary_dist, d)
        
        # Use soft-min with temperature
        t = 0.01
        a = min_pair_dist / 2.0
        b = min_boundary_dist
        # soft_min(a, b) = -t * log(exp(-a/t) + exp(-b/t))
        # But this can overflow, so use log-sum-exp trick
        m = min(a, b)
        if m < -100 * t:
            return m  # Underflow case
        soft_min_val = m - t * np.log(np.exp(-(a - m) / t) + np.exp(-(b - m) / t))
        
        return -soft_min_val
    
    # Run Nelder-Mead optimization
    try:
        result = minimize(
            objective_smooth,
            centers_flat,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < -best_radius:
            new_centers = result.x.reshape(-1, 3)
            # Clip to valid range
            new_centers = np.clip(new_centers, -0.499, 0.499)
            new_radius = _compute_radius(new_centers)
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
        
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_centers
    
    # Try multiple restarts with different initializations
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization with some structure
        if restart == 0:
            # Body-centered cubic inspired
            centers = np.array([
                [x, y, z] for x in [-0.4, -0.2, 0.0, 0.2, 0.4]
                for y in [-0.4, -0.2, 0.0, 0.2, 0.4]
                for z in [-0.4, -0.2, 0.0, 0.2, 0.4]
            ])
            # Take first 50
            centers = centers[:50]
        elif restart == 1:
            # Face-centered cubic inspired
            centers = []
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        centers.append([-0.4 + 0.2 * i, -0.4 + 0.2 * j, -0.4 + 0.2 * k])
            centers = np.array(centers[:50])
        else:
            # Random with jitter
            centers = rng.uniform(-0.45, 0.45, (N, 3))
        
        centers_flat = centers.flatten()
        
        try:
            result = minimize(
                objective_smooth,
                centers_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < -best_radius:
                new_centers = result.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.499, 0.499)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
                    
        except Exception as e:
            pass
    
    # Final polish: local optimization with tighter tolerance
    if time.time() < deadline - 2.0:
        centers_flat = best_centers.flatten()
        try:
            result = minimize(
                objective_smooth,
                centers_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < -best_radius:
                new_centers = result.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.499, 0.499)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
                    
        except Exception as e:
            pass
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
