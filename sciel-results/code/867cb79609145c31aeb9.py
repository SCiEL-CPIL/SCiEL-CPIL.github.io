"""LJ38 cluster optimization using distance-space optimization with Gram-matrix projection."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 150.0
START_TIME = time.time()


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)


def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    """
    scale = 1.0 / math.sqrt(2.0)
    
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def relax_ljfg(positions):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-6})
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def distances_to_coordinates(dists, max_iter=1000, tol=1e-6):
    """
    Convert pairwise distances to 3D coordinates using Gram matrix projection.
    dists: (N, N) matrix of pairwise distances.
    Returns: (N, 3) array of coordinates, or None if failed.
    """
    n = len(dists)
    # Ensure symmetry
    dists = 0.5 * (dists + dists.T)
    np.fill_diagonal(dists, 0.0)
    
    # Check for negative distances
    if np.any(dists < -tol):
        return None
    
    # Clip small negative values to zero
    dists = np.maximum(dists, 0.0)
    
    # r2 matrix
    r2 = dists ** 2
    
    # Gram matrix: G_ij = (||x_i||^2 + ||x_j||^2 - r2_ij) / 2
    # We need to find ||x_i||^2 values such that G is PSD and rank <= 3
    
    # Initialize ||x_i||^2 = 0 (origin-centered)
    norm_sq = np.zeros(n)
    
    for iteration in range(max_iter):
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        # Compute Gram matrix
        G = (norm_sq[:, np.newaxis] + norm_sq[np.newaxis, :] - r2) / 2.0
        np.fill_diagonal(G, norm_sq)
        
        # Check if G is PSD and rank <= 3
        try:
            eigvals, eigvecs = np.linalg.eigh(G)
        except np.linalg.LinAlgError:
            return None
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Check rank
        if eigvals[3] > tol:
            # Need to adjust norm_sq to reduce rank
            # Use Newton-Raphson: we want eigvals[3] = 0
            # Gradient of eigvals[3] w.r.t. norm_sq
            # For simplicity, use a heuristic: if rank > 3, reduce the largest norms
            # Actually, let's use a different approach: project G to rank 3
            
            # Project G to rank 3
            G_rank3 = eigvecs[:, :3] @ np.diag(eigvals[:3]) @ eigvecs[:, :3].T
            G_rank3 = 0.5 * (G_rank3 + G_rank3.T)
            np.fill_diagonal(G_rank3, 0.0)
            
            # Update norm_sq from G_rank3
            # G_ii = norm_sq_i, so norm_sq_i = G_ii
            new_norm_sq = np.diag(G_rank3).copy()
            new_norm_sq = np.maximum(new_norm_sq, 0.0)
            
            # Damped update
            norm_sq = 0.5 * norm_sq + 0.5 * new_norm_sq
            continue
        
        # Rank is <= 3, check if PSD
        if eigvals[0] < -tol:
            # Not PSD, need to adjust
            # Increase the most negative eigenvalue
            # This is complex, so let's just return the best we have
            break
        
        # Compute coordinates
        coords = eigvecs[:, :3] * np.sqrt(np.maximum(eigvals[:3], 0.0))
        
        # Verify the distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        r2_check = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2_check, 0.0)
        
        error = np.max(np.abs(np.sqrt(np.maximum(r2_check, 0.0)) - dists))
        if error < tol:
            return coords
        
        # If error is too large, adjust norm_sq
        # The error in G leads to error in coords
        # Let's try to fix by updating norm_sq based on the actual norms
        actual_norm_sq = np.sum(coords * coords, axis=1)
        norm_sq = 0.5 * norm_sq + 0.5 * actual_norm_sq
    
    # Final attempt
    G = (norm_sq[:, np.newaxis] + norm_sq[np.newaxis, :] - r2) / 2.0
    np.fill_diagonal(G, norm_sq)
    
    try:
        eigvals, eigvecs = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        return None
    
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    coords = eigvecs[:, :3] * np.sqrt(np.maximum(eigvals[:3], 0.0))
    return coords


def energy_from_distances(dists):
    """Compute energy from pairwise distances."""
    # Check for invalid distances
    if np.any(dists < 0.5):
        return 1e20
    
    # Convert to coordinates
    coords = distances_to_coordinates(dists)
    if coords is None:
        return 1e20
    
    if not check_validity(coords):
        return 1e20
    
    return lj_energy_vectorized(coords)


def construct():
    """Main entry point: use distance-space optimization starting from FCC truncated octahedron."""
    global START_TIME
    START_TIME = time.time()
    
    # Start from the known global minimum structure
    initial = make_fcc_truncated_octahedron()
    
    # First, relax it to get a good starting point
    relaxed_initial = relax_ljfg(initial)
    if not check_validity(relaxed_initial):
        relaxed_initial = initial
    
    best_positions = relaxed_initial
    best_energy = lj_energy_vectorized(relaxed_initial)
    
    # Get initial distances
    diff = relaxed_initial[:, np.newaxis, :] - relaxed_initial[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 0.0)
    initial_dists = np.sqrt(np.maximum(r2, 0.0))
    initial_dists = 0.5 * (initial_dists + initial_dists.T)
    
    # Optimize in distance space
    # We only need to optimize the upper triangle
    iu = np.triu_indices(N, k=1)
    n_vars = len(iu[0])
    
    def dists_to_upper(d):
        dists = np.zeros((N, N))
        dists[iu] = d
        dists[iu[1], iu[0]] = d
        return dists
    
    def energy_from_upper(d):
        dists = dists_to_upper(d)
        return energy_from_distances(dists)
    
    # Initial upper triangle
    initial_upper = initial_dists[iu]
    
    # Optimize
    try:
        result = minimize(energy_from_upper, initial_upper, method='L-BFGS-B',
                         options={'maxiter': 50, 'gtol': 1e-4, 'ftol': 1e-8})
        
        if result.success or result.fun < best_energy:
            opt_dists = dists_to_upper(result.x)
            opt_coords = distances_to_coordinates(opt_dists)
            
            if opt_coords is not None and check_validity(opt_coords):
                opt_energy = lj_energy_vectorized(opt_coords)
                if opt_energy < best_energy:
                    best_energy = opt_energy
                    best_positions = opt_coords
    except Exception:
        pass
    
    # Final relaxation
    final_relaxed = relax_ljfg(best_positions)
    if check_validity(final_relaxed):
        final_energy = lj_energy_vectorized(final_relaxed)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_relaxed
    
    # Ensure validity
    if not check_validity(best_positions):
        best_positions = initial
    
    return best_positions

# EVOLVE-BLOCK-END
