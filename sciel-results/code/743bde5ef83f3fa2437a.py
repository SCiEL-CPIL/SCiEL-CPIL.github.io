"""Physics-based sphere packing optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(centers):
    """Compute the packing score (minimum radius)."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min(arr, beta):
    """Compute soft minimum using log-sum-exp."""
    min_val = np.min(arr)
    return min_val - np.log(np.sum(np.exp(-beta * (arr - min_val)))) / beta

def _energy(centers, beta):
    """
    Energy function: negative of soft-min of constraints.
    We want to maximize min(pairwise_dist/2, wall_dist).
    So we minimize -soft_min of these values.
    """
    centers = np.asarray(centers, dtype=np.float64)
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_terms = np.min(wall_dists, axis=1)
    
    # Combine all constraints
    all_terms = np.concatenate([pair_terms[np.triu_indices_from(pair_terms, k=1)], wall_terms])
    
    # Soft min
    sm = _soft_min(all_terms, beta)
    return -sm

def _energy_grad(centers, beta):
    """
    Gradient of the energy function.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_terms = np.min(wall_dists, axis=1)
    
    # Combine all constraints
    all_terms = np.concatenate([pair_terms[np.triu_indices_from(pair_terms, k=1)], wall_terms])
    
    # Soft min weights
    min_val = np.min(all_terms)
    weights = np.exp(-beta * (all_terms - min_val))
    weights /= np.sum(weights)
    
    # Gradient of pairwise terms
    # For pair (i,j), term = dist(i,j)/2
    # d(term)/d(centers[i]) = (centers[i] - centers[j]) / (2 * dist(i,j))
    # d(term)/d(centers[j]) = (centers[j] - centers[i]) / (2 * dist(i,j))
    
    # We need to accumulate weighted gradients
    # This is expensive but necessary for accuracy
    
    # Instead, let's use a simpler approach: numerical gradient for now
    # But that's too slow. Let me think of a better way.
    
    # Actually, let's use a different approach: direct force-based optimization
    # without explicit gradient computation.
    return grad  # Placeholder

def _force_based_step(centers, beta, lr):
    """
    One step of force-based optimization.
    Apply repulsive forces between spheres and from walls.
    """
    centers = np.array(centers, dtype=np.float64, copy=True)
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Pairwise repulsive forces
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2) + 1e-12)
            # Force proportional to 1/dist^4
            force_mag = 1.0 / (dist**4 + 1e-8)
            force_dir = diff / (dist + 1e-12)
            forces[i] += force_mag * force_dir
            forces[j] -= force_mag * force_dir
    
    # Wall repulsive forces
    for i in range(n):
        for k in range(3):
            # Left wall
            d_left = centers[i, k] + 0.5
            if d_left < 0.1:
                forces[i, k] += 1.0 / (d_left**4 + 1e-8)
            # Right wall
            d_right = 0.5 - centers[i, k]
            if d_right < 0.1:
                forces[i, k] -= 1.0 / (d_right**4 + 1e-8)
    
    # Update positions
    centers += lr * forces
    
    # Clip to valid range (with small margin)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _local_refinement(centers, max_iter=500, tol=1e-10):
    """
    Local refinement using gradient descent with adaptive step size.
    """
    centers = np.array(centers, dtype=np.float64, copy=True)
    n = centers.shape[0]
    best_score = _score(centers)
    best_centers = centers.copy()
    
    lr = 0.01
    for it in range(max_iter):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2) + 1e-12)
                force_mag = 1.0 / (dist**4 + 1e-8)
                force_dir = diff / (dist + 1e-12)
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        # Wall repulsive forces
        for i in range(n):
            for k in range(3):
                d_left = centers[i, k] + 0.5
                if d_left < 0.15:
                    forces[i, k] += 1.0 / (d_left**4 + 1e-8)
                d_right = 0.5 - centers[i, k]
                if d_right < 0.15:
                    forces[i, k] -= 1.0 / (d_right**4 + 1e-8)
        
        # Adaptive step size
        step = centers + lr * forces
        step = np.clip(step, -0.499, 0.499)
        
        new_score = _score(step)
        if new_score > best_score:
            centers = step
            best_score = new_score
            best_centers = centers.copy()
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
    
    return best_centers, best_score

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. FCC-like lattice
    a = 1.0 / 4.0
    fcc = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = i * a + 0.5 * a * (j % 2) + 0.5 * a * (k % 2)
                y = j * a + 0.5 * a * (i % 2)
                z = k * a + 0.5 * a * (i % 2 + j % 2)
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    fcc.append([x, y, z])
    fcc = np.array(fcc)
    if len(fcc) >= N:
        inits.append(fcc[:N])
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # 4. Layered structure
    n_layers = 5
    per_layer = N // n_layers
    layered = []
    for layer in range(n_layers):
        z = -0.4 + 0.8 * layer / (n_layers - 1)
        for i in range(per_layer):
            angle = 2 * np.pi * i / per_layer
            r = 0.3
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            layered.append([x, y, z])
    # Add remaining
    for i in range(N - len(layered)):
        angle = 2 * np.pi * i / 10
        r = 0.2
        layered.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    inits.append(np.array(layered[:N]))
    
    # 5. BCC-like
    a = 1.0 / 3.0
    bcc = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bcc.append([i * a - 0.5 + a/2, j * a - 0.5 + a/2, k * a - 0.5 + a/2])
                # Corner
                bcc.append([i * a - 0.5, j * a - 0.5, k * a - 0.5])
    bcc = np.array(bcc)
    # Remove duplicates and clip
    bcc = bcc[np.all(np.abs(bcc) <= 0.5, axis=1)]
    if len(bcc) >= N:
        inits.append(bcc[:N])
    
    return inits

def construct():
    """
    Main function: find optimal sphere packing.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate initializations
    inits = _generate_initializations()
    
    best_centers = None
    best_score = 0.0
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = np.array(init, dtype=np.float64, copy=True)
        
        # Phase 1: Coarse optimization with large step
        for step in range(200):
            if time.time() > deadline:
                break
            centers = _force_based_step(centers, beta=10, lr=0.05)
        
        # Phase 2: Medium optimization
        for step in range(300):
            if time.time() > deadline:
                break
            centers = _force_based_step(centers, beta=50, lr=0.02)
        
        # Phase 3: Local refinement
        refined_centers, score = _local_refinement(centers, max_iter=300)
        
        if score > best_score:
            best_score = score
            best_centers = refined_centers.copy()
    
    # Final polish
    if best_centers is not None and time.time() < deadline - 10:
        polished, final_score = _local_refinement(best_centers, max_iter=500)
        if final_score > best_score:
            best_score = final_score
            best_centers = polished
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
