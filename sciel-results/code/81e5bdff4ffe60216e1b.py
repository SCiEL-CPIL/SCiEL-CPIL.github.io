# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import sys

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions, A=A, B=B, a=a, b=b, c=c, d=d, m=m):
    """Compute Dzugutov potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    i_idx, j_idx = np.triu_indices(n, k=1)
    r = dists[i_idx, j_idx]
    
    # Mask for r < a
    mask_a = r < a
    if np.any(mask_a):
        r_a = r[mask_a]
        # Avoid division by zero or negative base issues if r is very small
        # r^-m will blow up, which is correct for repulsion
        energy += np.sum(A * (r_a**(-m) - B) * np.exp(c / (r_a - a)))
        
    # Mask for a <= r < b
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r[mask_b]
        energy += np.sum(B * np.exp(d / (r_b - b)))
        
    return energy

def _dzugutov_gradient(positions, A=A, B=B, a=a, b=b, c=c, d=d, m=m):
    """Compute gradient of Dzugutov potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
        
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    r = dists[i_idx, j_idx]
    rij_vec = diff[i_idx, j_idx]
    
    force = np.zeros_like(rij_vec)
    
    # Mask for r < a
    mask_a = r < a
    if np.any(mask_a):
        r_a = r[mask_a]
        # dV/dr = A * [ -m * r^(-m-1) * exp(c/(r-a)) + (r^(-m) - B) * exp(c/(r-a)) * (-c/(r-a)^2) ]
        # = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^(-m) - B) / (r-a)^2 ]
        term1 = -m * r_a**(-m-1)
        term2 = -c * (r_a**(-m) - B) / (r_a - a)**2
        dVdr = A * np.exp(c / (r_a - a)) * (term1 + term2)
        force[mask_a] = dVdr[:, np.newaxis] * rij_vec[mask_a] / r_a[:, np.newaxis]
        
    # Mask for a <= r < b
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r[mask_b]
        # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
        dVdr = B * np.exp(d / (r_b - b)) * (-d / (r_b - b)**2)
        force[mask_b] = dVdr[:, np.newaxis] * rij_vec[mask_b] / r_b[:, np.newaxis]
        
    # Accumulate forces
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        grad[i] -= force[k]
        grad[j] += force[k]
        
    return grad

def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0])
    return verts

def _generate_mackay_55():
    """Generate a perfect 55-atom Mackay icosahedron."""
    verts = _get_icosahedron_vertices()
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.15
    for v in verts:
        points.append(v * r1)
        
    # Shell 2: 30 atoms (edge midpoints projected)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(verts[i] - verts[j])
            if dist < 1.5:
                edges.append((i, j))
                
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        norm_mid = mid / np.linalg.norm(mid)
        points.append(norm_mid * (r1 * np.sqrt(2)))
        
    # Shell 3: 12 atoms (vertices of dual icosahedron? No, Mackay shell 3 is 12 atoms at vertices of a larger icosahedron rotated? 
    # Actually, Mackay 55 is 1 + 12 + 30 + 12. The 12 atoms of shell 3 are at the vertices of the icosahedron scaled by sqrt(3)?
    # Let's check distances.
    # In ideal Mackay, r3 = sqrt(3) * r1.
    for v in verts:
        points.append(v * (r1 * np.sqrt(3)))
        
    return np.array(points[:55])

def _generate_defect_seeds(n, rng):
    """
    Generate diverse seeds for size n based on the 55-atom Mackay icosahedron.
    For n < 55, remove atoms. For n > 55, add atoms.
    """
    mackay = _generate_mackay_55()
    seeds = []
    
    if n == 55:
        seeds.append(mackay.copy())
    elif n < 55:
        # Remove 55 - n atoms
        # Strategy: Remove atoms from the outer shell first, then inner?
        # Or remove random atoms?
        # Better: Remove atoms that are "least bound" or just random for diversity.
        # We will generate multiple variants by removing different sets of atoms.
        num_remove = 55 - n
        # Shuffle indices
        idx = np.arange(55)
        # Generate 5 variants
        for _ in range(5):
            rng.shuffle(idx)
            remove_idx = idx[:num_remove]
            keep_idx = np.setdiff1d(idx, remove_idx)
            seed = mackay[keep_idx]
            # Center it
            seed = seed - np.mean(seed, axis=0)
            seeds.append(seed)
    else:
        # Add n - 55 atoms
        num_add = n - 55
        base = mackay.copy()
        # Add atoms on the surface
        # Find surface atoms (those with fewer than 12 neighbors?)
        # For simplicity, add at random positions on a sphere slightly larger than the cluster
        for _ in range(5):
            seed = base.copy()
            for _ in range(num_add):
                # Pick a random direction
                dir_vec = rng.normal(0, 1, 3)
                dir_vec /= np.linalg.norm(dir_vec)
                # Place at radius ~ max_radius * 1.1
                max_r = np.max(np.linalg.norm(base, axis=1))
                pos = dir_vec * (max_r * 1.1)
                seed = np.vstack([seed, pos])
            seed = seed - np.mean(seed, axis=0)
            seeds.append(seed)
            
    return seeds

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    # param is not used directly in our manual gradient, but relax might need it
    # The task says relax(positions, 'dzugutov') may be called.
    # We will implement our own local relaxation using gradient descent to avoid dependency issues or slow relax.
    
    start_time = time.time()
    deadline = start_time + 290
    
    best_energy = np.inf
    best_positions = None
    
    # Generate seeds
    rng = np.random.default_rng(n)
    seeds = _generate_defect_seeds(n, rng)
    
    # Add a random compact seed as fallback
    try:
        points = [np.zeros(3)]
        radius = 0.5
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                min_dist = min(np.linalg.norm(p - q) for q in points) if points else np.inf
                if min_dist >= 0.5:
                    points.append(p)
            else:
                radius += 0.1
        if len(points) == n:
            seed_rand = np.array(points) - np.mean(np.array(points), axis=0)
            seeds.append(seed_rand)
    except Exception:
        pass

    # Local Relaxation Function
    def local_relax(pos):
        """
        Gradient descent with adaptive step size.
        """
        pos = pos.copy()
        lr = 0.01
        for i in range(100):
            if time.time() > deadline - 5:
                break
            grad = _dzugutov_gradient(pos)
            # Check for NaNs
            if np.any(np.isnan(grad)):
                break
            # Adaptive step: if energy increases, reduce lr
            new_pos = pos - lr * grad
            # Enforce minimum distance constraint roughly?
            # If atoms overlap, gradient will be huge and push them apart.
            # But if they are too close, r^-16 explodes.
            # Simple check:
            i_idx, j_idx = np.triu_indices(n, k=1)
            dists = np.linalg.norm(new_pos[i_idx] - new_pos[j_idx], axis=1)
            if np.min(dists) < 0.1:
                # Push apart
                for k in range(len(i_idx)):
                    if dists[k] < 0.1:
                        i = i_idx[k]
                        j = j_idx[k]
                        diff = new_pos[i] - new_pos[j]
                        if np.linalg.norm(diff) < 1e-6:
                            diff = rng.normal(0, 0.01, 3)
                        diff /= np.linalg.norm(diff)
                        new_pos[i] += diff * 0.05
                        new_pos[j] -= diff * 0.05
                continue
                
            e_old = _dzugutov_energy(pos)
            e_new = _dzugutov_energy(new_pos)
            
            if e_new < e_old:
                pos = new_pos
                lr *= 1.5
            else:
                lr *= 0.5
                
            if np.linalg.norm(grad) < 1e-4:
                break
                
        return pos, _dzugutov_energy(pos)

    # Basin Hopping
    def basin_hopping(seed):
        nonlocal best_energy, best_positions
        
        current_pos = seed.copy()
        current_energy, _ = local_relax(current_pos)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos.copy()
            
        T = 100.0
        for it in range(20): # 20 iterations of BH
            if time.time() > deadline - 10:
                break
                
            # Step
            step_size = 0.5
            step = rng.normal(0, step_size, (n, 3))
            new_pos = current_pos + step
            
            # Relax
            relaxed_pos, relaxed_energy = local_relax(new_pos)
            
            # Metropolis
            delta_E = relaxed_energy - current_energy
            if delta_E < 0 or rng.rand() < np.exp(-delta_E / T):
                current_pos = relaxed_pos
                current_energy = relaxed_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos.copy()
                    
            # Cool down
            T *= 0.9

    # Run Basin Hopping for each seed
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        try:
            basin_hopping(seed)
        except Exception:
            pass

    # Fallback if no valid result
    if best_positions is None:
        # Try relaxing the first seed directly
        try:
            pos, e = local_relax(seeds[0])
            best_positions = pos
            best_energy = e
        except Exception:
            # Last resort random
            best_positions = rng.normal(0, 1, (n, 3))
            
    # Final check: ensure minimum distance constraint
    if len(best_positions) >= 2:
        i_idx, j_idx = np.triu_indices(len(best_positions), k=1)
        dists = np.linalg.norm(best_positions[i_idx] - best_positions[j_idx], axis=1)
        if np.min(dists) < 0.5:
            # Attempt to fix by expanding
            center = np.mean(best_positions, axis=0)
            best_positions = (best_positions - center) * 1.5 + center
            # Relax again
            try:
                pos, e = local_relax(best_positions)
                best_positions = pos
            except Exception:
                pass

    return best_positions
# EVOLVE-BLOCK-END
