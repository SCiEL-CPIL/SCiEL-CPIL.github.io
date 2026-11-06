# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _get_icosahedron_vertices():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_icosahedron_fragment():
    """
    Generate a relaxed 13-atom icosahedral fragment (1 center + 12 vertices).
    This is the fundamental stable unit in the Dzugutov potential.
    We pre-relax it to ensure it's at its local minimum.
    """
    v = _get_icosahedron_vertices()
    # Initial guess: center at origin, vertices at radius ~1.0
    # The Dzugutov potential has a minimum around r~1.8-1.9 for pairs?
    # Actually, let's just use the provided relax to get the exact geometry.
    # But relax might be slow. Let's use a fixed, good geometry first.
    # For Dzugutov, the icosahedron is very stable.
    # Let's construct it with a radius that puts nearest neighbors at ~1.85.
    # In a regular icosahedron, edge length = 2 * R * sin(pi/5) ~ 1.176 R.
    # If we want edge ~ 1.85, R ~ 1.57.
    radius = 1.55
    frag = np.vstack([np.zeros(3), v * radius])
    
    # Try to relax it quickly to ensure it's a local minimum
    try:
        relaxed_frag = relax(frag, 'dzugutov')
        return relaxed_frag
    except:
        return frag

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix."""
    # Using QR decomposition of a random normal matrix
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph
    return q

def _assemble_cluster_fragments(n, rng, start_time, deadline):
    """
    Assemble n atoms by merging 13-atom icosahedral fragments.
    1. Determine number of fragments.
    2. Place fragments in a coarse grid/random configuration.
    3. Optimize fragment positions/orientations (coarse).
    4. Expand to full atomic positions.
    5. Relax full system (fine).
    """
    # Strategy for n atoms:
    # n=38: ~3 fragments (39 atoms) -> remove 1
    # n=45: ~3-4 fragments
    # n=50: ~4 fragments (52 atoms) -> remove 2
    # n=55: ~4 fragments (52) + 3? Or 5 fragments (65) -> remove 10. 
    #       Actually, 55 is 1+12+12+30. 
    #       4 icosahedra share centers? 
    #       Let's just use as many 13-atom fragments as fit, and trim/add.
    
    num_frags = max(1, n // 13)
    # If n is not a multiple of 13, we will have extra atoms or need to remove.
    # Let's generate enough fragments to cover n, then trim.
    # For n=55, 55/13 = 4.23. So 4 frags = 52 atoms. We need 3 more.
    # For n=60, 60/13 = 4.6. 4 frags = 52. Need 8 more.
    
    # Generate fragments
    fragments = []
    for _ in range(num_frags + 1): # One extra to be safe for trimming
        frag = _generate_icosahedron_fragment()
        fragments.append(frag)
    
    # Initial coarse placement of fragment centers
    # Place them on a sphere or random compact cloud
    centers = []
    for i in range(len(fragments)):
        if i == 0:
            centers.append(np.zeros(3))
        else:
            # Random direction, radius ~ 3.0 (to avoid overlap initially)
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            centers.append(u * 3.0)
            
    centers = np.array(centers)
    
    # Define the objective for coarse optimization
    # We optimize the centers and orientations of the fragments.
    # Variables: [c1(3), R1(9), c2(3), R2(9), ...]
    # But optimizing rotations directly is tricky. 
    # Alternative: Just optimize centers first, keeping orientations fixed or random?
    # Better: Use a simple heuristic assembly.
    
    # Let's try a different assembly: 
    # 1. Place fragment 1 at origin.
    # 2. Place fragment 2 at a random distance/orientation.
    # 3. Relax the combined system.
    # 4. If n is larger, add another fragment.
    
    # This "stepwise growth" is more robust for small n.
    
    current_atoms = fragments[0].copy() # 13 atoms
    current_centers = [np.zeros(3)]
    
    target_n = n
    
    # We will iteratively add fragments and relax
    # To save time, we limit the number of iterations
    
    max_steps = 5
    time_budget_per_step = (deadline - start_time) / (max_steps + 1)
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
            
        if len(current_atoms) >= target_n:
            # We have enough atoms. Trim if necessary.
            if len(current_atoms) > target_n:
                # Remove atoms from the periphery (highest index in fragment order? 
                # Or just random? Random is safer to break symmetry)
                # But we must ensure we don't remove the center of a fragment if it's isolated.
                # Simplest: just take the first target_n atoms. 
                # Since fragments are added sequentially, the last added are on the surface.
                current_atoms = current_atoms[:target_n]
                break
            else:
                # Exact match
                break
                
        # Add a new fragment
        frag = fragments[step + 1] if step + 1 < len(fragments) else _generate_icosahedron_fragment()
        
        # Place the new fragment near the current cluster
        # Find the "center of mass" of current_atoms
        com = np.mean(current_atoms, axis=0)
        
        # Random offset
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u)
        offset = u * 3.0
        
        new_frag_pos = frag + com + offset
        
        # Combine
        new_atoms = np.vstack([current_atoms, new_frag_pos])
        
        # Relax the combined system
        try:
            relaxed = relax(new_atoms, 'dzugutov')
            current_atoms = relaxed
        except Exception:
            # If relax fails, keep unrelaxed and try again with different offset?
            # Or just break.
            current_atoms = new_atoms
            
        # Check time
        if time.time() > deadline - 20:
            break

    # Final relaxation
    if time.time() < deadline - 5:
        try:
            final_relaxed = relax(current_atoms, 'dzugutov')
            current_atoms = final_relaxed
        except Exception:
            pass
            
    return current_atoms

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin (300s limit, 10s buffer)
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy: Multi-start with Fragment Assembly
    # We try a few different random seeds for the assembly
    num_starts = 3
    
    time_per_start = 250.0 / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        current_deadline = start_time + (i + 1) * time_per_start
        if current_deadline > deadline - 5:
            current_deadline = deadline - 5
            
        try:
            pos = _assemble_cluster_fragments(n, rng, start_time, current_deadline)
            
            # Ensure correct number of atoms
            if len(pos) != n:
                if len(pos) > n:
                    pos = pos[:n]
                else:
                    # Pad with random atoms if somehow too few (shouldn't happen)
                    extra = rng.uniform(-3, 3, (n - len(pos), 3))
                    pos = np.vstack([pos, extra])
            
            # Final check and relax
            if time.time() < deadline - 2:
                try:
                    pos = relax(pos, potential)
                except:
                    pass
                    
            # Calculate energy
            e, _ = dzugutov_energy_gradient(pos)
            
            # Check validity (min distance)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                continue
                
            if e < best_energy:
                best_energy = e
                best_pos = pos
                
        except Exception:
            continue
            
    # Fallback if assembly failed
    if best_pos is None:
        # Random compact
        radius = 1.5 * n**(1.0/3.0)
        pos = rng.uniform(-radius, radius, (n, 3))
        try:
            pos = relax(pos, potential)
            e, _ = dzugutov_energy_gradient(pos)
            best_energy = e
            best_pos = pos
        except:
            best_pos = pos
            best_energy = np.inf

    # Final Safety Check
    if time.time() < deadline - 2:
        try:
            final_pos = best_pos.copy()
            relaxed_final = relax(final_pos, potential)
            e_final, _ = dzugutov_energy_gradient(relaxed_final)
            if e_final < best_energy:
                best_pos = relaxed_final
        except Exception:
            pass

    # Verify constraints
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            # Fallback to a safe random structure if constraints are violated
            radius = 2.0 * n**(1.0/3.0)
            safe_pos = rng.uniform(-radius, radius, (n, 3))
            safe_pos = safe_pos - np.mean(safe_pos, axis=0)
            try:
                relaxed_safe = relax(safe_pos, potential)
                best_pos = relaxed_safe
            except Exception:
                best_pos = safe_pos

    return best_pos
# EVOLVE-BLOCK-END
