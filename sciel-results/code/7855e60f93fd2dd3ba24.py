# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Pre-compute Icosahedron vectors
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
# 12 vertices of an icosahedron
_ICO_VERTS = np.array([
    [0, 1, _PHI], [0, -1, _PHI],
    [0, 1, -_PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [-1, _PHI, 0],
    [1, -_PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [-_PHI, 0, 1],
    [_PHI, 0, -1], [-_PHI, 0, -1]
])
# Normalize to unit length
_ICO_VERTS /= np.linalg.norm(_ICO_VERTS[0])

# 20 face centers (directions)
# Face centers of an icosahedron are the vertices of a dodecahedron
# Dodecahedron vertices can be derived from icosahedron or generated directly
# A standard set of 20 vectors pointing to face centers of the icosahedron defined above:
# We can compute them by averaging the 3 vertices of each face.
# Let's just hardcode the 20 normalized vectors for efficiency and robustness.
# The dodecahedron vertices are:
# (±1, ±1, ±1)
# (0, ±1/φ, ±φ)
# (±1/φ, ±φ, 0)
# (±φ, 0, ±1/φ)
# There are 8 + 12 = 20 vertices.
_DOD_VERTS = []
for sx in [-1, 1]:
    for sy in [-1, 1]:
        for sz in [-1, 1]:
            _DOD_VERTS.append([sx, sy, sz])
for sx in [-1, 1]:
    for sy in [-1, 1]:
        _DOD_VERTS.append([0, sx / _PHI, sy * _PHI])
        _DOD_VERTS.append([sx / _PHI, sy * _PHI, 0])
        _DOD_VERTS.append([sx * _PHI, 0, sy / _PHI])
_DOD_VERTS = np.array(_DOD_VERTS)
_DOD_VERTS /= np.linalg.norm(_DOD_VERTS[0])

def _generate_symmetric_positions(n, params):
    """
    Generate positions based on symmetric shell parameters.
    params: array of [r1, r2, r3, r4, r5, ...]
    Strategy:
    - Shell 1: 12 atoms at r1 (Icosahedron)
    - Shell 2: 20 atoms at r2 (Dodecahedron/Face centers)
    - Shell 3: 12 atoms at r3 (Icosahedron)
    - Shell 4: 20 atoms at r4 (Dodecahedron)
    - Shell 5: 12 atoms at r5 (Icosahedron)
    - Center: 1 atom at 0 (if n allows)
    
    We fill shells in order of "natural" icosahedral growth: 1, 12, 20, 12, 20, 12...
    However, standard Mackay growth is 1, 12, 42, 92...
    But simple shell stacking is a good approximation for many clusters.
    
    Let's define a specific mapping for the required Ns:
    n=38: 1 (center) + 12 (s1) + 20 (s2) + 5 (subset of s3? or just random remainder?)
          Better: 1 + 12 + 20 = 33. Remaining 5. 
          Maybe 1 + 12 + 25? No, 25 isn't a standard shell.
          Let's use flexible filling.
          
    General approach:
    1. Center (1)
    2. Ico Shell (12)
    3. Dod Shell (20)
    4. Ico Shell (12)
    5. Dod Shell (20)
    6. Ico Shell (12)
    
    Total capacities: 1, 13, 33, 45, 65, 77...
    
    Target Ns: 38, 55, 59, 75, 79, 80
    
    n=38: 1 + 12 + 20 + 5. The 5 atoms break symmetry. 
          To maintain symmetry, we might need to relax the "strict" symmetry for the remainder.
          Alternatively, we can use a "soft" symmetry where we place atoms on the shells 
          but allow slight displacements? No, that increases dimensions.
          
    Better Strategy: 
    Define the positions as a union of full shells and a "remainder" set that is 
    placed randomly but near the next shell, then relax. 
    BUT, the key is that the *majority* of atoms are locked to high-symmetry positions.
    
    Let's refine the parameterization:
    We will generate a base structure using full shells.
    For the remainder, we will place them at the vertices of the *next* shell 
    but randomly selected? No, that breaks symmetry.
    
    Alternative: 
    For n=38: Use 1+12+20 = 33. The remaining 5 can be placed at the centers of the 
    20 faces? No.
    
    Let's look at the specific Ns:
    38: 1+12+25? No.
    55: 1+12+42 (Mackay). 42 is not 12+20. 42 is a specific shell.
         The 42-atom shell in Mackay icosahedron consists of 12 face centers (at r2) 
         and 30 edge midpoints? Or similar.
         Actually, the 55-atom Mackay icosahedron has:
         - 1 center
         - 12 vertices (r1)
         - 30 edge midpoints (r2)
         - 20 face centers (r3)
         Total: 1+12+30+20 = 63? No, 1+12+42=55.
         The 42 atoms are: 12 face centers + 30 edge midpoints? 
         Let's check distances.
         Icosahedron edge length a.
         Vertex to center: R.
         Face center distance from center: R_face.
         Edge midpoint distance from center: R_edge.
         
         If we use this specific Mackay geometry for n=55, it's a very strong candidate.
         For other n, we use the "Shell Stacking" approximation.
    """
    positions = []
    
    # 1. Center
    if n >= 1:
        positions.append([0.0, 0.0, 0.0])
        n_remaining = n - 1
    else:
        n_remaining = n
        
    # Define shells in order of appearance in typical icosahedral growth
    # Shell types: 'ico' (12), 'edge' (30), 'face' (20)
    # Mackay 55: 1, 12, 30, 20. Radii: 0, R1, R2, R3.
    # General growth: 1, 12, 20, 12, 20... (Close packing like?)
    
    # We will use a hybrid approach:
    # If n == 55, use Mackay 55 geometry.
    # Else, use generic shell stacking: 1, 12, 20, 12, 20...
    
    if n == 55:
        # Mackay 55
        # Radii parameters: r1 (verts), r2 (edges), r3 (faces)
        # params: [r1, r2, r3]
        r1, r2, r3 = params[0], params[1], params[2]
        
        # 12 Vertices
        verts = _ICO_VERTS * r1
        positions.extend(verts)
        
        # 30 Edge Midpoints
        # Edges of icosahedron connect vertices i and j if dist(vert_i, vert_j) is min.
        # Let's compute edge midpoints explicitly.
        edge_mids = []
        v_norm = np.linalg.norm(_ICO_VERTS[0]) # 1.0
        # Find all edges
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(_ICO_VERTS[i] - _ICO_VERTS[j])
                # Min distance in unit icosahedron is 1/phi? No.
                # Vertices are (0,1,phi) etc. Dist between (0,1,phi) and (0,-1,phi) is 2.
                # We normalized to unit length.
                # Original dist 2. Norm 1.902. Scaled dist = 2/1.902 = 1.051.
                # So if d is close to 1.051, it's an edge.
                if abs(d - 1.05146) < 1e-3:
                    mid = (_ICO_VERTS[i] + _ICO_VERTS[j]) / 2.0
                    edge_mids.append(mid)
        
        # There should be 30 unique midpoints.
        # Let's ensure we have 30.
        # Deduplicate just in case (though i<j handles it)
        edge_mids = np.array(edge_mids)
        if len(edge_mids) != 30:
             # Fallback if calculation failed
             # Just use 30 random points on sphere? No, let's trust the geometry.
             # For robustness, if it fails, we'll pad with ico verts scaled.
             pass
        else:
             # Normalize edge mids to unit vector, then scale by r2
             edge_mids_norm = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
             positions.extend(edge_mids_norm * r2)
             
        # 20 Face Centers
        face_centers = _DOD_VERTS * r3
        positions.extend(face_centers)
        
        return np.array(positions)

    else:
        # Generic Shell Stacking
        # Shells: 12 (Ico), 20 (Dod), 12 (Ico), 20 (Dod)...
        # Radii: r1, r2, r3, r4...
        
        current_n = 0
        shell_idx = 0
        while n_remaining > 0 and shell_idx < len(params):
            r = params[shell_idx]
            
            # Determine shell size and type
            if shell_idx % 2 == 0:
                # Icosahedral shell (12 atoms)
                size = 12
                vecs = _ICO_VERTS
            else:
                # Dodecahedral shell (20 atoms)
                size = 20
                vecs = _DOD_VERTS
            
            # How many to take from this shell?
            take = min(size, n_remaining)
            
            if take > 0:
                # If we take a subset, we break symmetry.
                # To preserve symmetry as much as possible, we should only take full shells.
                # If we must take a partial shell, we take the first 'take' vectors.
                # This is a heuristic; relaxation will fix positions.
                selected_vecs = vecs[:take]
                positions.extend(selected_vecs * r)
                n_remaining -= take
                current_n += take
            
            shell_idx += 1
            
        # If we still have atoms remaining (n_remaining > 0), 
        # it means we ran out of pre-defined shells in params.
        # We should have enough params.
        # If not, place them randomly on the next sphere.
        if n_remaining > 0:
             r_last = params[-1] * 1.1
             # Random points on sphere
             rng = np.random.default_rng(42)
             for _ in range(n_remaining):
                 v = rng.normal(size=3)
                 v /= np.linalg.norm(v)
                 positions.append(v * r_last)
        
        return np.array(positions)

def _objective(params, n, kind):
    """
    Calculate energy for a given set of shell radii.
    """
    try:
        pos = _generate_symmetric_positions(n, params)
        if pos.shape[0] != n:
            return np.inf
        if not np.all(np.isfinite(pos)):
            return np.inf
        # Check min dist
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        if dists.min() < 0.35:
            return np.inf
            
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e
    except Exception:
        return np.inf

def _symmetric_search_worker(args):
    """
    Worker that performs optimization over the shell radii.
    """
    n, kind, seed, max_time = args
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Determine number of shells needed
    # Max n is 80.
    # 1 + 12 + 20 + 12 + 20 + 12 = 77.
    # So we need at least 5 shells (indices 0-4) plus center.
    # Let's use 6 parameters for safety (radii for shells 1-6).
    n_params = 6
    
    # Initial guess for radii
    # Typical SC equilibrium distance ~ 1.0-1.2
    # Shell 1 (Ico): r ~ 1.0
    # Shell 2 (Dod): r ~ 1.5
    # Shell 3 (Ico): r ~ 2.0
    # etc.
    initial_r = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    
    # Add some noise
    current_params = initial_r + rng.normal(0, 0.1, size=n_params)
    
    # Use Nelder-Mead to optimize the radii
    # This is a low-dimensional optimization (6D)
    
    best_e = np.inf
    best_params = current_params
    
    # Run multiple restarts in parallel? No, this worker is one restart.
    # We will run several of these workers in parallel with different seeds.
    
    # Simple local search / Nelder-Mead
    # SciPy minimize
    from scipy.optimize import minimize
    
    def fun(p):
        return _objective(p, n, kind)
    
    try:
        res = minimize(fun, current_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success and np.isfinite(res.fun) and res.fun < best_e:
            best_e = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # If Nelder-Mead failed or got stuck, try a few random perturbations
    for _ in range(10):
        if time.time() - start_time > max_time - 5:
            break
        perturbed = best_params + rng.normal(0, 0.2, size=n_params)
        e = _objective(perturbed, n, kind)
        if e < best_e:
            best_e = e
            best_params = perturbed
            # Quick local refine
            try:
                res2 = minimize(fun, best_params, method='Nelder-Mead', 
                                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res2.success and np.isfinite(res2.fun) and res2.fun < best_e:
                    best_e = res2.fun
                    best_params = res2.x
            except:
                pass
                
    # Final relaxation of the full 3n coordinates
    # The symmetric structure is a good initial guess for full relaxation.
    final_pos = _generate_symmetric_positions(n, best_params)
    
    # Run standard relax to fine-tune
    try:
        final_e, final_pos = relax(final_pos, 'sutton_chen', kind)
        if np.isfinite(final_e) and final_e < best_e:
            best_e = final_e
    except Exception:
        pass
        
    return (best_e, final_pos)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin
    
    # We will run 4 parallel symmetric search workers with different random seeds.
    # Each worker optimizes the 6 shell radii.
    
    tasks = []
    for i in range(4):
        tasks.append((n, kind, 42 + i, 60.0))
        
    best_e = np.inf
    best_p = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_symmetric_search_worker, t) for t in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
    
    # If we have time left, do a final basin hopping on the best symmetric solution
    # to break any remaining symmetry constraints that might be suboptimal.
    if best_p is not None and time.time() < deadline - 30.0:
        rng_bh = np.random.default_rng(999)
        # Perturb
        perturbed = best_p + rng_bh.normal(0, 0.5, size=best_p.shape)
        try:
            e_bh, p_bh = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(e_bh) and e_bh < best_e:
                best_e = e_bh
                best_p = p_bh
        except:
            pass
            
        # One more BH step
        if time.time() < deadline - 10.0:
            perturbed2 = best_p + rng_bh.normal(0, 0.3, size=best_p.shape)
            try:
                e_bh2, p_bh2 = relax(perturbed2, 'sutton_chen', kind)
                if np.isfinite(e_bh2) and e_bh2 < best_e:
                    best_e = e_bh2
                    best_p = p_bh2
            except:
                pass

    if best_p is None:
        # Fallback to random
        rng = np.random.default_rng(0)
        r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
        pts = rng.uniform(-r_est, r_est, size=(n, 3))
        try:
            best_e, best_p = relax(pts, 'sutton_chen', kind)
        except:
            best_p = pts

    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
