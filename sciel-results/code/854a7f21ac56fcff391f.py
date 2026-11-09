"""
Global minima search for Sutton-Chen transition-metal clusters.
Uses a hybrid strategy combining the deterministic LAMMPS cluster growth algorithm
with multi-start local relaxation from structured seeds.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

def _validate_positions(pos):
    """Check for finite coordinates and minimum distance constraints."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # Vectorized distance check
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35

def _fcc_cluster(n, spacing=0.72):
    """Generate an FCC cluster of n atoms."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]

def _icosahedral_cluster(n, spacing=0.72):
    """Generate an icosahedral cluster of n atoms."""
    if n <= 0:
        return np.empty((0, 3))
    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    for v in verts:
        points.append(v * spacing)
    if len(points) >= n:
        return np.array(points[:n])
        
    # Edge midpoints
    mids = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 1.0 < d < 2.0:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                mids.append(mid)
    for m in mids:
        points.append(m * spacing * np.sqrt(3))
    if len(points) >= n:
        return np.array(points[:n])
        
    # Fill remaining with random compact
    remaining = n - len(points)
    if remaining > 0:
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            p = rng.uniform(-spacing*2, spacing*2, 3)
            points.append(p)
    return np.array(points[:n])

def _try_relax(pos, potential, kind, max_time=10.0):
    """Attempt to relax positions using the provided helper."""
    start = time.time()
    try:
        if not _validate_positions(pos):
            return np.inf, None
        # The helper 'relax' is assumed to be efficient. 
        # We wrap it in a try block in case it hangs or fails.
        energy, relaxed = relax(pos, potential, kind)
        if time.time() - start > max_time:
            return np.inf, None
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None

def _lammps_grow_cluster(n_target, kind, spacing=0.72, time_limit=200.0):
    """
    Deterministic growth algorithm.
    Starts with a small core and adds atoms one by one, minimizing energy at each step.
    """
    start_time = time.time()
    
    # 1. Initialize core (Tetrahedron or small Icosahedron fragment)
    # A tetrahedron is a good starting point for close-packed structures
    phi = (1 + np.sqrt(5)) / 2
    tet_verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tet_verts = tet_verts / np.linalg.norm(tet_verts, axis=1, keepdims=True) * spacing
    
    current_pos = tet_verts.copy()
    n_current = 4
    
    # If target is smaller, just take the first n_target atoms
    if n_target <= n_current:
         e, p = _try_relax(current_pos[:n_target], 'sutton_chen', kind, max_time=5.0)
         if p is not None:
             return e, p
         return np.inf, None

    # 2. Growth Loop
    while n_current < n_target:
        if time.time() - start_time > time_limit:
            break
            
        # Generate candidate positions for the new atom
        # Strategy: Place near existing atoms (shells) and random directions
        candidates = []
        
        # Radial shells around center and existing atoms
        for atom in current_pos:
            for r in [spacing * 0.8, spacing, spacing * 1.1, spacing * 1.5]:
                # Generate points on sphere
                for theta in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    for phi_angle in np.linspace(0, np.pi, 4, endpoint=False):
                        p = atom + r * np.array([
                            np.sin(phi_angle) * np.cos(theta),
                            np.sin(phi_angle) * np.sin(theta),
                            np.cos(phi_angle)
                        ])
                        candidates.append(p)
                        
        # Random candidates
        rng = np.random.default_rng(int(time.time()*1000) + n_current)
        for _ in range(50):
            p = rng.uniform(-spacing*2, spacing*2, 3)
            candidates.append(p)
            
        candidates = np.array(candidates)
        
        # Filter invalid candidates (too close to existing)
        valid_cands = []
        for p in candidates:
            dists = np.linalg.norm(current_pos - p, axis=1)
            if np.min(dists) > 0.4: # Safety margin
                valid_cands.append(p)
                
        if not valid_cands:
             # Fallback: place far away
             valid_cands = [np.array([spacing*2, 0, 0])]
             
        valid_cands = np.array(valid_cands)
        
        # Optimize position of new atom (fixed core)
        best_new_atom = None
        best_e_new = np.inf
        
        # We can't easily vectorize the minimization for each candidate, 
        # so we pick a subset of best candidates based on quick energy eval
        # or just minimize the best few.
        
        # Quick energy evaluation to prune
        quick_energies = []
        for i, p in enumerate(valid_cands):
             # Approximate energy: just interaction with core
             dists = np.linalg.norm(current_pos - p, axis=1)
             r2 = np.sum(dists**2)
             # Rough SC energy
             e_rep = np.sum(dists**(-_KIND_PARAMS[kind][0]))
             S = np.sum(dists**(-_KIND_PARAMS[kind][1]))
             e_att = -_KIND_PARAMS[kind][2] * np.sqrt(S)
             quick_energies.append(e_rep + e_att)
             
        quick_energies = np.array(quick_energies)
        # Take top 10 lowest energy candidates
        top_indices = np.argsort(quick_energies)[:10]
        
        for idx in top_indices:
            p0 = valid_cands[idx]
            
            # Define objective for single atom minimization
            def obj_new_atom(x):
                x = x.reshape(3)
                new_pos = np.vstack([current_pos, x])
                if not _validate_positions(new_pos):
                    return 1e10, np.zeros(3)
                e, g = sutton_chen_energy_gradient(new_pos, kind)
                if not np.isfinite(e):
                    return 1e10, np.zeros(3)
                return e, g[-1] # Gradient of last atom only
            
            try:
                res = minimize(
                    fun=obj_new_atom,
                    x0=p0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.fun < best_e_new:
                    best_e_new = res.fun
                    best_new_atom = res.x
            except Exception:
                continue
                
        if best_new_atom is None:
            # Fallback: add at the best quick-energy position
            best_new_atom = valid_cands[top_indices[0]]
            
        # Add the new atom
        current_pos = np.vstack([current_pos, best_new_atom])
        n_current += 1
        
        # Full relaxation of the entire cluster
        # This is crucial to allow the core to adjust
        e_full, p_full = _try_relax(current_pos, 'sutton_chen', kind, max_time=5.0)
        if p_full is not None:
            current_pos = p_full
        else:
            # If relaxation fails, keep the unrelaxed position but check validity
            if not _validate_positions(current_pos):
                # Try to fix by pushing apart
                current_pos = current_pos + np.random.normal(0, 0.1, current_pos.shape)
                
    # Final full relaxation
    e_final, p_final = _try_relax(current_pos, 'sutton_chen', kind, max_time=30.0)
    if p_final is not None:
        return e_final, p_final
    return np.inf, None

def construct(instance):
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    best_energy = np.inf
    best_positions = None
    
    # 1. LAMMPS Growth (Deterministic, often finds global minima for cohesive clusters)
    e_grow, p_grow = _lammps_grow_cluster(n, kind, time_limit=120.0)
    if p_grow is not None:
        best_energy = e_grow
        best_positions = p_grow
    
    # 2. Multi-start from structured seeds
    # Only if time permits
    if time.time() < deadline - 60.0:
        seeds = [
            _fcc_cluster(n, 0.72),
            _fcc_cluster(n, 0.65),
            _fcc_cluster(n, 0.80),
            _icosahedral_cluster(n, 0.72),
            _icosahedral_cluster(n, 0.65)
        ]
        
        # Add some random compact seeds
        rng = np.random.default_rng(42)
        for _ in range(5):
            radius = 0.72 * n ** (1.0/3.0) * 0.8
            points = []
            for _ in range(n):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    points.append(p)
            if len(points) == n:
                seeds.append(np.array(points))
                
        for seed in seeds:
            if time.time() > deadline - 30.0:
                break
            e, p = _try_relax(seed, potential, kind, max_time=15.0)
            if p is not None and e < best_energy:
                best_energy = e
                best_positions = p
    
    # 3. Basin Hopping / Perturbation from best found
    if best_positions is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for scale in [0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline - 5.0:
                break
            perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
            e, p = _try_relax(perturbed, potential, kind, max_time=10.0)
            if p is not None and e < best_energy:
                best_energy = e
                best_positions = p
    
    # 4. Final Polish
    if best_positions is not None:
        try:
            e_final, p_final = relax(best_positions, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final) and e_final < best_energy:
                best_positions = p_final
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        best_positions = _fcc_cluster(n, 0.72)
        
    return best_positions
# EVOLVE-BLOCK-END
