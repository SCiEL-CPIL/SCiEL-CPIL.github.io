"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    # Calculate all pairwise distances efficiently for small n
    # For n ~ 80, n^2 is 6400, which is fast.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < min_dist):
        return False
    return True


def _generate_icosahedron(n, seed=0):
    """
    Generate a cluster with icosahedral symmetry.
    For small n, it's a truncated icosahedron or nested shells.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices of icosahedron) -> N=13
    Shell 2: 42 atoms (edges + faces) -> N=55 (Mackay icosahedron)
    Shell 3: 92 atoms -> N=147
    
    For other N, we truncate the shells or add random atoms on the surface.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (radius 1)
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        # Shell 1: 12 vertices
        pts.extend(v[:min(12, n - 1)])
        
    if n > 13:
        # Shell 2: 42 atoms
        # Generate 42 points: 30 edge midpoints (scaled to radius 2) + 12 face centers (scaled to radius 2)
        # Actually, in Mackay, the shell radii are 1, 2, 3...
        # But the atoms are at specific symmetry positions.
        # Let's generate the 42 points of the second shell of a Mackay icosahedron.
        # The 42 points are the 30 edge centers and 12 face centers of the icosahedron, 
        # but scaled to radius 2.
        
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:  # Adjacent vertices
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
        
        faces = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if 1.5 < d1 < 2.5 and 1.5 < d2 < 2.5 and 1.5 < d3 < 2.5:
                        center = (v[i] + v[j] + v[k]) / 3
                        center = center * 2.0 / np.linalg.norm(center)
                        faces.append(center)
        
        shell2_pts.extend(faces[:12])
        
        # Deduplicate
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
                
        remaining = min(42, n - len(pts))
        pts.extend(unique_pts[:remaining])
        
    if n > 55:
        # Shell 3: 92 atoms (radius 3)
        # This is more complex. For simplicity, if n > 55, we can add atoms on a sphere of radius 3
        # using a Fibonacci sphere or random points with minimum distance constraint.
        rng = np.random.default_rng(seed + n)
        remaining = n - len(pts)
        for _ in range(remaining * 10):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
            # Check min distance to existing points
            if all(np.linalg.norm(p - q) > 0.4 for q in pts):
                pts.append(p)
                
    return np.array(pts[:n])


def _generate_decahedron(n, seed=0):
    """
    Generate a cluster with decahedral symmetry (Mackay-decahedron).
    Composed of 4 tetrahedra and 4 octahedra? Or just a distorted icosahedron.
    A simpler approach: Start with an icosahedron and perturb it slightly.
    """
    base = _generate_icosahedron(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # Apply a small random perturbation to break symmetry
    perturbation = rng.normal(0, 0.1, base.shape)
    return base + perturbation


def _generate_fcc(n, seed=0):
    """Generate n atoms in FCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n, seed=0):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_random(n, seed=0):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.5 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _relax_and_check(positions, kind, max_iter=1000):
    """
    Relax positions using L-BFGS-B and return (energy, positions).
    Returns (np.inf, positions) if invalid or fails.
    """
    if not _is_valid(positions):
        return np.inf, positions
    
    try:
        energy, gradient = sutton_chen_energy_gradient(positions, kind)
        if not np.isfinite(energy):
            return np.inf, positions
        
        def fun(x):
            e, g = sutton_chen_energy_gradient(x, kind)
            return e, g
            
        res = minimize(fun, positions, method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.success and _is_valid(res.x):
            return res.fun, res.x
        else:
            # Try the helper relax function as a fallback
            e, pos = relax(positions, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(pos):
                return e, pos
            return np.inf, positions
    except Exception:
        return np.inf, positions


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # List of generators to try
    generators = [
        _generate_icosahedron,
        _generate_decahedron,
        _generate_fcc,
        _generate_bcc,
        _generate_random
    ]
    
    # Phase 1: Generate and relax multiple initial structures
    # We will try each generator with a few different seeds/perturbations
    candidates = []
    
    for gen in generators:
        if _time_left() < 15.0:
            break
        
        # Try 2-3 variations for each generator
        for seed in [0, 1, 2]:
            if _time_left() < 10.0:
                break
            
            try:
                base = gen(n, seed=seed)
                if not _is_valid(base):
                    # Scale down if atoms are too close
                    base = base * 1.2
                    if not _is_valid(base):
                        continue
                
                e, pos = _relax_and_check(base, kind, max_iter=500)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
                    candidates.append((e, pos.copy()))
            except Exception:
                continue
    
    # Phase 2: Basin Hopping on the best candidate
    if best_pos is not None and _time_left() > 30.0:
        rng = np.random.default_rng(12345 + n)
        current_e = best_energy
        current_pos = best_pos.copy()
        
        # Simple Basin Hopping
        for i in range(10): # 10 trials
            if _time_left() < 10.0:
                break
            
            sigma = 0.1 * (0.9 ** i)
            perturbed = current_pos + rng.normal(0, sigma, current_pos.shape)
            
            if not _is_valid(perturbed):
                continue
                
            e_new, pos_new = _relax_and_check(perturbed, kind, max_iter=300)
            
            if np.isfinite(e_new) and e_new < current_e:
                current_e = e_new
                current_pos = pos_new.copy()
                if e_new < best_energy:
                    best_energy = e_new
                    best_pos = pos_new.copy()
            else:
                # Keep current best for next step (standard BH)
                pass 

    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-3, 3, (n, 3))
            for attempt in range(50):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-3, 3, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
