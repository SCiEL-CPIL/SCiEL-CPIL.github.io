"""
Strategy: Multi-start global optimization for Sutton-Chen transition-metal clusters.
Focus on FCC/HCP templates with adaptive basin hopping and parallel restarts.
"""

import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 310.0
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, compactly centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    # Estimate radius for n atoms in FCC
    R = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.8
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 4
    n_side = int(np.ceil(np.sqrt(n))) + 4
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift_x, shift_y = 0.0, 0.0
        else:
            shift_x = a / 2
            shift_y = a * np.sqrt(3) / 6
        
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift_x
                y = j * a * np.sqrt(3) / 2.0 + shift_y
                r2 = x**2 + y**2 + z**2
                R2 = (n / 4) ** (2/3) * a**2 * 4.0
                if r2 <= R2:
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        # Pad with random points near existing
        while len(points) < n:
            base = points[len(points) % len(points)]
            p = base + rng.normal(0, 0.5, 3)
            points = np.vstack([points, p])
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * (n / 4.0) ** (1/3.0) * 2.8
    points = []
    min_dist = 0.65 * spacing
    max_attempts = n * 80
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Icosahedral arrangement: center + 12 + 42 + ... shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    if n > 1:
        shell1 = ico_verts * spacing
        for v in shell1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        # Second shell: FCC-like at distance ~2*spacing
        a = spacing * np.sqrt(2)
        n_side = 5
        second_shell = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        r = np.linalg.norm(p)
                        if 1.6 * spacing < r < 2.4 * spacing:
                            second_shell.append(p / r * 2.0 * spacing)
        
        seen = set()
        unique_second = []
        for p in second_shell:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_second.append(p)
        
        for p in unique_second:
            if len(points) < n:
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-3.5*spacing, 3.5*spacing, 3)
        if np.linalg.norm(p) < 4.0 * spacing:
            points.append(p)
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _bcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * 2.0 / np.sqrt(3)
    R = (3 * n * a**3 / (8 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.8
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.3):
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def _basin_hopping(positions, potential, kind, rng, n_iters=100, base_scale=0.4, min_time=5.0):
    """
    Basin hopping with adaptive perturbation scale.
    """
    best_energy, best_pos = relax(positions, potential, kind)
    
    for i in range(n_iters):
        if _time_left() < min_time:
            break
        
        # Adaptive perturbation scale: start larger, decrease over time
        frac = i / max(1, n_iters - 1)
        scale = base_scale * (1.0 - 0.5 * frac) + 0.1
        new_pos = _perturb(best_pos, rng, scale)
        
        try:
            e_new, pos_new = relax(new_pos, potential, kind)
            if not np.isfinite(e_new):
                continue
            
            # Metropolis acceptance
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new
            else:
                T = 0.03 * abs(best_energy) + 0.01
                if rng.random() < np.exp(-(e_new - best_energy) / T):
                    best_energy = e_new
                    best_pos = pos_new
        except:
            pass
    
    return best_energy, best_pos

def _high_precision_polish(positions, potential, kind):
    """Final high-precision polish pass with tighter tolerance."""
    try:
        def obj(x):
            pos = x.reshape(-1, 3)
            e, _ = sutton_chen_energy_gradient(pos, kind)
            return e
        
        def grad(x):
            pos = x.reshape(-1, 3)
            _, g = sutton_chen_energy_gradient(pos, kind)
            return g.flatten()
        
        x0 = positions.flatten()
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
        )
        
        if result.success and result.fun < np.inf:
            return result.fun, result.x.reshape(-1, 3)
        else:
            return relax(positions, potential, kind)
    except:
        return relax(positions, potential, kind)

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    templates = []
    
    # FCC - primary template for SC potential (most important)
    try:
        fcc = _fcc_cluster(size, 1.0, rng)
        templates.append(fcc)
    except:
        pass
    
    # FCC with slightly different spacing
    try:
        fcc2 = _fcc_cluster(size, 0.95, rng)
        templates.append(fcc2)
    except:
        pass
    
    # HCP
    try:
        hcp = _hcp_cluster(size, 1.0, rng)
        templates.append(hcp)
    except:
        pass
    
    # Icosahedral
    try:
        ico = _icosahedral_shell(size, 1.0, rng)
        templates.append(ico)
    except:
        pass
    
    # BCC
    try:
        bcc = _bcc_cluster(size, 1.0, rng)
        templates.append(bcc)
    except:
        pass
    
    # Random compact (multiple with different seeds)
    for i in range(6):
        try:
            rand = _random_compact(size, 1.0, rng)
            templates.append(rand)
        except:
            pass
    
    # Relax each template and keep best
    for t in templates:
        if _time_left() < 10.0:
            break
        try:
            e, pos = relax(t, potential, kind)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    if best_positions is None:
        best_positions = templates[0]
    
    # Basin hopping from best structure with more iterations
    n_bh = max(80, int((_time_left() - 20.0) * 3))
    best_energy, best_positions = _basin_hopping(
        best_positions, potential, kind, rng, 
        n_iters=n_bh, base_scale=0.5
    )
    
    # Additional basin hopping from other good templates if time permits
    if _time_left() > 40.0:
        for t in templates[:5]:
            if _time_left() < 20.0:
                break
            try:
                e, pos = _basin_hopping(t, potential, kind, rng, n_iters=40, base_scale=0.4)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except:
                pass
    
    # Final high-precision polish pass
    if _time_left() > 8.0:
        try:
            e_polished, pos_polished = _high_precision_polish(best_positions, potential, kind)
            if e_polished < best_energy:
                best_energy = e_polished
                best_positions = pos_polished
        except:
            pass
    
    # Verify no atoms are too close
    dists = np.linalg.norm(best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.35:
        # Fallback: use a simple random structure
        best_positions = _random_compact(size, 1.0, rng)
    
    return best_positions
