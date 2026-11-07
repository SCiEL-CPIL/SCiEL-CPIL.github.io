"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _get_params(kind):
    """Get potential parameters."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _energy_of_pair(r, n_exp, m_exp, c):
    """Compute pair energy for a single pair at distance r."""
    if r < 0.35:
        return np.inf
    return r ** (-n_exp) - c * r ** (-m_exp / 2.0)


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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


def _generate_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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


def _generate_hcp(n):
    """Generate n atoms in HCP-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    a = 1.0
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            for layer in range(4):
                z = layer * 0.816
                if layer % 2 == 1:
                    x += 1.0 / 3
                    y += 1.0 / np.sqrt(3)
                p = np.array([x, y, z], dtype=float)
                candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedral(n):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
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
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
        rng = np.random.default_rng(123 + n)
        remaining = n - len(pts)
        dirs = rng.normal(size=(remaining, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radius = 1.9
        for d in dirs:
            pts.append(d * radius)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.3 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _perturb(positions, rng, sigma):
    """Add Gaussian noise to positions."""
    return positions + rng.normal(0, sigma, positions.shape)


def _growth_cluster(n, kind, seed=42):
    """
    Deterministic atom-by-atom growth algorithm.
    Start with 2 atoms, then add one atom at a time at the position
    that minimizes the total energy contribution.
    """
    n_exp, m_exp, c = _get_params(kind)
    rng = np.random.default_rng(seed)
    
    # Start with 2 atoms at reasonable distance
    # Equilibrium distance for Sutton-Chen: where r^-n - c*r^(-m/2) is minimized
    # d/dr [r^-n - c*r^(-m/2)] = -n*r^(-n-1) + c*m/2 * r^(-m/2-1) = 0
    # n*r^(-n) = c*m/2 * r^(-m/2)
    # r^(n-m/2) = 2n/(c*m)
    r_eq = (2.0 * n_exp / (c * m_exp)) ** (1.0 / (n_exp - m_exp / 2.0))
    
    pts = [np.zeros(3), np.array([r_eq, 0, 0], dtype=float)]
    
    for target_n in range(3, n + 1):
        if _time_left() < 15.0:
            break
        
        # Generate candidate positions for the new atom
        # Strategy: place near existing atoms at various offsets
        candidates = []
        
        # For each existing atom, try positions at various distances and angles
        for i, p_i in enumerate(pts):
            # Try distances from 0.7*r_eq to 1.5*r_eq
            for r_mult in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
                r_new = r_mult * r_eq
                # Try various directions
                for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    for phi in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                        dx = r_new * np.sin(phi) * np.cos(theta)
                        dy = r_new * np.sin(phi) * np.sin(theta)
                        dz = r_new * np.cos(phi)
                        p_new = p_i + np.array([dx, dy, dz])
                        candidates.append(p_new)
        
        # Also add some random candidates
        for _ in range(20):
            angle1 = rng.uniform(0, 2 * np.pi)
            angle2 = rng.uniform(-np.pi/2, np.pi/2)
            r_new = rng.uniform(0.7, 1.5) * r_eq
            p_new = np.array([
                r_new * np.cos(angle1) * np.cos(angle2),
                r_new * np.sin(angle1) * np.cos(angle2),
                r_new * np.sin(angle2)
            ])
            candidates.append(p_new)
        
        # Evaluate each candidate by computing the energy contribution
        best_candidate = None
        best_energy = np.inf
        
        for p_new in candidates:
            # Check minimum distance constraint
            dists = np.linalg.norm(np.array(pts) - p_new, axis=1)
            if np.any(dists < 0.35):
                continue
            
            # Compute energy contribution of adding this atom
            e_contrib = 0.0
            for p_i in pts:
                r = np.linalg.norm(p_i - p_new)
                if r < 0.35:
                    e_contrib = np.inf
                    break
                e_contrib += _energy_of_pair(r, n_exp, m_exp, c)
            
            if e_contrib < best_energy:
                best_energy = e_contrib
                best_candidate = p_new
        
        if best_candidate is None:
            # Fallback: place at a random position
            best_candidate = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)])
        
        pts.append(best_candidate)
    
    return np.array(pts[:n])


def _growth_cluster_with_relax(n, kind, seed=42):
    """Growth cluster followed by relaxation."""
    base = _growth_cluster(n, kind, seed)
    if not _is_valid(base):
        base = base * 1.2
    if not _is_valid(base):
        return None, None
    energy, relaxed = relax(base, 'sutton_chen', kind)
    if np.isfinite(energy) and _is_valid(relaxed):
        return energy, relaxed
    return None, None


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Time budget for this instance
    instance_budget = max(5.0, _time_left() - 10.0)
    
    # Phase 1: Try growth-based generators with multiple seeds
    for seed in [42, 123, 456, 789, 1011, 2022]:
        if _time_left() < 15.0:
            break
        try:
            energy, relaxed = _growth_cluster_with_relax(n, kind, seed)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Phase 2: Try lattice generators
    generators = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedral]
    for gen in generators:
        if _time_left() < 15.0:
            break
        try:
            base = gen(n)
            if not _is_valid(base):
                base = base * 1.2
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Phase 3: Random restarts
    rng = np.random.default_rng(42 + n + hash(kind) % 1000)
    n_restarts = 30
    for i in range(n_restarts):
        if _time_left() < 15.0:
            break
        
        if best_pos is not None and i % 3 == 0:
            sigma = 0.1 + 0.1 * (i / n_restarts)
            pos = _perturb(best_pos, rng, sigma)
        else:
            pos = _generate_random(n, 42 + i * 7 + n)
        
        if not _is_valid(pos):
            continue
            
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Phase 4: Basin hopping on best found
    if best_pos is not None and _time_left() > 15.0:
        for i in range(15):
            if _time_left() < 10.0:
                break
            sigma = 0.05 + 0.05 * (i / 15)
            pos = _perturb(best_pos, rng, sigma)
            if not _is_valid(pos):
                continue
            try:
                energy, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-2, 2, (n, 3))
            for attempt in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
