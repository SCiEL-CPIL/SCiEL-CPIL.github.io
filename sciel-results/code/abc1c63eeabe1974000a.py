"""Sutton-Chen transition-metal cluster global minima: multi-start from diverse
structured seeds (fcc, hcp, icosahedral, Mackay), followed by basin hopping
refinement with Metropolis acceptance on the best candidates.
Uses a strict wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 318.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators: produce compact, valid (min-pair-distance) (n,3) arrays.
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    """Fill the fcc lattice with the n sites closest to the origin."""
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cell = np.array([i, j, k])
                for b in basis:
                    p = (cell + b)
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    """Fill an hcp-like lattice (two interpenetrating triangular layers)."""
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * h
        is_b = (layer % 2 != 0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3.0) / 2.0) * j
                if is_b:
                    x += a * 0.5
                    y += a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _icosahedral_seed(n, rng, scale=1.0):
    """Start from an icosahedron and fill with fcc points."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts_list = [p * 1.0 for p in ico]
    fcc = _fcc_seed(n, rng, scale=1.0)
    all_pts = list(pts_list)
    for p in fcc:
        if len(all_pts) >= n:
            break
        ok = True
        for q in all_pts:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            all_pts.append(p)
    if len(all_pts) < n:
        for p in fcc:
            if len(all_pts) >= n:
                break
            all_pts.append(p)
    pts = np.array(all_pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _bcp_seed(n, rng, scale=1.0):
    """Bicapped icosahedron / Mackay-like seed for transition metals."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    
    # First shell: 12 icosahedron vertices
    for v in ico_verts:
        pts.append(v * 1.0)
    
    # Second shell: 30 points (edges of icosahedron midpoints)
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.0:  # edge length ~1
                    edges.append((i, j))
        for i, j in edges:
            if len(pts) >= n:
                break
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            pts.append(mid * 1.62)  # slightly larger radius
    
    # Third shell: 20 points (face centers)
    if n > 43:
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if d1 < 1.0 and d2 < 1.0 and d3 < 1.0:
                        faces.append((i, j, k))
        for i, j, k in faces:
            if len(pts) >= n:
                break
            centroid = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            pts.append(centroid * 2.1)
    
    # Fill remaining with fcc
    if len(pts) < n:
        fcc = _fcc_seed(n, rng, scale=1.0)
        for p in fcc:
            if len(pts) >= n:
                break
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4:
                    ok = False
                    break
            if ok:
                pts.append(p)
    
    pts = np.array(pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    """Random points in a ball with a minimum-pair-distance constraint."""
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 30000
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    if len(pts) < n:
        md = min_dist
        while len(pts) < n:
            md *= 0.95
            if md < 0.36:
                while len(pts) < n:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                break
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= md for q in pts):
                    pts.append(p)
    pts = np.array(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    return pts


def _jitter(seed, rng, amount):
    return seed + rng.normal(0.0, amount, seed.shape)


def _is_valid(pts, min_dist=0.35):
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    if n < 2:
        return True
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist


def _try_relax(seed, potential, kind, maxiter=3000):
    """Relax a seed; return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    parameter = kind

    base_scale = 1.0
    rng = np.random.default_rng(12345 + n)

    # Build a diverse list of seeds
    seeds = []

    # 1. Structured seeds (Mackay icosahedral, fcc, hcp, icosahedral)
    for gen in (_bcp_seed, _fcc_seed, _hcp_seed, _icosahedral_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        # Jittered variants to break symmetry
        for amt in (0.05, 0.12, 0.2, 0.3):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    # 2. Random compact seeds
    for _ in range(8):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.55)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None
    local_minima = [] # Store (energy, positions) for basin hopping

    # Phase 1: Relax seeds one at a time under the deadline
    for seed in seeds:
        if _remaining() < 15.0:
            break
        e, pos = _try_relax(seed, potential, parameter)
        if e < best_energy:
            best_energy = e
            best_positions = pos
        if pos is not None:
            local_minima.append((e, pos))

    # Fallback if nothing valid
    if best_positions is None:
        s = _fcc_seed(n, rng, scale=base_scale)
        if not _is_valid(s, 0.35):
            s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.36)
        e, pos = _try_relax(s, potential, parameter)
        if pos is not None:
            best_energy = e
            best_positions = pos
            local_minima.append((e, pos))
        else:
            side = int(np.ceil(n ** (1.0 / 3.0)))
            grid = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        grid.append([i - side // 2, j - side // 2, k - side // 2])
            pts = np.array(grid[:n], dtype=float) * 0.75
            best_positions = pts

    # Phase 2: Basin Hopping Refinement
    # We take the top few local minima and apply basin hopping to them.
    # This allows escaping local minima that the initial relaxation got stuck in.
    
    # Sort local minima by energy
    local_minima.sort(key=lambda x: x[0])
    
    # Take the best 3 unique minima (or fewer if not available)
    candidates = []
    seen_energies = set()
    for e, pos in local_minima:
        # Simple check to avoid duplicate candidates
        if e < 1e-9: 
             continue
        candidates.append((e, pos))
        if len(candidates) >= 3:
            break
            
    # If we have no candidates (should not happen), use best_positions
    if not candidates and best_positions is not None:
        candidates.append((best_energy, best_positions))

    for start_e, start_pos in candidates:
        current_e = start_e
        current_pos = start_pos.copy()
        
        # Basin Hopping Loop
        # Coarse-to-fine amplitude schedule
        amplitudes = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]
        
        # Metropolis temperature schedule (linearly cooling)
        T_start = 0.5
        T_end = 0.01
        total_steps = sum(5 if amp > 0.1 else 3 for amp in amplitudes)
        step_idx = 0
        
        for amp in amplitudes:
            if _remaining() < 5.0:
                break
            
            # Perform a few perturb-relax cycles at this amplitude
            # Number of cycles decreases as amplitude decreases (focus on fine-tuning)
            num_cycles = 5 if amp > 0.1 else 3
            
            for _ in range(num_cycles):
                if _remaining() < 5.0:
                    break
                
                # Current temperature for Metropolis acceptance
                T = T_start * (T_end / T_start) ** (step_idx / max(1, total_steps - 1))
                step_idx += 1
                
                # Perturb
                perturbed = _jitter(current_pos, rng, amp)
                
                # Relax
                e_new, pos_new = _try_relax(perturbed, potential, parameter)
                
                if pos_new is None:
                    continue
                    
                # Metropolis acceptance criterion
                delta_e = e_new - current_e
                if delta_e < 0:
                    accept = True
                else:
                    if T > 1e-6:
                        accept = rng.random() < np.exp(-delta_e / T)
                    else:
                        accept = False
                
                if accept:
                    current_e = e_new
                    current_pos = pos_new
                
                # Update global best
                if current_e < best_energy:
                    best_energy = current_e
                    best_positions = current_pos.copy()

    return best_positions
# EVOLVE-BLOCK-END
