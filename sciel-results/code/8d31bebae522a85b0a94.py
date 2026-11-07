"""Deterministic growth algorithm for Sutton-Chen transition-metal clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _sutton_chen_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Pair terms
    pair_rep = np.sum(dist ** (-n_exp))
    pair_att = np.sum(dist ** (-m_exp))
    
    # Self terms (remove diagonal)
    total_rep = np.sum(pair_rep)
    total_att = np.sum(pair_att)
    
    energy = 0.5 * total_rep - c * np.sum(np.sqrt(np.maximum(pair_att - np.diag(pair_att), 0)))
    return energy


def _sutton_chen_energy_grad(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-10, 1e-10, dist)
    
    # Pair repulsive term gradient: d/dr (r^-n) = -n * r^(-n-1)
    # Pair attractive term: we need the sum for each atom
    rep_force = n_exp * dist_safe ** (-n_exp - 1)
    att_sum = np.sum(dist_safe ** (-m_exp), axis=1) - np.diag(dist_safe ** (-m_exp))
    att_sum = np.maximum(att_sum, 1e-10)
    
    # Gradient of repulsive part
    grad_rep = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            grad_rep[i] += n_exp * dist_safe[i, j] ** (-n_exp - 1) * (positions[i] - positions[j]) / dist_safe[i, j]
    
    # Gradient of attractive part: -c * d/dr sqrt(sum_j r_ij^-m)
    # d/dr sqrt(S) = 1/(2*sqrt(S)) * dS/dr
    # dS_i/dr_i = sum_{j!=i} d(r_ij^-m)/dr_i = sum_{j!=i} (-m) * r_ij^(-m-1) * (r_i - r_j)/r_ij
    grad_att = np.zeros_like(positions)
    for i in range(n):
        sqrt_S = np.sqrt(att_sum[i])
        for j in range(n):
            if i == j:
                continue
            dS_dr = -m_exp * dist_safe[i, j] ** (-m_exp - 1) * (positions[i] - positions[j]) / dist_safe[i, j]
            grad_att[i] += dS_dr / (2 * sqrt_S)
    
    grad = 0.5 * grad_rep - c * grad_att
    energy = _sutton_chen_energy(positions, n_exp, m_exp, c)
    return energy, grad


def _get_params(kind):
    """Get Sutton-Chen parameters for given kind."""
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408),
    }
    return params.get(kind, (12, 6, 144.41))


def _relax_cluster(positions, kind, max_iter=1000):
    """Relax a cluster using L-BFGS-B."""
    n_exp, m_exp, c = _get_params(kind)
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _sutton_chen_energy_grad(pos, n_exp, m_exp, c)
        return e, g.ravel()
    
    result = minimize(fun, positions.ravel(), method='L-BFGS-B', 
                      jac=True, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    return result.fun, result.x.reshape(-1, 3)


def _generate_interstitial_candidates(cluster, n_exp, m_exp, c, num_candidates=200):
    """Generate candidate positions for new atom near existing cluster."""
    n = len(cluster)
    if n == 0:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Compute center of mass
    com = np.mean(cluster, axis=0)
    
    # Compute typical distance scale
    if n >= 2:
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(cluster[i] - cluster[j])
                if 0.3 < d < 3.0:
                    dists.append(d)
        if dists:
            scale = np.median(dists)
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    # Generate candidates in a shell around the cluster
    candidates = []
    rng = np.random.default_rng(42)
    
    # Try positions near existing atoms (interstitial sites)
    for atom in cluster:
        for _ in range(10):
            direction = rng.normal(size=3)
            direction /= (np.linalg.norm(direction) + 1e-10)
            r = scale * (0.5 + 0.5 * rng.random())
            candidate = atom + r * direction
            candidates.append(candidate)
    
    # Try positions in the bulk region
    for _ in range(100):
        direction = rng.normal(size=3)
        direction /= (np.linalg.norm(direction) + 1e-10)
        r = scale * (0.3 + 1.5 * rng.random())
        candidate = com + r * direction
        candidates.append(candidate)
    
    return np.array(candidates)


def _find_best_interstitial(cluster, kind, num_candidates=300):
    """Find the best position to add a new atom."""
    n_exp, m_exp, c = _get_params(kind)
    
    if len(cluster) == 0:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate candidates
    candidates = _generate_interstitial_candidates(cluster, n_exp, m_exp, c, num_candidates)
    
    best_energy = np.inf
    best_pos = None
    
    for cand in candidates:
        # Skip if too close to any existing atom
        if np.any(np.linalg.norm(cluster - cand, axis=1) < 0.4):
            continue
        
        new_cluster = np.vstack([cluster, cand.reshape(1, 3)])
        energy = _sutton_chen_energy(new_cluster, n_exp, m_exp, c)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = cand
    
    if best_pos is None:
        # Fallback: place at a reasonable distance
        com = np.mean(cluster, axis=0)
        best_pos = com + np.array([scale, 0, 0]) * 1.5
    
    return best_pos.reshape(1, 3)


def construct(instance):
    """Construct cluster using deterministic growth algorithm."""
    size = instance["n"]
    kind = instance["kind"]
    
    n_exp, m_exp, c = _get_params(kind)
    
    # Start with a small seed cluster
    if size <= 2:
        if size == 1:
            return np.array([[0.0, 0.0, 0.0]])
        else:
            return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Initialize with a small relaxed cluster
    # Start with a tetrahedron or small FCC fragment
    if size >= 4:
        # Tetrahedron
        a = 1.0
        cluster = np.array([
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a]
        ])
        # Relax it
        energy, cluster = _relax_cluster(cluster, kind)
    else:
        # Triangle
        a = 1.0
        cluster = np.array([
            [a, 0, 0],
            [-a/2, a*np.sqrt(3)/2, 0],
            [-a/2, -a*np.sqrt(3)/2, 0]
        ])
        energy, cluster = _relax_cluster(cluster, kind)
    
    # Grow the cluster one atom at a time
    while len(cluster) < size:
        # Find best interstitial site
        new_pos = _find_best_interstitial(cluster, kind, num_candidates=200)
        cluster = np.vstack([cluster, new_pos])
        
        # Relax the entire cluster
        energy, cluster = _relax_cluster(cluster, kind, max_iter=500)
        
        # Ensure minimum distance constraint
        # If any atoms are too close, nudge them apart
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                d = np.linalg.norm(cluster[i] - cluster[j])
                if d < 0.35:
                    direction = cluster[i] - cluster[j]
                    if np.linalg.norm(direction) < 1e-10:
                        direction = np.array([1.0, 0, 0])
                    direction /= np.linalg.norm(direction)
                    shift = (0.35 - d + 0.01) / 2
                    cluster[i] += shift * direction
                    cluster[j] -= shift * direction
    
    # Final relaxation with more iterations
    energy, cluster = _relax_cluster(cluster, kind, max_iter=2000)
    
    # Final check: ensure no atoms are too close
    for i in range(len(cluster)):
        for j in range(i+1, len(cluster)):
            d = np.linalg.norm(cluster[i] - cluster[j])
            if d < 0.35:
                direction = cluster[i] - cluster[j]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0, 0])
                direction /= np.linalg.norm(direction)
                shift = (0.35 - d + 0.01) / 2
                cluster[i] += shift * direction
                cluster[j] -= shift * direction
    
    # Verify finite coordinates
    if not np.all(np.isfinite(cluster)):
        # Fallback to random construction
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0/3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 0.72 for q in points):
                points.append(p)
        _, cluster = relax(np.array(points), 'sutton_chen', kind)
    
    return cluster
# EVOLVE-BLOCK-END
