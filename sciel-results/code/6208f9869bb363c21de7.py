"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 150s, but we must return before that.
# We'll use a deadline of 110s to be safe.
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions of shape (N, 3)."""
    n = pos.shape[0]
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero (shouldn't happen if min distance > 0.5)
    np.fill_diagonal(r2, 1.0)
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    # Sum over all pairs, but we've counted each pair twice, so divide by 2
    # Also subtract the diagonal terms (which we set to 1, so r6_inv=1, r12_inv=1, energy=0)
    total = 4.0 * np.sum(r12_inv - r6_inv) / 2.0
    # The diagonal terms: r2=1, r6_inv=1, r12_inv=1, so 4*(1-1)=0, so they don't contribute
    return total

def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2
    # dE/dr2 = 4 * (-6*r2^-7 + 12*r2^-13) * 2 * (pos_i - pos_j) ... 
    # Actually: E_pair = 4*(r^-12 - r^-6)
    # dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos_i = dE/dr * d r/dpos_i = dE/dr * (pos_i - pos_j)/r
    # = 4*(-12*r^-13 + 6*r^-7) * (pos_i - pos_j)/r
    # = 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # In terms of r2: r^-14 = r2^-7, r^-8 = r2^-4
    # dE/dpos_i = 4*(-12*r2^-7 + 6*r2^-4) * diff[i,j]
    # Note: diff[i,j] = pos_i - pos_j
    # The gradient for atom i is sum over j != i of 4*(-12*r2[i,j]^-7 + 6*r2[i,j]^-4) * diff[i,j]
    # Let's compute this efficiently
    # factor[i,j] = 4*(-12*r2_inv[i,j]^7 + 6*r2_inv[i,j]^4)
    r2_inv4 = r2_inv ** 4
    r2_inv7 = r2_inv ** 7
    factor = 4.0 * (-12.0 * r2_inv7 + 6.0 * r2_inv4)  # (n, n)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
    return grad

def _construct_mackay_icosahedron():
    """
    Construct a Mackay icosahedron with 55 atoms (two shells).
    Shell 1: 12 atoms on the vertices of a regular icosahedron
    Shell 2: 42 atoms on the edges and faces of the icosahedron
    
    Actually, the Mackay icosahedron with n=2 has:
    - 1 central atom
    - 12 atoms at distance 1 (first shell, vertices of icosahedron)
    - 30 atoms at distance sqrt(2) (second shell, midpoints of edges)
    - 12 atoms at distance 2 (second shell, vertices of larger icosahedron)
    Total: 1 + 12 + 30 + 12 = 55
    
    Let's construct this properly.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Vertices of a regular icosahedron (12 vertices)
    # Standard coordinates:
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    icos_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit length
    norms = np.linalg.norm(icos_vertices, axis=1, keepdims=True)
    icos_unit = icos_vertices / norms
    
    # The LJ minimum is at r = 2^(1/6) ≈ 1.12246
    # For the Mackay icosahedron, we need to scale appropriately.
    # The known energy is -279.248470 for the optimal scaling.
    # Let's parameterize: scale the icosahedron by factor s.
    # Atom 0: center at origin
    # Atoms 1-12: at s * icos_unit[i]
    # Atoms 13-42: midpoints of edges, at s * (icos_unit[i] + icos_unit[j])/2 for each edge
    # Atoms 43-54: at 2*s * icos_unit[i]
    
    # We need to find the edges of the icosahedron.
    # Two vertices are connected if their distance is the edge length.
    # For unit icosahedron, edge length = 2 (since vertices like (0,1,phi) and (0,-1,phi) have distance 2)
    # Wait, let's check: (0,1,phi) to (0,-1,phi): distance = 2. (0,1,phi) to (1,phi,0): distance = sqrt(1 + (phi-1)^2 + phi^2)
    # phi ≈ 1.618, phi-1 ≈ 0.618, phi^2 ≈ 2.618
    # distance = sqrt(1 + 0.382 + 2.618) = sqrt(4) = 2. Good.
    
    # Find all edges (pairs of vertices with distance = 2 for unit icosahedron)
    n_ico = 12
    edges = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            d = np.linalg.norm(icos_unit[i] - icos_unit[j])
            if np.isclose(d, 2.0 / np.linalg.norm(icos_vertices[0]), atol=1e-10):
                edges.append((i, j))
    
    # Actually, let me just compute distances and find the minimum
    dists = np.zeros((n_ico, n_ico))
    for i in range(n_ico):
        for j in range(n_ico):
            dists[i,j] = np.linalg.norm(icos_unit[i] - icos_unit[j])
    min_dist = dists[np.triu_indices(n_ico, k=1)].min()
    edges = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            if np.isclose(dists[i,j], min_dist, atol=1e-10):
                edges.append((i, j))
    
    # Construct the 55 atoms
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]  # Center
    
    # First shell: 12 atoms
    for i in range(12):
        pos[1+i] = icos_unit[i]
    
    # Second shell: 30 edge midpoints
    for k, (i, j) in enumerate(edges):
        pos[13+k] = 0.5 * (icos_unit[i] + icos_unit[j])
    
    # Second shell: 12 outer vertices
    for i in range(12):
        pos[43+i] = 2.0 * icos_unit[i]
    
    return pos

def _relax(pos, max_iter=200, tol=1e-10):
    """Relax positions using L-BFGS."""
    def fun(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad(x):
        return _lj_grad(x.reshape(-1, 3))
    
    result = minimize(fun, pos.flatten(), jac=grad, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    return result.x.reshape(-1, 3), result.fun

def construct():
    start_time = time.time()
    
    # Strategy: 
    # 1. Start with Mackay icosahedron (known to be near global minimum)
    # 2. Relax it with L-BFGS
    # 3. Also try other initializations and keep the best
    
    best_pos = None
    best_energy = np.inf
    
    # Deadline check helper
    def time_left():
        return DEADLINE_SECONDS - (time.time() - start_time)
    
    # Try 1: Mackay icosahedron with optimal scaling
    # The known minimum energy is -279.248470
    # Let's try to find the right scaling by relaxing
    if time_left() > 10:
        mackay = _construct_mackay_icosahedron()
        # Try different scalings
        for scale in [1.0, 1.05, 0.95, 1.1, 0.9, 1.15, 0.85]:
            if time_left() < 5:
                break
            pos_scaled = mackay * scale
            # Ensure minimum distance >= 0.5
            min_d = np.min(np.linalg.norm(pos_scaled[:, np.newaxis] - pos_scaled[np.newaxis, :], axis=2)[np.triu_indices(55, k=1)])
            if min_d >= 0.5:
                pos_relaxed, energy = _relax(pos_scaled, max_iter=100, tol=1e-12)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos_relaxed.copy()
    
    # Try 2: Random liquid-like initializations
    if time_left() > 30:
        np.random.seed(42)
        for trial in range(5):
            if time_left() < 20:
                break
            # Random positions in a sphere
            r = np.random.uniform(0, 4, size=(55, 3))
            theta = np.random.uniform(0, 2*np.pi, size=55)
            phi = np.random.uniform(0, np.pi, size=55)
            pos_rand = np.zeros((55, 3))
            pos_rand[:, 0] = r * np.sin(phi) * np.cos(theta)
            pos_rand[:, 1] = r * np.sin(phi) * np.sin(theta)
            pos_rand[:, 2] = r * np.cos(phi)
            
            # Check min distance
            min_d = np.min(np.linalg.norm(pos_rand[:, np.newaxis] - pos_rand[np.newaxis, :], axis=2)[np.triu_indices(55, k=1)])
            if min_d >= 0.5:
                pos_relaxed, energy = _relax(pos_rand, max_iter=100, tol=1e-10)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos_relaxed.copy()
    
    # Final high-precision polish
    if best_pos is not None and time_left() > 10:
        pos_polished, energy_polished = _relax(best_pos, max_iter=500, tol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
