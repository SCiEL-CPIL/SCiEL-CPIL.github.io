"""Dzugutov cluster global minima: Mackay icosahedral seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 290.0  # seconds, safety margin before 300s kill
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a
    #      + B exp(d/(r-b)) for r < b
    energy = 0.0
    mask_a = r < a
    mask_b = (r >= a) & (r < b)
    
    if np.any(mask_a):
        r_a = r[mask_a]
        v_a = A * (r_a ** (-m) - B) * np.exp(c / (r_a - a))
        energy += np.sum(v_a)
    
    if np.any(mask_b):
        r_b = r[mask_b]
        v_b = B * np.exp(d / (r_b - b))
        energy += np.sum(v_b)
    
    # Each pair counted twice in the sum, so divide by 2
    energy /= 2.0
    return energy


def _generate_icosahedral_shell(n_atoms, scale=1.0, twist=0.0):
    """Generate atoms on a Mackay icosahedral structure."""
    # Icosahedron vertices (golden ratio)
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # Mackay icosahedron: layers at radii corresponding to icosahedron vertices
    # Core: 1 atom at center
    # First shell: 12 atoms (icosahedron vertices)
    # Second shell: 30 atoms (edges of icosahedron)
    # Third shell: 20 atoms (faces of icosahedron)
    # Fourth shell: 60 atoms (second generation vertices)
    # etc.
    
    positions = [np.zeros(3)]  # center
    
    # First shell: 12 icosahedron vertices
    for v in vertices:
        positions.append(v * scale)
    
    # Second shell: 30 edge midpoints scaled
    # Edges of icosahedron: pairs of vertices at distance 2/sqrt(1+phi^2)
    edge_pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < 1.1:  # edge threshold
                edge_pairs.append((i, j))
    
    for i, j in edge_pairs:
        midpoint = (vertices[i] + vertices[j]) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)
        positions.append(midpoint * 2.0 * scale)
    
    # Third shell: 20 face centers
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(vertices[i] - vertices[j])
                d2 = np.linalg.norm(vertices[j] - vertices[k])
                d3 = np.linalg.norm(vertices[k] - vertices[i])
                if d1 < 1.1 and d2 < 1.1 and d3 < 1.1:
                    face_center = (vertices[i] + vertices[j] + vertices[k]) / 3
                    face_center = face_center / np.linalg.norm(face_center)
                    faces.append(face_center)
    
    for fc in faces:
        positions.append(fc * 2.4 * scale)
    
    # Fourth shell: 60 atoms (second generation)
    # These are at the midpoints of edges of the second shell
    # Approximate by taking vertices at radius ~3.0
    for v in vertices:
        for _ in range(5):
            # Add perturbations to create 5 positions near each vertex direction
            angle = np.random.uniform(0, 2*np.pi)
            offset = np.array([np.cos(angle), np.sin(angle), 0]) * 0.3
            p = v * 3.0 * scale + offset
            positions.append(p)
    
    # If we have more than n_atoms, trim; if fewer, add random positions
    positions = np.array(positions)
    
    if len(positions) > n_atoms:
        # Keep the most central ones
        radii = np.linalg.norm(positions, axis=1)
        idx = np.argsort(radii)[:n_atoms]
        positions = positions[idx]
    elif len(positions) < n_atoms:
        # Add random positions in a shell
        remaining = n_atoms - len(positions)
        for _ in range(remaining):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            r = np.random.uniform(2.5, 3.5) * scale
            positions = np.vstack([positions, direction * r])
    
    # Apply twist rotation to upper half
    if twist != 0:
        center = np.linalg.norm(positions, axis=1)
        upper = center > 1.5 * scale
        cos_t = np.cos(twist)
        sin_t = np.sin(twist)
        rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        positions[upper] = positions[upper] @ rot.T
    
    return positions


def _fix_minimum_distance(pos, min_dist=0.5, max_iter=100):
    """Fix any pairs closer than min_dist by pushing them apart."""
    pos = pos.copy()
    for _ in range(max_iter):
        n = len(pos)
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    if d < 1e-10:
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = (pos[i] - pos[j]) / d
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
                    moved = True
        if not moved:
            break
    return pos


def _basin_hopping(pos, potential, parameter, rng, max_iter=50, kick_scale=0.3):
    """Perform basin hopping: perturb, relax, keep if better."""
    current_pos = pos.copy()
    current_energy = _dzugutov_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for i in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Random kick
        kick = rng.normal(0, kick_scale, size=current_pos.shape)
        new_pos = current_pos + kick
        new_pos = _fix_minimum_distance(new_pos, 0.5)
        
        # Relax
        try:
            energy, relaxed_pos = relax(new_pos, potential, parameter)
        except:
            continue
        
        # Accept if better
        if energy < current_energy:
            current_pos = relaxed_pos
            current_energy = energy
        
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
        
        # Occasionally increase kick scale
        if i % 10 == 9:
            kick_scale *= 1.1
    
    return best_pos, best_energy


def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Mackay icosahedral seeds with various twists and scales
    for twist in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for scale in [0.9, 1.0, 1.1]:
            pos = _generate_icosahedral_shell(size, scale=scale, twist=twist)
            pos = _fix_minimum_distance(pos, 0.5)
            candidates.append(pos)
    
    # Random compact clusters
    radius = 1.1 * size ** (1.0 / 3.0) * 0.75
    for _ in range(8):
        points = []
        attempts = 0
        while len(points) < size and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            candidates.append(np.array(points))
    
    # Run basin hopping on all candidates in parallel
    best_pos = None
    best_energy = np.inf
    
    def process_candidate(pos):
        if _remaining() < 1.0:
            return None
        try:
            # First relax
            energy, relaxed = relax(pos, potential, parameter)
            # Then basin hopping
            pos_bh, energy_bh = _basin_hopping(relaxed, potential, parameter, rng, max_iter=30)
            return (energy_bh, pos_bh)
        except:
            return None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_candidate, pos) for pos in candidates]
        for future in as_completed(futures):
            if _remaining() < 1.0:
                break
            result = future.result()
            if result is not None:
                energy, pos = result
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
    
    # Final high-precision polish
    if best_pos is not None and _remaining() > 3.0:
        try:
            energy, polished = relax(best_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = polished
        except:
            pass
    
    if best_pos is None:
        # Fallback: return first candidate
        best_pos = candidates[0]
    
    return best_pos
# EVOLVE-BLOCK-END
