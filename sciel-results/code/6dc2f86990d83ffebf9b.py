import numpy as np
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0

def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN

def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.35:
        return False
    return True

def _icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_mackay_icosahedron(n):
    """
    Generate Mackay icosahedral clusters.
    n=13: 1 + 12
    n=55: 1 + 12 + 42
    """
    if n == 13:
        verts = _icosahedron_vertices()
        # Scale so edge length is ~1.0? 
        # Edge length of unit icosahedron (radius 1) is sqrt(2 + 2/phi) ~ 1.902
        # We want to match SC equilibrium. SC potential min is around r=1.0-1.1.
        # Let's scale to radius ~0.5 so edge is ~0.95.
        center = np.zeros(3)
        return np.vstack([center, verts * 0.5])
    
    elif n == 55:
        verts1 = _icosahedron_vertices()
        
        # We need 42 more atoms.
        # The 2nd shell of Mackay icosahedron consists of:
        # - 12 face centers
        # - 30 edge midpoints
        # We need the connectivity of the icosahedron to find edges and faces.
        
        # Let's hardcode the faces and edges of the standard icosahedron defined above.
        # Vertices indices:
        # 0: [0, 1, phi]
        # 1: [0, -1, phi]
        # 2: [0, 1, -phi]
        # 3: [0, -1, -phi]
        # 4: [1, phi, 0]
        # 5: [-1, phi, 0]
        # 6: [1, -phi, 0]
        # 7: [-1, -phi, 0]
        # 8: [phi, 0, 1]
        # 9: [phi, 0, -1]
        # 10: [-phi, 0, 1]
        # 11: [-phi, 0, -1]
        
        # Faces (20 triangles)
        faces = [
            (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (4, 8, 0), # Top cap? No, let's be careful.
            # Standard icosahedron faces:
            # Let's just compute edges by distance.
        ]
        
        # Compute all pairwise distances to find edges (shortest distance)
        d = np.linalg.norm(verts1[:, np.newaxis] - verts1[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        # Edge is if distance is close to min_d
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(d[i,j] - min_d) < 1e-6:
                    edges.append((i, j))
        
        # Edge midpoints
        edge_midpoints = []
        for i, j in edges:
            mid = (verts1[i] + verts1[j]) / 2.0
            edge_midpoints.append(mid)
        edge_midpoints = np.array(edge_midpoints)
        
        # Face centers
        # We need to find the 20 faces.
        # A face is a triplet of vertices that are mutually connected by edges.
        # We have the list of edges.
        # Let's create an adjacency matrix.
        adj = np.zeros((12, 12), dtype=bool)
        for i, j in edges:
            adj[i, j] = True
            adj[j, i] = True
            
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                if adj[i, j]:
                    for k in range(j+1, 12):
                        if adj[i, k] and adj[j, k]:
                            faces.append((i, j, k))
                            
        # There should be 20 faces.
        face_centers = []
        for i, j, k in faces:
            center = (verts1[i] + verts1[j] + verts1[k]) / 3.0
            face_centers.append(center)
        face_centers = np.array(face_centers)
        
        # The 2nd shell atoms are edge_midpoints and face_centers.
        # But they are not at the same radius.
        # In Mackay icosahedron, the 2nd shell is a distorted icosphere.
        # The edge midpoints are closer to the center than face centers?
        # Actually, in a regular icosahedron, the distance from center to edge midpoint is:
        # r_edge = 0.5 * sqrt(2 + 2/phi) / sqrt(1 + 1/phi^2) ... no.
        # Let's just project them onto a sphere of radius R2.
        # What is R2?
        # In Mackay icosahedron, the distance between 1st shell atom and 2nd shell atom is roughly the same as 1st shell edge length.
        # Let's assume R1 = 0.5 (from our scaling).
        # Edge length L = 0.5 * sqrt(2 + 2/phi) ~ 0.951.
        # We want the 2nd shell to be at a radius such that the cluster is compact.
        # Usually, R2 is around 1.5 * R1? Or determined by packing.
        # Let's try to scale the raw midpoints and face centers to have a consistent radius.
        # The distance from origin to edge midpoint of unit icosahedron:
        # v1 = [0, 1, phi], v2 = [1, phi, 0]. mid = [0.5, (1+phi)/2, phi/2].
        # norm(mid) = sqrt(0.25 + (1+phi)^2/4 + phi^2/4) = 0.5 * sqrt(1 + (1+phi)^2 + phi^2)
        # (1+phi)^2 + phi^2 = 1 + 2phi + phi^2 + phi^2 = 1 + 2phi + 2phi^2 = 1 + 2phi + 2(1+phi) = 3 + 4phi.
        # sqrt(3 + 4phi) ~ sqrt(3 + 6.472) = sqrt(9.472) ~ 3.077.
        # So r_edge_unit = 0.5 * 3.077 / 2? No, the formula was 0.5 * sqrt(...).
        # Wait, mid = (v1+v2)/2. |mid| = 0.5 * |v1+v2|.
        # |v1+v2|^2 = |v1|^2 + |v2|^2 + 2 v1.v2.
        # v1.v2 = 0*1 + 1*phi + phi*0 = phi.
        # |v1|=1. So |v1+v2|^2 = 1 + 1 + 2phi = 2 + 2phi.
        # |v1+v2| = sqrt(2+2phi) ~ 1.902.
        # So r_edge_unit = 0.5 * 1.902 = 0.951.
        
        # Distance from origin to face center of unit icosahedron:
        # Face center is (v1+v2+v3)/3.
        # v1, v2, v3 are mutually orthogonal? No.
        # v1=[0,1,phi], v2=[1,phi,0], v3=[phi,0,1]? No, these are not a face.
        # Let's pick a face: (0, 1, 4) -> v0, v1, v4.
        # v0=[0,1,phi], v1=[0,-1,phi], v4=[1,phi,0].
        # sum = [1, phi, 2phi].
        # |sum|^2 = 1 + phi^2 + 4phi^2 = 1 + 5phi^2.
        # phi^2 = phi+1 ~ 2.618. 5*2.618 = 13.09. Sum = 14.09.
        # |sum| ~ 3.75.
        # r_face_unit = 3.75 / 3 = 1.25.
        
        # So in the unit icosahedron (radius 1), edge midpoints are at r=0.951 and face centers at r=1.25.
        # In our cluster, we scaled 1st shell to radius 0.5.
        # So we should scale the 2nd shell points by 0.5 as well?
        # If we do, edge midpoints are at 0.475 and face centers at 0.625.
        # This creates a "spiky" 2nd shell.
        # In Mackay icosahedron, the 2nd shell is a "truncated icosahedron" like shape? No, it's just a shell of 42 atoms.
        # The atoms relax to minimize energy.
        # Let's just generate the points at their geometric positions scaled by 0.5 and let relax handle it.
        
        shell2 = np.vstack([edge_midpoints, face_centers]) * 0.5
        
        center = np.zeros(3)
        all_pts = np.vstack([center, verts1 * 0.5, shell2])
        
        return all_pts

    return None

def _generate_fcc_chunk(n):
    """
    Generate an FCC-like cluster.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    a = 1.0
    tri_nums = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105]
    
    def get_tri_layer(k, z=0.0):
        pts = []
        for i in range(k):
            for j in range(k - i):
                x = i * a + j * a * 0.5
                y = j * a * np.sqrt(3) / 2
                pts.append([x, y, z])
        return np.array(pts) if pts else np.array([])
    
    points = []
    z = 0.0
    dz = a * np.sqrt(2/3)
    
    current_total = 0
    layer_idx = 0
    
    while current_total < n and layer_idx < len(tri_nums):
        k = layer_idx + 1
        num_atoms = tri_nums[layer_idx]
        
        layer_pts = get_tri_layer(k, z)
        
        if layer_idx % 2 == 1:
            offset_x = a / 3
            offset_y = a / np.sqrt(3)
            layer_pts[:, 0] += offset_x
            layer_pts[:, 1] += offset_y
            
        points.append(layer_pts)
        current_total += num_atoms
        z += dz
        layer_idx += 1
        
    if not points:
        return np.zeros((n, 3))
        
    all_pts = np.vstack(points)
    center = all_pts.mean(axis=0)
    all_pts -= center
    
    if len(all_pts) > n:
        dists = np.linalg.norm(all_pts, axis=1)
        order = np.argsort(dists)
        all_pts = all_pts[order[:n]]
        
    return all_pts

def _generate_bcc_chunk(n):
    """
    Generate a BCC-like cluster.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    a = 1.0
    L = int(np.cbrt(n / 2.0)) + 1
    
    pts = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                pts.append([i*a, j*a, k*a])
                if i < L-1 and j < L-1 and k < L-1:
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    all_pts = np.array(pts)
    center = all_pts.mean(axis=0)
    all_pts -= center
    
    if len(all_pts) > n:
        dists = np.linalg.norm(all_pts, axis=1)
        order = np.argsort(dists)
        all_pts = all_pts[order[:n]]
        
    return all_pts

def _generate_random_compact(n, rng, radius=0.8):
    """
    Generate a random compact cluster.
    """
    if n <= 0:
        return np.zeros((0, 3))
    
    R = radius * n ** (1.0/3.0) * 1.5
    
    points = [np.zeros(3)]
    min_dist = 0.4
    
    attempts = 0
    max_attempts = n * 2000
    
    while len(points) < n and attempts < max_attempts:
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r = R * np.cbrt(rng.uniform(0, 1))
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-R, R, 3))
        
    return np.array(points)

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_positions = None
    
    starts = []
    
    # 1. Structural Seeds
    ico = _generate_mackay_icosahedron(n)
    if ico is not None and _validate_positions(ico):
        starts.append(ico)
        
    fcc = _generate_fcc_chunk(n)
    if len(fcc) == n and _validate_positions(fcc):
        starts.append(fcc)
        
    bcc = _generate_bcc_chunk(n)
    if len(bcc) == n and _validate_positions(bcc):
        starts.append(bcc)
        
    # 2. Random Compact Starts
    num_random = 5
    for i in range(num_random):
        rcp = _generate_random_compact(n, rng)
        if len(rcp) == n and _validate_positions(rcp):
            starts.append(rcp)
            
    if not starts:
        pts = np.zeros((n, 3))
        pts[:, 0] = np.arange(n) * 0.8
        starts.append(pts)

    def _relax_one(start):
        try:
            energy, positions = relax(start, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        
        for future in as_completed(futures):
            if _get_time_left() < 5.0:
                break
                
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                pass

    # If we still have time, try more random starts
    attempts = 0
    while _get_time_left() > 10.0 and attempts < 10:
        attempts += 1
        try:
            rcp = _generate_random_compact(n, rng)
            if not _validate_positions(rcp):
                continue
            energy, positions = _relax_one(rcp)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
