import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time
from scipy.spatial.distance import pdist

def _generate_fcc_like(n):
    """
    Generate an FCC-like cluster of size n by selecting the closest n atoms 
    from an infinite FCC lattice to the origin.
    """
    # FCC lattice vectors
    # a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
    # Or simpler: generate points in a cubic box and filter for FCC positions.
    # To ensure we get exactly n atoms compactly, we generate a large set and pick the n closest to center.
    
    max_coord = int(np.ceil(n ** (1/3))) + 2
    coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC points are at (i, j, k) where i+j+k is even? No.
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = i + dx
                    y = j + dy
                    z = k + dz
                    coords.append([x, y, z])
    
    coords = np.array(coords)
    # Center the cluster
    coords -= coords.mean(axis=0)
    
    # Sort by distance to origin
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    
    # Select n closest atoms
    selected = coords[indices[:n]]
    
    # Scale to typical interatomic distance ~1.0
    # The nearest neighbor distance in this lattice is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ~ 0.707
    # We want nearest neighbor ~ 1.0, so scale by 1/sqrt(0.5) = sqrt(2)
    # However, the `relax` function handles scaling/optimization. 
    # But good initial spacing helps. Let's scale so min dist is ~0.8-1.0.
    if n > 1:
        min_d = np.min(pdist(selected))
        if min_d > 1e-6:
            scale_factor = 1.0 / min_d
            selected = selected * scale_factor
            
    return selected

def _generate_hcp_like(n):
    """
    Generate an HCP-like cluster of size n.
    """
    a = 1.0 # lattice constant
    c = np.sqrt(8/3) * a
    points = []
    
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # A layer: z = k*c
                # HCP structure: 
                # Layer 0 (A): (0,0,0)
                # Layer 1 (B): (1/3, 1/sqrt(3), 1/3) * c? 
                # Let's use standard hexagonal coords.
                # a1 = (a, 0, 0), a2 = (a/2, a*sqrt(3)/2, 0)
                # Layer A: r = i*a1 + j*a2
                # Layer B: r = i*a1 + j*a2 + (a/2, a*sqrt(3)/6, c/2) ?? 
                # Standard HCP: 
                # Atoms at (0,0,0) and (1/3, 2/3, 1/2) in fractional coords of hex prism.
                
                # Simpler: Generate hexagonal grid in xy plane.
                x_a = a * (i + 0.5 * (j % 2))
                y_a = a * np.sqrt(3)/2 * j
                z_a = c * k
                points.append([x_a, y_a, z_a])
                
                # B layer offset
                x_b = a * (i + 0.5 * (j % 2)) + a/3
                y_b = a * np.sqrt(3)/2 * j - a * np.sqrt(3)/6
                z_b = c * k + c/3 # Wait, HCP is ABAB. 
                # In HCP, the second layer is at c/2 height? No, c is the height of 2 layers.
                # So layer 0 at z=0, layer 1 at z=c/2.
                # Let's redefine c as the distance between adjacent layers.
                # If c_lattice is the unit cell height, layer spacing is c_lattice/2.
                # Let's use layer_spacing = np.sqrt(2/3) * a.
                layer_spacing = np.sqrt(2/3) * a
                z_b = layer_spacing * k + layer_spacing # This is getting messy.
                
                # Let's stick to a simpler "stacked disks" approach for HCP-like.
                # Or just use the FCC generator and it's close enough? 
                # No, HCP is distinct.
                
                # Let's try a different construction:
                # Generate hexagonal close packed layers.
                pass
                
    # The above loop is inefficient and buggy. Let's use a simpler radial growth for HCP.
    # Actually, for small clusters, FCC and HCP are often confused by optimizers.
    # Let's just return the FCC generator result for HCP kind as well, 
    # or a slightly perturbed FCC, because the energy landscape is similar.
    # BUT, the prompt asks for HCP-like.
    
    # Let's implement a proper HCP generator.
    points = []
    a = 1.0
    c_layer = np.sqrt(2/3) * a # distance between layers
    
    # Generate layers
    max_r = int(np.ceil(n ** (1/3))) + 2
    for k in range(-max_r, max_r + 1):
        z = k * c_layer
        # Determine if this is A or B layer
        if k % 2 == 0:
            offset_x = 0
            offset_y = 0
        else:
            offset_x = a/3
            offset_y = a * np.sqrt(3) / 6
            
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal grid
                x = a * (i + 0.5 * (j % 2)) + offset_x
                y = a * np.sqrt(3)/2 * j + offset_y
                points.append([x, y, z])
                
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    selected = points[indices[:n]]
    
    if n > 1:
        min_d = np.min(pdist(selected))
        if min_d > 1e-6:
            scale_factor = 1.0 / min_d
            selected = selected * scale_factor
            
    return selected

def _generate_ico_like(n):
    """
    Generate an Icosahedral-like cluster (Mackay icosahedra).
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    r1 = 1.0
    for v in ico_verts:
        points.append(v * r1)
        
    if n <= 13:
        return np.array(points[:n])
    
    # Shell 2: 30 points (midpoints of edges) + 20 points (face centers) = 50?
    # Mackay icosahedra:
    # Shell 1: 12
    # Shell 2: 30 (edge centers)
    # Total 43.
    
    # Generate edge centers
    # Icosahedron has 30 edges.
    # We can find edges by checking distance between vertices.
    # Distance between connected vertices is 2.
    # Norm of v is 1.
    # d^2 = 2 - 2 dot(v1, v2).
    # For connected, dot = 1/phi.
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dot = np.dot(ico_verts[i], ico_verts[j])
            if abs(dot - 1/phi) < 1e-6:
                edges.append((i, j))
                
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        # Normalize to radius r2
        r2 = np.linalg.norm(mid) * (1.0 + 0.5) # Approximate radius for shell 2
        # Actually, in Mackay icosahedra, the radial distance is proportional to shell index.
        # Shell 1 radius ~ 1.
        # Shell 2 radius ~ sqrt(1 + 1/phi^2) ??
        # Let's just scale the midpoint to have same norm as shell 1 vertices? No, they are further out.
        # Let's use r2 = 1.25 * r1 as a heuristic.
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 0:
            mid_scaled = mid / mid_norm * (r1 * 1.25)
            points.append(mid_scaled)
            
    if n <= 43:
        return np.array(points[:n])
        
    # Shell 3: 92 points? 
    # This is getting complicated. For n > 43, we can just add random points on the surface 
    # or use a spherical shell.
    
    # For simplicity and robustness, if n > 43, we fall back to a "spherical shell" 
    # that is perturbed icosahedral.
    
    current_points = np.array(points)
    target_count = n
    
    if len(current_points) < target_count:
        rng = np.random.default_rng(42)
        # Add points on a spherical shell of radius r3
        r3 = r1 * 1.5
        while len(current_points) < target_count:
            dir_vec = rng.randn(3)
            dir_vec /= np.linalg.norm(dir_vec)
            new_p = dir_vec * r3
            # Check if it's too close to existing points
            dists = np.linalg.norm(current_points - new_p, axis=1)
            if np.min(dists) > 0.5 * r3:
                current_points = np.vstack([current_points, new_p])
            elif rng.rand() > 0.9:
                current_points = np.vstack([current_points, new_p])
                
    current_points = current_points[:target_count]
    
    # Scale to min distance ~ 1.0
    if n > 1:
        min_d = np.min(pdist(current_points))
        if min_d > 1e-6:
            scale_factor = 1.0 / min_d
            current_points = current_points * scale_factor
            
    return current_points

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    candidates = []
    
    # 1. FCC-like (High quality)
    try:
        fcc_pts = _generate_fcc_like(n)
        candidates.append(fcc_pts)
    except Exception:
        fcc_pts = None
        
    # 2. HCP-like (High quality)
    try:
        hcp_pts = _generate_hcp_like(n)
        candidates.append(hcp_pts)
    except Exception:
        hcp_pts = None
        
    # 3. Icosahedral-like (High quality)
    try:
        ico_pts = _generate_ico_like(n)
        candidates.append(ico_pts)
    except Exception:
        ico_pts = None

    # 4. Jittered versions of the above to escape local minima
    # We only jitter if the base candidate was generated successfully
    if fcc_pts is not None:
        for i in range(3):
            rng = np.random.default_rng(100 + i)
            jittered = fcc_pts + rng.normal(0, 0.1, (n, 3))
            candidates.append(jittered)
            
    if hcp_pts is not None:
        for i in range(3):
            rng = np.random.default_rng(200 + i)
            jittered = hcp_pts + rng.normal(0, 0.1, (n, 3))
            candidates.append(jittered)
            
    if ico_pts is not None:
        for i in range(3):
            rng = np.random.default_rng(300 + i)
            jittered = ico_pts + rng.normal(0, 0.1, (n, 3))
            candidates.append(jittered)

    # 5. A few random compact clusters as a safety net
    for i in range(5):
        try:
            rng = np.random.default_rng(900 + i)
            # Generate random points in a sphere
            pts = rng.normal(0, 1, (n, 3))
            pts /= np.max(np.linalg.norm(pts, axis=1)) * 0.5
            candidates.append(pts)
        except Exception:
            pass

    best_energy = float('inf')
    best_positions = None
    
    # Relax all candidates
    for pts in candidates:
        if not np.all(np.isfinite(pts)):
            continue
        # Check minimum distance to avoid singularities
        try:
            min_d = np.min(pdist(pts))
            if min_d < 0.35:
                continue
        except Exception:
            continue
            
        try:
            energy, relaxed_pts = relax(pts, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pts
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: random
        rng = np.random.default_rng(0)
        best_positions = rng.normal(0, 1, (n, 3))
        
    return best_positions
