import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_fcc_cluster(n, spacing=1.0):
    """Generate n points from an FCC lattice closest to the origin."""
    # Generate a large enough grid of FCC points
    # FCC points: (i, j, k) where i,j,k are integers, and i+j+k is even? No.
    # FCC basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Let's just generate a cubic grid of points that are FCC compatible.
    # A simpler way: Generate points in a sphere of radius R.
    # Estimate R: volume ~ n * spacing^3. R ~ (3n/4pi)^(1/3) * spacing * some factor.
    
    # Let's use a grid generation.
    max_coord = int(np.ceil((n ** (1/3)) * 2)) + 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Check if (i, j, k) is an FCC point
                # FCC condition: i, j, k all integers, and i+j+k is even.
                # Wait, standard FCC lattice points can be described as:
                # (i, j, k) where i, j, k are integers and i+j+k is even.
                # Let's verify.
                # (0,0,0) sum 0 (even) - OK
                # (1,1,0) sum 2 (even) - OK. Distance sqrt(2).
                # (1,0,0) sum 1 (odd) - Not FCC.
                # Nearest neighbors in FCC are at distance sqrt(2) * a/2?
                # If lattice constant is a, neighbors are at (a/2, a/2, 0) etc.
                # So if we scale by 2, neighbors are at (1,1,0).
                # Let's use a=1.0. Then FCC points are at (i, j, k) with i+j+k even, scaled by 0.5?
                # No, let's just generate the standard FCC points.
                # Basis vectors: a1=(0.5, 0.5, 0), a2=(0.5, 0, 0.5), a3=(0, 0.5, 0.5).
                # Any point is u*a1 + v*a2 + w*a3.
                # x = 0.5(u+v), y = 0.5(v+w), z = 0.5(w+u).
                # 2x = u+v, 2y = v+w, 2z = w+u.
                # u = x+y-z, v = x+z-y, w = y+z-x.
                # For u,v,w to be integers, x,y,z must be integers or half-integers such that sums are integers.
                # Basically, points where 2x, 2y, 2z are integers and 2x+2y+2z is even?
                # Let's just iterate over integer grid for u,v,w and calculate positions.
                pass

    # Simpler approach for FCC:
    # Generate points (i, j, k) where i, j, k are integers.
    # Include point if (i+j+k) % 2 == 0.
    # Scale by 0.5 * lattice_constant.
    # Let's try this.
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([0.5 * i, 0.5 * j, 0.5 * k])
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _generate_hcp_cluster(n, spacing=1.0):
    """Generate n points from an HCP lattice closest to the origin."""
    # HCP structure:
    # Layer 0: Hexagonal grid.
    # Layer 1: Hexagonal grid shifted, z offset.
    # a = 1.0, c/a = sqrt(8/3) ~ 1.633
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    
    pts = []
    max_coord = int(np.ceil((n ** (1/3)) * 2)) + 5
    
    # Generate hexagonal grid points
    # Hex grid basis:
    # b1 = (1, 0, 0)
    # b2 = (0.5, sqrt(3)/2, 0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            # x = i + 0.5*j
            # y = sqrt(3)/2 * j
            x = i + 0.5 * j
            y = np.sqrt(3.0)/2.0 * j
            # Layer 0
            pts.append([x, y, 0.0])
            # Layer 1 (shifted by 1/3, 2/3 in hex coords? Or just shifted in x,y)
            # In HCP, the next layer sits in the hollows.
            # Hollows are at (1/3, 2/3) and (2/3, 1/3) in fractional hex coords.
            # Let's shift by (1/3 * a, 2/3 * sqrt(3)/2 * a, c/2)
            # Actually, simpler: shift by (a/3, sqrt(3)a/6, c/2)
            x_shift = x + a/3.0
            y_shift = y + np.sqrt(3.0)*a/6.0
            pts.append([x_shift, y_shift, c/2.0])
            
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def construct(instance):
    """
    Construct a low-energy cluster configuration.
    """
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0  # Total budget, but we have 6 instances. 
    # The prompt says "all calls together within 300s".
    # So we don't know how much time is left. 
    # However, usually these tasks are evaluated sequentially or in parallel.
    # If sequential, we have ~50s per call. If parallel, 300s total.
    # To be safe, let's assume we have a global budget. 
    # But `construct` is called per instance.
    # Let's just be efficient. 50s is plenty.
    
    # Generate candidates
    candidates = []
    
    # FCC
    try:
        fcc_pts = _generate_fcc_cluster(n)
        candidates.append(fcc_pts)
    except Exception as e:
        pass
        
    # HCP
    try:
        hcp_pts = _generate_hcp_cluster(n)
        candidates.append(hcp_pts)
    except Exception as e:
        pass
        
    # Maybe a "randomized" FCC to break symmetry?
    # For now, just FCC and HCP.
    
    best_energy = np.inf
    best_positions = None
    
    for pos in candidates:
        if time.time() - start_time > 45.0: # Safety margin
            break
        try:
            energy, relaxed_pos = relax(pos, "sutton_chen", kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            continue
            
    if best_positions is None:
        # Fallback to random if everything failed
        rng = np.random.default_rng(n)
        radius = n ** (1/3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        energy, relaxed_pos = relax(np.array(points), "sutton_chen", kind)
        best_positions = relaxed_pos
        
    return best_positions
