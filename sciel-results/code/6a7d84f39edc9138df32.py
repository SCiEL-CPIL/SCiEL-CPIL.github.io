import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _generate_mackay_icosahedron():
    """
    Generate coordinates for a 55-atom Mackay icosahedron.
    Returns an array of shape (55, 3).
    """
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron with edge length 1
    # Standard vertices:
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    # These have edge length 2. So we divide by 2 to get edge length 1.
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) / 2.0
    
    # Center atom
    center = np.array([0.0, 0.0, 0.0])
    
    # 1st Shell: 12 vertices
    shell1 = verts
    
    # 2nd Shell: 30 midpoints of edges of the icosahedron
    # We need to find all unique edges.
    # An edge connects two vertices if their distance is 1.
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(shell1[i] - shell1[j]) < 1.1: # Tolerance
                edges.append((i, j))
    
    shell2 = []
    for i, j in edges:
        mid = (shell1[i] + shell1[j]) / 2.0
        shell2.append(mid)
    shell2 = np.array(shell2)
    
    # 3rd Shell: 12 centers of faces of the icosahedron
    # We need to find all unique faces (triangles).
    # A face is defined by 3 vertices that are all connected.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(shell1[i] - shell1[j]) > 1.1:
                continue
            for k in range(j+1, 12):
                if np.linalg.norm(shell1[i] - shell1[k]) < 1.1 and np.linalg.norm(shell1[j] - shell1[k]) < 1.1:
                    # Check if i,j,k form a face (they should, in a convex polyhedron)
                    centroid = (shell1[i] + shell1[j] + shell1[k]) / 3.0
                    # Avoid duplicates
                    is_dup = False
                    for f in faces:
                        if np.allclose(centroid, f, atol=1e-5):
                            is_dup = True
                            break
                    if not is_dup:
                        faces.append(centroid)
                        
    shell3 = np.array(faces)
    
    # Combine
    coords = np.vstack([center.reshape(1,3), shell1, shell2, shell3])
    return coords

def construct(instance):
    """
    Find the lowest-energy structure for the given Dzugutov cluster instance.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Global deadline management
    start_time = time.time()
    total_time_limit = 300.0
    # We have 6 instances. Let's aim to finish each in ~45s to be safe.
    # But construct is called per instance. We don't know the order or if time is shared.
    # The prompt says "all calls together within 300 s".
    # So we should track global time if possible, but we can't easily share state between calls 
    # unless we use a global variable. Let's use a global variable to track elapsed time.
    
    if not hasattr(construct, "_start_time"):
        construct._start_time = time.time()
    
    elapsed_global = time.time() - construct._start_time
    # Assume 6 calls. If this is the k-th call, we have (300 - elapsed_global) left.
    # To be safe, let's allocate a fixed budget per call, e.g., 45 seconds.
    # If we exceed 45s, we return the best we have.
    call_budget = 45.0
    
    # Generate base Mackay Icosahedron (55 atoms)
    mackay_55 = _generate_mackay_icosahedron()
    
    # Scale the Mackay icosahedron to a reasonable size.
    # The potential minimum is around r ~ 1.9.
    # In the Mackay icosahedron (edge 1), nearest neighbors are distance 1.
    # Let's scale by factor 1.9 so that NN distance is ~1.9.
    scale = 1.9
    mackay_55_scaled = mackay_55 * scale
    
    candidates = []
    
    # Strategy: 
    # 1. If n == 55, use the full Mackay.
    # 2. If n < 55, remove atoms from the outer shells.
    # 3. If n > 55, add atoms.
    
    if n == 55:
        candidates.append(mackay_55_scaled.copy())
    elif n < 55:
        # Remove atoms from the outer shells (shell3, then shell2, then shell1)
        # Shell 1: 12 atoms (indices 1-12)
        # Shell 2: 30 atoms (indices 13-42)
        # Shell 3: 12 atoms (indices 43-54)
        # Center: 1 atom (index 0)
        
        # We want to keep the core.
        # Let's try removing from Shell 3 first, then Shell 2, then Shell 1.
        # We need to remove (55 - n) atoms.
        num_to_remove = 55 - n
        
        # Create a list of indices to keep
        # Indices: 0 (center), 1-12 (shell1), 13-42 (shell2), 43-54 (shell3)
        
        # We will generate a few variations of removal to see which relaxes best.
        # Variation 1: Remove random outer atoms.
        # Variation 2: Remove symmetric sets.
        
        # For simplicity, let's just remove the last (55-n) atoms from the list 
        # if we order them by distance from center (outermost first).
        # But Shell 3 atoms are furthest, then Shell 2, then Shell 1.
        # So we can just take the first n atoms if we sort by distance?
        # Let's sort indices by distance from center, descending.
        dists = np.linalg.norm(mackay_55_scaled, axis=1)
        sorted_indices = np.argsort(dists)[::-1] # Descending
        
        # Keep the first n atoms in this sorted order? No, that keeps the inner ones.
        # We want to keep the inner ones. So we keep the ones with smallest distance.
        # So we take the first n indices from argsort(dists) (ascending).
        keep_indices = np.argsort(dists)[:n]
        candidates.append(mackay_55_scaled[keep_indices])
        
        # Also try a random removal
        rng = np.random.default_rng(n)
        for _ in range(2):
            remove_indices = rng.choice(55, size=num_to_remove, replace=False)
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[remove_indices] = False
            candidates.append(mackay_55_scaled[keep_mask])
            
    elif n > 55:
        # Add atoms
        num_to_add = n - 55
        base = mackay_55_scaled.copy()
        
        # Add atoms in a shell outside the 55-atom structure.
        # Radius of 55-atom structure: max distance is ~ 1.9 * (distance of shell3 from center)
        # Let's just add random points in a sphere of radius R_max + 2.0
        max_r = np.max(np.linalg.norm(base, axis=1))
        
        rng = np.random.default_rng(n)
        for _ in range(3):
            new_atoms = []
            for _ in range(num_to_add):
                # Random point in shell
                while True:
                    p = rng.uniform(-1, 1, 3)
                    r = np.linalg.norm(p)
                    if 0.8 < r < 1.2:
                        p = p / r * (max_r + 1.5)
                        # Check distance to existing atoms
                        if np.all(np.linalg.norm(base - p, axis=1) > 1.0):
                            new_atoms.append(p)
                            break
            new_structure = np.vstack([base, np.array(new_atoms)])
            candidates.append(new_structure)
            
    # Fallback: Random compact cluster
    rng = np.random.default_rng(n + 100)
    for _ in range(2):
        points = []
        radius = 2.0 * n ** (1.0/3.0)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 1.2 for q in points):
                points.append(p)
        candidates.append(np.array(points))
        
    best_energy = float('inf')
    best_positions = None
    
    for i, cand in enumerate(candidates):
        # Check time
        current_time = time.time()
        if current_time - start_time > call_budget:
            break
            
        try:
            energy, positions = relax(cand, potential, instance.get("rho", instance.get("kind")))
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            pass
            
    if best_positions is None:
        # Return the first candidate as a last resort
        best_positions = candidates[0]
        
    return best_positions
