import numpy as np
from scipy.spatial import Delaunay, cKDTree

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base inverse distance
    d_safe = np.where(dm > 0, dm, 1e-9)
    base = 1.0 / (d_safe + 1e-9)
    np.fill_diagonal(base, 1e-9)
    
    # Build Delaunay triangulation
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a = int(simplices[i]) if False else int(simplex[i])
                b = int(simplex[(i + 1) % 3])
                if a != b:
                    delaunay_edges.add((min(a, b), max(a, b)))
    except Exception:
        delaunay_edges = set()
    
    # Add nearest-neighbor edges (k=6) for connectivity
    try:
        tree = cKDTree(coords)
        _, nn_indices = tree.query(coords, k=7)  # 7 including self
        for i in range(n):
            for j in nn_indices[i, 1:]:
                j = int(j)
                if j != i:
                    delaunay_edges.add((min(i, j), max(i, j)))
    except Exception:
        pass
    
    # Compute Clarke-Wright savings ratio
    d0 = dm[0, :]
    d0_rows = d0[:, np.newaxis]
    d0_cols = d0[np.newaxis, :]
    num = d0_rows + d0_cols - dm
    den = d0_rows + d0_cols
    den_safe = np.where(den > 0, den, 1e-9)
    savings_ratio = np.clip(num / den_safe, 0.0, 1.0)
    
    # Build heuristic matrix
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Process Delaunay edges
    for (a, b) in delaunay_edges:
        dist_ab = dm[a, b]
        
        # Base: inverse distance
        w = 1.0 / (dist_ab + 1e-9)
        
        # Clarke-Wright savings bonus for customer-customer edges
        if a > 0 and b > 0:
            sr = savings_ratio[a, b]
            w *= (1.0 + 0.5 * sr)
        
        # Demand compatibility bonus
        if a == 0 or b == 0:
            # Depot edges: slight bonus for lower-demand customers
            other = b if a == 0 else a
            if other > 0:
                w *= (1.0 + 0.1 * (capacity - dem[other]) / capacity)
        else:
            # Customer-customer: bonus if both fit in one route segment
            if dem[a] + dem[b] <= capacity:
                w *= 1.2
            else:
                w *= 0.8
        
        heuristic[a, b] = w
        heuristic[b, a] = w
    
    # For non-Delaunay edges, keep them very low but not zero
    # This enforces the planar structure while allowing some flexibility
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) not in delaunay_edges and (j, i) not in delaunay_edges:
                # Very low weight for non-Delaunay edges
                heuristic[i, j] = 1e-9
    
    # Ensure diagonal is zero
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all entries are positive
    heuristic[heuristic <= 0] = 1e-9
    
    # Normalize so median is around 1.0
    positive_vals = heuristic[heuristic > 1e-8]
    if len(positive_vals) > 0:
        med = np.median(positive_vals)
        if med > 0:
            heuristic = heuristic / med
    
    return heuristic
