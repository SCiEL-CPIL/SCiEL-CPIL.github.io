# EVOLVE-BLOCK-START
import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices using a bipartite radial parameterization.
    Vertices alternate between two radii R1 and R2 on a regular 10-gon angular grid.
    This reduces the search space to one parameter (the ratio R2/R1) while maintaining
    strict convexity and cyclic order. We optimize this ratio to maximize the minimum
    vertex score defined by the task contract.
    """
    n = 20
    # Angular grid: 20 points, step 2*pi/20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def generate_polygon(rho):
        """Generate polygon vertices given ratio rho = R2/R1 (R1=1)."""
        radii = np.ones(n)
        radii[1::2] = rho  # Odd indices get R2
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)

    def normalize(polygon):
        """Subtract centroid and divide by mean distance to centroid."""
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return None, 0.0
        normalized = centered / mean_dist
        return normalized, mean_dist

    def is_valid_convex(polygon):
        """Check if polygon is strictly convex and has no repeated points."""
        n = len(polygon)
        if n < 3:
            return False
        
        # Check for repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(polygon[i] - polygon[j]) < 1e-9:
                    return False
            
        # Check turning direction
        turns = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i+1) % n]
            p2 = polygon[(i+2) % n]
            
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Cross product in 2D
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            
            if abs(cross) < 1e-9:
                return False # Non-convex or collinear
            turns.append(cross)
        
        # All turns must have the same sign
        if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
            return False
            
        return True

    def vertex_score(vertex, polygon):
        """Calculate score for a single vertex."""
        # Calculate distances to all other vertices
        diffs = polygon - vertex
        dists = np.linalg.norm(diffs, axis=1)
        
        # Set distance to self to infinity to ignore
        idx = np.where(np.linalg.norm(diffs) < 1e-9)[0]
        if len(idx) > 0:
            dists[idx] = np.inf
            
        # Sort distances
        sorted_dists = np.sort(dists)
        
        # Check d_0 < 0.05
        if sorted_dists[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(sorted_dists)
        # Handle case where min gap is 0
        min_gap = np.min(gaps)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1}) / 2
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            selected = sorted_dists[0:4]
        elif k == 17: # n-2 = 18, so k can be up to 18? 
            # Wait, sorted_dists has 19 elements (indices 0..18).
            # gaps has 18 elements (indices 0..17).
            # So k is in 0..17.
            selected = sorted_dists[15:19]
        else:
            # d_{k-1} .. d_{k+2}
            start = k - 1
            end = k + 3 # exclusive
            selected = sorted_dists[start:end]
            
        # Calculate raw
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        return score

    def total_score(polygon):
        """Calculate minimum vertex score over all vertices."""
        normalized, _ = normalize(polygon)
        if normalized is None:
            return 0.0
            
        if not is_valid_convex(normalized):
            return 0.0
            
        scores = []
        for i in range(len(normalized)):
            s = vertex_score(normalized[i], normalized)
            scores.append(s)
            
        return np.min(scores)

    # Optimization loop
    # We search for the optimal rho = R2/R1
    # Initial guess from analysis: rho ~ 0.4 or 1.5
    best_rho = 1.0
    best_score = -1.0
    
    # Grid search
    # Range: 0.1 to 5.0
    rho_values = np.linspace(0.1, 5.0, 1000)
    
    for rho in rho_values:
        poly = generate_polygon(rho)
        score = total_score(poly)
        if score > best_score:
            best_score = score
            best_rho = rho
            
    # Refine search around best_rho
    if best_score > 0:
        # Narrow down the grid
        step = (5.0 - 0.1) / 1000
        refined_rhos = np.linspace(best_rho - 5*step, best_rho + 5*step, 500)
        for rho in refined_rhos:
            if rho <= 0: continue
            poly = generate_polygon(rho)
            score = total_score(poly)
            if score > best_score:
                best_score = score
                best_rho = rho
                
        # Even finer
        step2 = (10*step) / 500
        refined_rhos2 = np.linspace(best_rho - 2*step2, best_rho + 2*step2, 200)
        for rho in refined_rhos2:
            if rho <= 0: continue
            poly = generate_polygon(rho)
            score = total_score(poly)
            if score > best_score:
                best_score = score
                best_rho = rho

    return generate_polygon(best_rho)

# EVOLVE-BLOCK-END
