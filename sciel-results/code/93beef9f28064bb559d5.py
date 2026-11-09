import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14():
    """
    Returns an array of shape (14, 3) of finite point coordinates.
    Maximizes (d_min / d_max)^2.
    """
    
    # Define tetrahedron vertices (unit vectors)
    # Vertices of a regular tetrahedron centered at origin
    phi = (1 + np.sqrt(5)) / 2
    # Using standard coordinates scaled to unit length
    # [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1] are not regular tetrahedron vertices on sphere?
    # Yes, they are.
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, -1, -1])
    v3 = np.array([-1, 1, -1])
    v4 = np.array([-1, -1, 1])
    
    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    v4 = v4 / np.linalg.norm(v4)
    
    vertices = np.array([v1, v2, v3, v4])
    
    # Generate 10 edge midpoints (unit vectors)
    midpoints = []
    for i in range(4):
        for j in range(i + 1, 4):
            m = vertices[i] + vertices[j]
            m = m / np.linalg.norm(m)
            midpoints.append(m)
    midpoints = np.array(midpoints)
    
    # The points are:
    # 4 vertices scaled by a (fixed to 1 for scale invariance)
    # 10 midpoints scaled by b
    # We optimize b.
    
    def objective(b):
        # b is the radius for midpoints
        if b < 0:
            return 0.0
        pts = np.vstack([vertices, b * midpoints])
        
        # Calculate pairwise distances
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
            
        ratio = d_min / d_max
        return ratio * ratio
        
    def neg_objective(b):
        return -objective(b)
        
    # Initial guess: b = 1/sqrt(3) (midpoints on inscribed sphere) or b=1 (circumscribed)
    # Let's scan a range to find a good initial guess
    b_vals = np.linspace(0.1, 2.0, 100)
    scores = [objective(b) for b in b_vals]
    best_b = b_vals[np.argmax(scores)]
    
    # Refine using minimize_scalar
    res = minimize_scalar(neg_objective, bounds=(0.1, 2.0), method='bounded', 
                          options={'xatol': 1e-12})
    
    if res.success:
        best_b = res.x
    else:
        # Fallback to grid best
        pass
        
    # Construct final points
    # a = 1.0
    final_points = np.vstack([vertices, best_b * midpoints])
    
    return final_points

if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    # Calculate score for verification
    dists = pdist(pts)
    score = (np.min(dists) / np.max(dists)) ** 2
    print(f"Score: {score}")
