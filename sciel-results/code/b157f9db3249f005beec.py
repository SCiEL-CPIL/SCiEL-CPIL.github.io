import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def penalty(points):
    # Penalty for points outside the triangle using barycentric coordinates
    # P = u*V0 + v*V1 + w*V2
    # x = v + 0.5w
    # y = sqrt(3)/2 w
    # w = 2y/sqrt(3)
    # v = x - 0.5w
    # u = 1 - v - w
    pen = 0.0
    for p in points:
        x, y = p
        w = 2 * y / np.sqrt(3)
        v = x - 0.5 * w
        u = 1 - v - w
        if u < 0: pen += u*u
        if v < 0: pen += v*v
        if w < 0: pen += w*w
    return pen

def objective(params):
    points = params.reshape(N, 2)
    min_area = min_triangle_area(points)
    pen = penalty(points)
    # We want to maximize min_area, so minimize -min_area
    # Add penalty to keep points inside
    return -min_area + 1000.0 * pen

def generate_hierarchical_init():
    """
    Generate an initial configuration by hierarchically subdividing the triangle.
    This ensures points are structurally balanced and spread out.
    """
    # Recursive subdivision:
    # Level 0: 1 triangle
    # Level 1: 4 triangles
    # Level 2: 16 triangles
    # We want 11 points. 
    # Strategy: 
    # 1. Generate centroids of the 4 Level-1 triangles.
    # 2. Generate centroids of the 16 Level-2 triangles.
    # 3. Select 11 points from the union or a specific mix to ensure good coverage.
    # A better heuristic for N=11:
    # Use the 3 vertices? No, points on boundary can be suboptimal for min area if interior points are too close.
    # Let's use the centroids of a 4x4 grid in barycentric coordinates mapped to cartesian, 
    # but filtered to be inside and distinct.
    # Actually, let's stick to the prompt's "hierarchical subdivision" idea.
    # Subdivide the main triangle into 4. Then subdivide each of those?
    # Let's create a set of candidate points from multiple levels of subdivision.
    
    candidates = []
    
    # Level 0: Centroid
    c0 = (V0 + V1 + V2) / 3.0
    candidates.append(c0)
    
    # Level 1: Centroids of the 4 sub-triangles
    m01 = (V0 + V1) / 2.0
    m12 = (V1 + V2) / 2.0
    m20 = (V2 + V0) / 2.0
    
    c1_0 = (V0 + m01 + m20) / 3.0
    c1_1 = (V1 + m12 + m01) / 3.0
    c1_2 = (V2 + m20 + m12) / 3.0
    c1_3 = (m01 + m12 + m20) / 3.0 # This is the centroid of the medial triangle, same as c0
    
    candidates.extend([c1_0, c1_1, c1_2])
    
    # Level 2: Subdivide the 3 corner triangles of Level 1
    # Triangle 0: V0, m01, m20
    # Midpoints of its edges:
    # e00 = (V0+m01)/2, e01 = (m01+m20)/2, e02 = (m20+V0)/2
    # Centroids of its 4 sub-triangles:
    def get_sub_centroids(v_a, v_b, v_c):
        m_ab = (v_a + v_b) / 2.0
        m_bc = (v_b + v_c) / 2.0
        m_ca = (v_c + v_a) / 2.0
        return [
            (v_a + m_ab + m_ca) / 3.0,
            (v_b + m_bc + m_ab) / 3.0,
            (v_c + m_ca + m_bc) / 3.0,
            (m_ab + m_bc + m_ca) / 3.0
        ]
    
    # Subdivide the 3 corner triangles from Level 1
    # T0: V0, m01, m20
    subs0 = get_sub_centroids(V0, m01, m20)
    candidates.extend(subs0)
    
    # T1: V1, m12, m01
    subs1 = get_sub_centroids(V1, m12, m01)
    candidates.extend(subs1)
    
    # T2: V2, m20, m12
    subs2 = get_sub_centroids(V2, m20, m12)
    candidates.extend(subs2)
    
    # Now we have 1 + 3 + 4 + 4 + 4 = 16 candidates.
    # We need to select 11.
    # To maximize spread, we should pick the ones that are most "central" to their local region 
    # or just pick a diverse subset.
    # A simple strategy: 
    # 1. Always include the main centroid (c0).
    # 2. Include the 3 corner centroids (c1_0, c1_1, c1_2).
    # 3. From the 12 level-2 centroids, pick 7 more that are farthest from existing points.
    
    selected = [c0, c1_0, c1_1, c1_2]
    remaining = subs0 + subs1 + subs2
    
    # Greedy farthest-point sampling
    for _ in range(N - len(selected)):
        best_idx = -1
        max_min_dist = -1
        for i, p in enumerate(remaining):
            min_d = min(np.linalg.norm(p - s) for s in selected)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_idx = i
        if best_idx != -1:
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
            
    return np.array(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    best_score = -float('inf')
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Hierarchical subdivision init (Primary)
    hier_init = generate_hierarchical_init()
    inits.append(hier_init.flatten())
    
    # 2. Random initializations for robustness
    rng = np.random.default_rng(42)
    for _ in range(10):
        r1 = rng.random(N)
        r2 = rng.random(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r2[mask]
        r2[mask] = 1 - r1[mask]
        b0 = 1 - r1 - r2
        b1 = r1
        b2 = r2
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i] = b0[i]*V0 + b1[i]*V1 + b2[i]*V2
        inits.append(pts.flatten())
        
    # 3. Symmetric hexagonal-ish init
    center = np.array([0.5, np.sqrt(3)/6])
    sym_pts = [center]
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        sym_pts.append(np.array([x, y]))
    # Add 4 more near corners/edges to make 11
    # Corners: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Midpoints: (0.5,0), (0.75, sqrt(3)/4), (0.25, sqrt(3)/4)
    # Let's just add 4 random points inside to complete the set for this init
    for _ in range(4):
        while True:
            p = rng.uniform(0, 1, 2)
            if point_in_triangle(p):
                sym_pts.append(p)
                break
    inits.append(np.array(sym_pts).flatten())

    # Run optimization for each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init
        # Nelder-Mead
        try:
            # More iterations for the first (best) init
            maxiter = 5000 if i == 0 else 2000
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7})
            
            # Evaluate the result
            pts = res.x.reshape(N, 2)
            
            # Check validity: all points inside
            valid = all(point_in_triangle(p) for p in pts)
            
            if valid:
                score = min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    
            # Also check if the initial point was better (in case optimization degraded it)
            init_pts = init.reshape(N, 2)
            if all(point_in_triangle(p) for p in init_pts):
                init_score = min_triangle_area(init_pts)
                if init_score > best_score:
                    best_score = init_score
                    best_points = init_pts.copy()
                    
        except Exception:
            pass

    # If no valid points found (should not happen with hierarchical init), return it anyway
    if best_points is None:
        best_points = generate_hierarchical_init()
        
    return best_points

# EVOLVE-BLOCK-END
