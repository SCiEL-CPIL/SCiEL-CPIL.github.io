import numpy as np
from scipy.optimize import minimize

N = 12

def hexagon_contains_point(cx, cy, s, theta, px, py, margin=0.0):
    """Check if point (px,py) is inside regular hexagon with center (cx,cy), side s, rotation theta (radians)."""
    # The 6 side normals for a regular hexagon with rotation theta
    # Vertices are at angles theta + k*pi/3 for k=0..5, at distance s from center
    # Side normals point outward at angles theta + pi/6 + k*pi/3 for k=0..5
    inradius = s * np.sqrt(3) / 2
    for k in range(6):
        angle = theta + np.pi/6 + k * np.pi/3
        nx = np.cos(angle)
        ny = np.sin(angle)
        dist = (px - cx) * nx + (py - cy) * ny
        if dist > inradius + margin:
            return False
    return True

def min_enclosing_hexagon_side(centers, angles=None):
    """Find minimum side length of regular hexagon containing all unit hexagons at given centers.
    centers: list of (x, y) for N inner hexagons (all with 0 rotation)
    Returns optimal (cx, cy, s, theta)
    """
    if angles is None:
        angles = [0.0] * len(centers)
    
    centers = np.array(centers)
    cx0 = np.mean(centers[:, 0])
    cy0 = np.mean(centers[:, 1])
    
    def objective(params):
        cx, cy, theta = params
        # For each of 6 side directions, compute max projection of any point on the inner hexagons
        # Each inner hexagon at (xi, yi) with 0 rotation: max projection in direction n is dot((xi,yi), n) + 1
        # (circumradius = 1 for unit hexagon)
        # Constraint: max_i [dot((xi,yi), n_k) + 1] - dot((cx,cy), n_k) <= s * sqrt(3)/2
        # So s >= (max_i [dot((xi,yi), n_k) + 1] - dot((cx,cy), n_k)) / (sqrt(3)/2)
        s_min = 0.0
        for k in range(6):
            angle = theta + np.pi/6 + k * np.pi/3
            nx = np.cos(angle)
            ny = np.sin(angle)
            # Max projection of inner hexagon points in this direction
            # For a unit hexagon at (xi, yi) with 0 rotation, the support function in direction (nx, ny)
            # is dot((xi,yi), (nx,ny)) + h_inner(nx, ny)
            # For a unit regular hexagon (side 1, 0 rotation), the support function is:
            # max over its 6 vertices of dot(vertex, (nx,ny))
            # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius 1
            # h_inner = max(cos(a)*nx + sin(a)*ny for a in [0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3])
            # This equals max(|nx|, |ny|, |nx/2 + sqrt(3)*ny/2|, |nx/2 - sqrt(3)*ny/2|, ...) 
            # Actually for a regular hexagon with circumradius 1, h(n) = max over 6 vertex directions
            vertex_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
            h_inner = np.max(np.cos(vertex_angles - np.arctan2(ny, nx)))
            # h_inner = max(cos(va - angle_of_n)) for va in vertex_angles
            # = max over vertices of cos(angle_between)
            # Since circumradius is 1, h_inner = max(cos(theta_v - theta_n))
            
            projections = centers[:, 0] * nx + centers[:, 1] * ny + h_inner
            max_proj = np.max(projections)
            center_proj = cx * nx + cy * ny
            s_k = (max_proj - center_proj) / (np.sqrt(3) / 2)
            s_min = max(s_min, s_k)
        return s_min
    
    # Optimize over cx, cy, theta
    def obj(params):
        return objective(params)
    
    best_s = float('inf')
    best_params = None
    
    # Try multiple initial theta values
    for theta_init in np.linspace(0, np.pi/3, 7):
        res = minimize(obj, [cx0, cy0, theta_init], method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
    
    cx, cy, theta = best_params
    return cx, cy, best_s, theta

def construct():
    # Try various hexagonal cluster arrangements for 12 hexagons
    # Hexagonal grid: row spacing = 3*sqrt(3)/2, col spacing = 3, alternating offset = 1.5
    
    arrangements = []
    
    # Arrangement 1: 3-4-3-2
    # Row 0: 3 hexagons at x = -1.5, 0, 1.5 (relative), y = 0
    # Row 1: 4 hexagons at x = -3, -1, 1, 3, y = 3*sqrt(3)/2
    # Row 2: 3 hexagons at x = -1.5, 0, 1.5, y = 3*sqrt(3)
    # Row 3: 2 hexagons at x = -1, 1, y = 9*sqrt(3)/2
    row_h = 3 * np.sqrt(3) / 2
    arr1 = []
    for i, x in enumerate([-1.5, 0, 1.5]):
        arr1.append([x, 0, 0.0])
    for x in [-3, -1, 1, 3]:
        arr1.append([x, row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr1.append([x, 2*row_h, 0.0])
    for x in [-1, 1]:
        arr1.append([x, 3*row_h, 0.0])
    arrangements.append(arr1)
    
    # Arrangement 2: 2-3-4-3
    arr2 = []
    for x in [-1, 1]:
        arr2.append([x, 0, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr2.append([x, row_h, 0.0])
    for x in [-3, -1, 1, 3]:
        arr2.append([x, 2*row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr2.append([x, 3*row_h, 0.0])
    arrangements.append(arr2)
    
    # Arrangement 3: 4-3-3-2
    arr3 = []
    for x in [-3, -1, 1, 3]:
        arr3.append([x, 0, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr3.append([x, row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr3.append([x, 2*row_h, 0.0])
    for x in [-1, 1]:
        arr3.append([x, 3*row_h, 0.0])
    arrangements.append(arr3)
    
    # Arrangement 4: 2-4-3-3
    arr4 = []
    for x in [-1, 1]:
        arr4.append([x, 0, 0.0])
    for x in [-3, -1, 1, 3]:
        arr4.append([x, row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr4.append([x, 2*row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr4.append([x, 3*row_h, 0.0])
    arrangements.append(arr4)
    
    # Arrangement 5: 3-3-4-2
    arr5 = []
    for x in [-1.5, 0, 1.5]:
        arr5.append([x, 0, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr5.append([x, row_h, 0.0])
    for x in [-3, -1, 1, 3]:
        arr5.append([x, 2*row_h, 0.0])
    for x in [-1, 1]:
        arr5.append([x, 3*row_h, 0.0])
    arrangements.append(arr5)
    
    # Arrangement 6: 2-3-3-4
    arr6 = []
    for x in [-1, 1]:
        arr6.append([x, 0, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr6.append([x, row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr6.append([x, 2*row_h, 0.0])
    for x in [-3, -1, 1, 3]:
        arr6.append([x, 3*row_h, 0.0])
    arrangements.append(arr6)
    
    # Arrangement 7: 3-2-4-3
    arr7 = []
    for x in [-1.5, 0, 1.5]:
        arr7.append([x, 0, 0.0])
    for x in [-1, 1]:
        arr7.append([x, row_h, 0.0])
    for x in [-3, -1, 1, 3]:
        arr7.append([x, 2*row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr7.append([x, 3*row_h, 0.0])
    arrangements.append(arr7)
    
    # Arrangement 8: 4-2-3-3
    arr8 = []
    for x in [-3, -1, 1, 3]:
        arr8.append([x, 0, 0.0])
    for x in [-1, 1]:
        arr8.append([x, row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr8.append([x, 2*row_h, 0.0])
    for x in [-1.5, 0, 1.5]:
        arr8.append([x, 3*row_h, 0.0])
    arrangements.append(arr8)
    
    # Also try a more compact arrangement: center + 11 around
    # Central hexagon at (0,0), then 6 around it at distance 3, then 5 more
    arr9 = [[0, 0, 0.0]]
    for k in range(6):
        angle = k * np.pi / 3
        arr9.append([3 * np.cos(angle), 3 * np.sin(angle), 0.0])
    # Add 5 more in the gaps
    for k in range(5):
        angle = np.pi/6 + k * 2 * np.pi / 5
        arr9.append([3 * np.sqrt(3) * np.cos(angle), 3 * np.sqrt(3) * np.sin(angle), 0.0])
    arrangements.append(arr9)
    
    best_result = None
    best_s = float('inf')
    
    for arr in arrangements:
        centers = [(p[0], p[1]) for p in arr]
        cx, cy, s, theta = min_enclosing_hexagon_side(centers)
        if s < best_s:
            best_s = s
            best_result = (arr, cx, cy, s, theta)
    
    inner, cx, cy, s, theta = best_result
    return inner, (cx, cy), s, np.degrees(theta)

