import numpy as np
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)

def construct():
    # Triangle vertices
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, S3 / 2])
    
    # Function to compute min triangle area
    def min_triangle_area(pts):
        min_area = np.inf
        # Use itertools.combinations for clarity and speed
        for i, j, k in itertools.combinations(range(N), 3):
            a = pts[i]
            b = pts[j]
            c = pts[k]
            # Area = 0.5 * |cross(b-a, c-a)|
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area = area
        return min_area

    def objective(x):
        pts = x.reshape(-1, 2)
        return -min_triangle_area(pts)

    # Constraints for COBYLA
    # g(x) >= 0
    def c1(x):
        # y >= 0
        pts = x.reshape(-1, 2)
        return np.min(pts[:, 1])
        
    def c2(x):
        # y <= sqrt(3) x  => sqrt(3) x - y >= 0
        pts = x.reshape(-1, 2)
        return np.min(S3 * pts[:, 0] - pts[:, 1])
        
    def c3(x):
        # y <= sqrt(3) (1 - x) => sqrt(3) (1 - x) - y >= 0
        pts = x.reshape(-1, 2)
        return np.min(S3 * (1 - pts[:, 0]) - pts[:, 1])

    constraints = [
        {'type': 'ineq', 'fun': c1},
        {'type': 'ineq', 'fun': c2},
        {'type': 'ineq', 'fun': c3}
    ]
    
    # Multi-start strategy
    best_val = -np.inf
    best_pts = None
    
    # Generate diverse initial points
    starts = []
    
    # 1. Random starts
    for _ in range(20):
        # Generate random points in the triangle
        # Use barycentric coordinates
        u = np.random.rand(N)
        v = np.random.rand(N)
        # Normalize to sum to 1
        u = u / u.sum(axis=1, keepdims=True)
        v = v / v.sum(axis=1, keepdims=True)
        # Actually, standard random in triangle:
        # r1, r2 ~ U(0,1)
        # if r1+r2 > 1, r1=1-r1, r2=1-r2
        # p = r1*A + r2*B + (1-r1-r2)*C
        pts_rand = np.zeros((N, 2))
        for i in range(N):
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            pts_rand[i] = r1 * v1 + r2 * v2 + (1 - r1 - r2) * v0
        starts.append(pts_rand.flatten())
        
    # 2. Symmetric-ish starts (Mirror across x=0.5)
    # 5 pairs + 1 center
    # Center
    center = np.array([0.5, S3/6])
    # Pairs
    for _ in range(5):
        # Random point in left half
        r1 = np.random.rand() * 0.5
        r2 = np.random.rand() * S3 * r1 # y <= sqrt(3)*x
        # Ensure inside
        p_left = np.array([r1, r2])
        p_right = np.array([1 - r1, r2])
        # Check if p_left is inside
        # y >= 0 (ok)
        # y <= sqrt(3)*x (ok by construction)
        # y <= sqrt(3)*(1-x) => r2 <= sqrt(3)*(1-r1)
        if r2 > S3 * (1 - r1):
            r2 = S3 * (1 - r1) * 0.9
            p_left = np.array([r1, r2])
            p_right = np.array([1 - r1, r2])
        
        pts_sym = np.vstack([center, p_left, p_right])
        # We need 11 points. 1 center + 5 pairs = 11.
        # Let's construct 5 pairs properly.
        pairs = []
        for k in range(5):
             r1 = np.random.rand() * 0.45 + 0.05 # x in [0.05, 0.5]
             max_y = min(S3 * r1, S3 * (1 - r1))
             r2 = np.random.rand() * max_y
             p_left = np.array([r1, r2])
             p_right = np.array([1 - r1, r2])
             pairs.append(p_left)
             pairs.append(p_right)
        pts_sym = np.vstack([center] + pairs)
        starts.append(pts_sym.flatten())

    # 3. Grid-like start
    # Place points on a coarse grid inside
    pts_grid = []
    # Simple heuristic: distribute points
    # This is a bit ad-hoc, but provides a different topology
    # 3 rows
    # Row 1 (bottom): 4 points
    # Row 2 (mid): 3 points
    # Row 3 (top): 2 points
    # Row 4 (apex): 1 point? Total 10. Need 11.
    # Let's just use the random starts for now, they are robust.
    
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='COBYLA', constraints=constraints,
                           options={'maxiter': 500, 'rhobeg': 0.1})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_pts = res.x.reshape(-1, 2)
        except:
            pass
            
    # If no result, fallback to a simple grid
    if best_pts is None:
         # Fallback: 11 points on a grid
         best_pts = np.array([
             [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.7, 0.05], [0.9, 0.05],
             [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2],
             [0.35, 0.4], [0.65, 0.4]
         ])
         # Project to inside
         for i in range(N):
             px, py = best_pts[i]
             if py < 0: py = 0.01
             if py > S3 * px:
                 # Project
                 # Line y = sqrt(3)x. Normal (sqrt(3), -1).
                 # d = (sqrt(3)*px - py) / (3+1) = ...
                 # Simpler: scale towards center
                 cx, cy = 0.5, S3/6
                 t = 0.9
                 px = t * px + (1-t) * cx
                 py = t * py + (1-t) * cy
             if py > S3 * (1 - px):
                 cx, cy = 0.5, S3/6
                 t = 0.9
                 px = t * px + (1-t) * cx
                 py = t * py + (1-t) * cy
             best_pts[i] = [px, py]

    return best_pts
