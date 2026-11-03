import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Initial guess: Hexagonal lattice
    # Spacing d = sqrt(3) for unit hexagons
    d = np.sqrt(3)
    
    # Generate points in a hexagonal lattice pattern
    # We want 12 points.
    # A 3x4 rectangle in lattice coords?
    # Or a hexagonal cluster.
    # Let's try a 3x4 grid in axial coordinates converted to Cartesian.
    # Axial coords (q, r) to Cartesian (x, y) for pointy-top or flat-top?
    # Let's just place them manually in a compact shape.
    
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: 5 points? Total 12.
    
    # Let's use a simple grid for initialization and let optimizer fix it.
    # 3 rows, 4 columns
    pts = []
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * d
            y = (i - 1.0) * d * 1.5 # Offset rows for hex packing
            # Stagger rows
            if i % 2 == 1:
                x += d / 2
            pts.append([x, y])
            
    x0 = np.array(pts).flatten()
    
    def objective(params):
        pts = params.reshape(-1, 2)
        # Calculate required side length for each point
        # s_req = max( x, -x, y/sqrt(3), -y/sqrt(3), x+y/sqrt(3), -x+y/sqrt(3) )
        # Wait, the 6th term was -x + y/sqrt(3). What about the bottom slants?
        # The constraints were:
        # 1. x <= s
        # 2. -x <= s
        # 3. y/sqrt(3) <= s
        # 4. -y/sqrt(3) <= s
        # 5. x + y/sqrt(3) <= s  (Top-Right side? No, Top-Left/Right?)
        # Let's re-verify the 6 linear forms.
        # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # Lines:
        # 1. x = s  -> x <= s
        # 2. x = -s -> -x <= s
        # 3. y = s*sqrt(3)/2 -> y/sqrt(3) <= s
        # 4. y = -s*sqrt(3)/2 -> -y/sqrt(3) <= s
        # 5. Line through (s,0) and (s/2, s*sqrt(3)/2). Eq: y = -sqrt(3)(x-s) -> y = -sqrt(3)x + sqrt(3)s -> y + sqrt(3)x <= sqrt(3)s -> y/sqrt(3) + x <= s
        # 6. Line through (-s,0) and (-s/2, s*sqrt(3)/2). Eq: y = sqrt(3)(x+s) -> y = sqrt(3)x + sqrt(3)s -> y - sqrt(3)x <= sqrt(3)s -> y/sqrt(3) - x <= s
        # 7. Line through (s/2, -s*sqrt(3)/2) and (s,0). Eq: y = sqrt(3)(x-s) -> y = sqrt(3)x - sqrt(3)s -> y - sqrt(3)x >= -sqrt(3)s -> -y/sqrt(3) + x <= s ? 
        #    Check: -y + sqrt(3)x >= -sqrt(3)s -> sqrt(3)x - y >= -sqrt(3)s -> -sqrt(3)x + y <= sqrt(3)s -> -x + y/sqrt(3) <= s.
        #    Wait, I already have -x + y/sqrt(3) from line 6?
        #    Line 6: y - sqrt(3)x <= sqrt(3)s -> y/sqrt(3) - x <= s.
        #    Line 7: y - sqrt(3)x >= -sqrt(3)s -> y/sqrt(3) - x >= -s -> -(y/sqrt(3) - x) <= s -> x - y/sqrt(3) <= s.
        # 8. Line through (-s/2, -s*sqrt(3)/2) and (-s,0). Eq: y = -sqrt(3)(x+s) -> y = -sqrt(3)x - sqrt(3)s -> y + sqrt(3)x >= -sqrt(3)s -> -(y/sqrt(3) + x) <= s -> -x - y/sqrt(3) <= s.
        
        # So the 6 terms are:
        # 1. x
        # 2. -x
        # 3. y/sqrt(3)
        # 4. -y/sqrt(3)
        # 5. x + y/sqrt(3)
        # 6. -x + y/sqrt(3)  (Wait, is this correct? Let's check symmetry)
        #    The set of normals is 6 vectors at 0, 60, 120, 180, 240, 300 deg.
        #    Normals for flat-top hexagon (vertices at 0, 60...):
        #    Side 1 (Right): Normal (1, 0). Dot: x.
        #    Side 2 (Top-Right): Normal (1/2, sqrt(3)/2). Dot: 0.5x + 0.866y.
        #       Condition: Dot <= A = s*sqrt(3)/2.
        #       0.5x + 0.866y <= 0.866s
        #       x/sqrt(3) + y/3 <= s/sqrt(3) ?
        #       Multiply by sqrt(3): x + y/sqrt(3) <= s. Correct.
        #    Side 3 (Top-Left): Normal (-1/2, sqrt(3)/2). Dot: -0.5x + 0.866y.
        #       -0.5x + 0.866y <= 0.866s
        #       Multiply by sqrt(3): -x + y/sqrt(3) <= s. Correct.
        #    Side 4 (Left): Normal (-1, 0). Dot: -x.
        #       -x <= s.
        #    Side 5 (Bottom-Left): Normal (-1/2, -sqrt(3)/2). Dot: -0.5x - 0.866y.
        #       -0.5x - 0.866y <= 0.866s
        #       Multiply by sqrt(3): -x - y/sqrt(3) <= s.
        #    Side 6 (Bottom-Right): Normal (1/2, -sqrt(3)/2). Dot: 0.5x - 0.866y.
        #       0.5x - 0.866y <= 0.866s
        #       Multiply by sqrt(3): x - y/sqrt(3) <= s.
        
        # So the 6 terms to maximize are:
        # 1. x
        # 2. -x
        # 3. y/sqrt(3)
        # 4. -y/sqrt(3)
        # 5. x + y/sqrt(3)
        # 6. -x + y/sqrt(3)
        # 7. x - y/sqrt(3)
        # 8. -x - y/sqrt(3)
        # Actually, terms 1,2 are dominated by 5,6,7,8?
        # If x > 0, x + y/sqrt(3) >= x if y>=0. If y<0, x - y/sqrt(3) >= x.
        # So we just need the max of the 4 slanted terms?
        # Let's check.
        # If y=0, terms are x, -x, x, -x. Max is |x|.
        # If x=0, terms are y/sqrt(3), -y/sqrt(3). Max is |y|/sqrt(3).
        # So yes, max of the 4 slanted terms covers the vertical/horizontal ones?
        # Let's verify.
        # Max( x+y/sqrt(3), x-y/sqrt(3), -x+y/sqrt(3), -x-y/sqrt(3) )
        # = Max( x + |y|/sqrt(3), -x + |y|/sqrt(3) )
        # = |x| + |y|/sqrt(3) ?
        # No.
        # If x=1, y=10.
        # x+y/sqrt(3) = 1+5.77 = 6.77
        # x-y/sqrt(3) = 1-5.77 = -4.77
        # -x+y/sqrt(3) = -1+5.77 = 4.77
        # -x-y/sqrt(3) = -1-5.77 = -6.77
        # Max is 6.77.
        # Formula |x| + |y|/sqrt(3) = 1 + 5.77 = 6.77.
        # It seems the hexagonal norm is $|x| + \frac{|y|}{\sqrt{3}}$?
        # Let's check a point on the boundary.
        # Vertex (s, 0). Norm: s + 0 = s. Correct.
        # Vertex (s/2, s*sqrt(3)/2). Norm: s/2 + (s*sqrt(3)/2)/sqrt(3) = s/2 + s/2 = s. Correct.
        # So the required side $s_{req}$ for a point $(x,y)$ is $|x| + \frac{|y|}{\sqrt{3}}$.
        
        s_reqs = np.abs(pts[:, 0]) + np.abs(pts[:, 1]) / np.sqrt(3)
        return np.max(s_reqs)
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    
    if res.success or res.fun < objective(x0):
        opt_pts = res.x.reshape(-1, 2)
        min_s = res.fun
    else:
        opt_pts = x0.reshape(-1, 2)
        min_s = objective(x0)
        
    # The outer side length S is min_s + 1
    S = min_s + 1.0
    
    # Construct inner hexagons
    # All aligned (angle 0)
    inner = []
    for p in opt_pts:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
