import numpy as np
import math
import time

N = 11

def hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Returns 6 vertices of a regular hexagon."""
    angles = np.deg2rad(angle_deg + np.arange(0, 360, 60))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def get_layouts():
    """Generates candidate layouts of 11 unit hexagons on a hex grid."""
    layouts = []
    
    # Helper to add a hex at axial coordinates
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        return x, y

    # Layout 1: Straight line
    # Centers along x-axis
    line_centers = [(2.0 * i, 0.0) for i in range(-(N//2), N - N//2)]
    layouts.append(line_centers)
    
    # Layout 2: 7-hex cluster + 4 from 2nd ring
    # 7-hex cluster: q,r in { (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) }
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # 2nd ring axial coords
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    ring_2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), 
              (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    
    # Try adding 4 from ring 2 in different patterns
    # Pattern A: 4 consecutive
    for start in range(12):
        indices = [ (start + i) % 12 for i in range(4) ]
        extra = [ring_2[i] for i in indices]
        layout = base_7 + extra
        # Convert to cartesian
        cart_layout = [axial_to_cartesian(q, r) for q, r in layout]
        layouts.append(cart_layout)
        
    # Pattern B: 4 spread out (every 3rd)
    indices = [0, 3, 6, 9]
    extra = [ring_2[i] for i in indices]
    layout = base_7 + extra
    cart_layout = [axial_to_cartesian(q, r) for q, r in layout]
    layouts.append(cart_layout)
    
    # Pattern C: 2 opposite pairs
    indices = [0, 6, 1, 7]
    extra = [ring_2[i] for i in indices]
    layout = base_7 + extra
    cart_layout = [axial_to_cartesian(q, r) for q, r in layout]
    layouts.append(cart_layout)
    
    # Layout 3: Rectangular-ish 3x3 + 2
    # Row 0: 3 hexes
    # Row 1: 3 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # This is not a standard hex grid tiling, but let's try a "brick" layout
    # Actually, let's just use the ones generated above. They are likely sufficient.
    
    return layouts

def min_enclosing_hexagon(points):
    """
    Finds the minimum side length S and center (cx, cy) and angle theta
    of a regular hexagon enclosing the given points.
    """
    points = np.array(points)
    if len(points) == 0:
        return 0.0, (0.0, 0.0), 0.0
        
    best_S = float('inf')
    best_theta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Sweep angle from 0 to 60 degrees
    # Step size 0.1 degrees for precision
    thetas = np.arange(0, 60, 0.1)
    
    for theta_deg in thetas:
        theta = np.deg2rad(theta_deg)
        
        # The 3 directions of the hexagon axes (normals to sides)
        # Angle 0, 60, 120 relative to the hexagon orientation
        # If hexagon is rotated by theta, the axes are at theta, theta+60, theta+120
        axes = [theta, theta + np.pi/3, theta + 2*np.pi/3]
        
        # Project points onto these 3 axes
        # Width in each direction
        widths = []
        for ax in axes:
            cos_ax = np.cos(ax)
            sin_ax = np.sin(ax)
            # Projection
            proj = points[:, 0] * cos_ax + points[:, 1] * sin_ax
            w = np.max(proj) - np.min(proj)
            widths.append(w)
            
        w0, w1, w2 = widths
        
        # Constraints:
        # 2S >= w0
        # sqrt(3)S >= w1
        # sqrt(3)S >= w2
        S_candidate = max(w0 / 2.0, w1 / math.sqrt(3), w2 / math.sqrt(3))
        
        if S_candidate < best_S:
            best_S = S_candidate
            best_theta = theta_deg
            # Calculate center
            # The center is the midpoint of the projections in each direction
            # We need to find cx, cy such that the projections are centered.
            # For each axis, the midpoint of the projection range is m_i.
            # The center (cx, cy) projected onto axis i must be m_i.
            # We have 3 equations for 2 unknowns (cx, cy).
            # We can solve this using least squares or just pick the first two axes.
            
            # Let's compute midpoints
            mids = []
            for ax in axes:
                cos_ax = np.cos(ax)
                sin_ax = np.sin(ax)
                proj = points[:, 0] * cos_ax + points[:, 1] * sin_ax
                m = (np.max(proj) + np.min(proj)) / 2.0
                mids.append(m)
                
            # Solve for cx, cy
            # cos(ax0)*cx + sin(ax0)*cy = m0
            # cos(ax1)*cx + sin(ax1)*cy = m1
            # cos(ax2)*cx + sin(ax2)*cy = m2
            
            A = np.array([
                [np.cos(axes[0]), np.sin(axes[0])],
                [np.cos(axes[1]), np.sin(axes[1])]
            ])
            b = np.array([mids[0], mids[1]])
            
            try:
                center = np.linalg.solve(A, b)
                best_cx, best_cy = center
            except np.linalg.LinAlgError:
                # Fallback to centroid
                best_cx, best_cy = np.mean(points, axis=0)
                
    return best_S, (best_cx, best_cy), best_theta

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    layouts = get_layouts()
    
    best_result = None
    best_score = float('inf')
    
    for layout in layouts:
        if time.time() > deadline:
            break
            
        # Get all vertices of the inner hexagons
        # Assume inner hexagons are aligned with angle 0 for the calculation
        # We will rotate the outer hexagon to fit them.
        # The inner hexagons themselves can be rotated, but since they are in a grid,
        # they are aligned with each other. We can assume they are all at angle 0
        # relative to the grid, and we find the outer hexagon that fits them.
        
        all_vertices = []
        for cx, cy in layout:
            verts = hexagon_vertices(cx, cy, 0.0)
            all_vertices.extend(verts.tolist())
            
        S, (cx_out, cy_out), angle_out = min_enclosing_hexagon(all_vertices)
        
        if S < best_score:
            best_score = S
            best_result = (layout, (cx_out, cy_out), S, angle_out)
            
    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    layout, (cx_out, cy_out), S, angle_out = best_result
    
    # Construct the return values
    # Inner rows: (x, y, angle_degrees)
    # The inner hexagons are at the layout centers, angle 0.
    # Note: The outer hexagon is rotated by angle_out.
    # Does this matter for the inner hexagons?
    # The task says "each inner hexagon may translate and rotate".
    # Our calculation assumed inner hexagons are at angle 0.
    # If we rotate the outer hexagon by angle_out, the inner hexagons are still at angle 0.
    # This is a valid configuration.
    
    inner = [[cx, cy, 0.0] for cx, cy in layout]
    
    return inner, (cx_out, cy_out), S, angle_out

