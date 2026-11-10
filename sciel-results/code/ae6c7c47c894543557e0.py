# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import heapq

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for the corridor such that the intersection of all placements
    (including the start (0,0,0) and end (-2,0,90)) has maximum area.
    Uses a Minkowski erosion approach: finds the largest disk (approximating the sofa)
    that can be kept inside the moving corridor, and returns the corresponding path.
    """
    
    # 1. Define the Corridor Geometry
    # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
    # We represent the corridor as a set of vertices for distance calculation?
    # Or just use the analytic check.
    
    def point_in_corridor(u, v):
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return in_arm1 | in_arm2

    def disk_inside_corridor(cx, cy, r, x, y, a_deg):
        """
        Check if a disk of radius r centered at (cx, cy) in world frame
        is entirely inside the corridor placed at (x, y, a_deg).
        
        This is equivalent to: The minimum distance from the disk center to the 
        boundary of the corridor (in the corridor's local frame) is >= r.
        """
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Transform disk center to corridor local frame
        dx = cx - x
        dy = cy - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # The corridor is the union of two rectangles.
        # R1: [0, 1] x [-3, 1]
        # R2: [-3, 1] x [0, 1]
        
        # Distance from (u, v) to R1
        # If (u, v) is inside R1, dist is 0.
        # If outside, dist is the distance to the nearest edge/corner.
        
        def dist_to_rect(u, v, u_min, u_max, v_min, v_max):
            du = max(u_min - u, 0, u - u_max)
            dv = max(v_min - v, 0, v - v_max)
            return np.sqrt(du**2 + dv**2)
            
        d1 = dist_to_rect(u, v, 0, 1, -3, 1)
        d2 = dist_to_rect(u, v, -3, 1, 0, 1)
        
        # The disk is inside the union if it is inside R1 OR inside R2?
        # No, if the disk straddles the boundary between R1 and R2, it might be fine 
        # as long as it doesn't go outside the union.
        # The union is convex? No.
        # However, if the disk is inside R1, it is in the union.
        # If the disk is inside R2, it is in the union.
        # What if it's in neither? Then it's outside.
        # What if it's partially in R1 and partially in R2?
        # The "gap" is the region outside both.
        # The distance to the complement of the union is the minimum of the distances to the "obstacles".
        # The obstacle is the set of points NOT in R1 AND NOT in R2.
        
        # This is tricky. A simpler check:
        # Sample points on the disk boundary?
        # Or, note that the "corner" of the L-shape is at (0,0) in local frame?
        # R1 is right/up. R2 is left/up.
        # The "inner" corner is at (0,0) but the region (u<0, v<0) is outside.
        # The region (u>1, v>1) is outside.
        
        # Let's use a sampling approach for robustness, or a simpler geometric check.
        # Given the small size of the "sofa" (disk), we can check if the 
        # distance to the "outside" is > r.
        
        # The outside is defined by:
        # 1. u < 0 and v < 0 (Bottom-Left of the L)
        # 2. u > 1 and v > 1 (Top-Right of the L)
        # 3. u > 1 and v < 0 (Right of R1, Below R2) -> Wait, R2 is v>=0.
        # Let's list the constraints for the COMPLEMENT:
        # Not in R1: (u < 0) or (u > 1) or (v < -3) or (v > 1)
        # Not in R2: (u < -3) or (u > 1) or (v < 0) or (v > 1)
        
        # This is complex. Let's use the distance to the "boundary" of the union.
        # The boundary consists of line segments.
        # We can approximate the containment by checking if the disk center is 
        # at least r away from any point NOT in the corridor.
        
        # For efficiency, we'll use a simplified check:
        # The disk is inside if it is inside R1 OR inside R2 OR 
        # (it is in the "corner" region where R1 and R2 meet, but not outside).
        
        # Actually, if d1 <= r and d2 <= r, it's not necessarily inside.
        # If d1 <= r, it's inside R1 (and thus union).
        # If d2 <= r, it's inside R2 (and thus union).
        # If d1 > r and d2 > r, it might still be inside if it's in the "overlap" 
        # region that is not covered by the "distance to rect" logic?
        # No, if it's not in R1 and not in R2, it's outside.
        
        # So the condition is: (d1 <= r) OR (d2 <= r).
        # Wait, if the disk is in the region (0<u<1, 0<v<1), it is in BOTH.
        # d1 will be 0 (if inside R1) and d2 will be 0 (if inside R2).
        
        # What if the disk is in (u<0, v>0)? It is in R2 if u>=-3.
        # d2 will be 0.
        
        # So, `d1 <= r OR d2 <= r` is the correct condition for the disk to be 
        # fully contained in the union?
        # Yes, because if it's not in R1 and not in R2, it's outside.
        # If it is in R1, d1=0 (for the center) but we need the *entire* disk.
        # So we need `dist_to_complement >= r`.
        
        # The complement is the set of points where `not (in_R1 or in_R2)`.
        # This is `(not in_R1) AND (not in_R2)`.
        
        # Let's calculate the distance to the complement directly.
        # The complement has several regions.
        # We can approximate by checking the distance to the "inner corner" (0,0) 
        # and the "outer corners".
        
        # For a robust and fast check, we will sample the disk boundary.
        # This is safer for the L-shape.
        angles = np.linspace(0, 2*np.pi, 16)
        for theta in angles:
            px = u + r * np.cos(theta)
            py = v + r * np.sin(theta)
            if not point_in_corridor(px, py):
                return False
        return True

    # 2. Discretize SE(2)
    # x, y range: The corridor moves from (0,0) to (-2,0).
    # The sofa is likely near the center of the path.
    # Let's search for the sofa center in [-2, 0] x [-1, 1].
    
    # We will optimize the disk center (cx, cy) and radius r.
    # We will run A* for a given disk to find a path.
    
    def find_path_for_disk(cx, cy, r):
        # Grid for SE(2)
        x_range = np.arange(-3.0, 1.5, 0.2)
        y_range = np.arange(-3.0, 1.5, 0.2)
        a_range = np.arange(0, 91, 5)
        
        # Map (x_idx, y_idx, a_idx) to id
        def get_id(xi, yi, ai):
            return (xi * len(y_range) + yi) * len(a_range) + ai
            
        # Precompute validity for all states
        # This is expensive. We will do it on the fly or cache.
        
        # Start and End
        start_id = get_id(np.argmin(np.abs(x_range - 0.0)), np.argmin(np.abs(y_range - 0.0)), 0)
        end_id = get_id(np.argmin(np.abs(x_range - (-2.0))), np.argmin(np.abs(y_range - 0.0)), np.argmin(np.abs(a_range - 90.0)))
        
        # A* Search
        # Heuristic: Euclidean distance in (x, y, a) space
        def heuristic(id1, id2):
            # Decode
            ai1 = id1 % len(a_range)
            tmp = id1 // len(a_range)
            yi1 = tmp % len(y_range)
            xi1 = tmp // len(y_range)
            
            ai2 = id2 % len(a_range)
            tmp = id2 // len(a_range)
            yi2 = tmp % len(y_range)
            xi2 = tmp // len(y_range)
            
            dx = (x_range[xi1] - x_range[xi2]) / 0.2
            dy = (y_range[yi1] - y_range[yi2]) / 0.2
            da = abs(a_range[ai1] - a_range[ai2]) / 5.0
            return np.sqrt(dx**2 + dy**2 + da**2)
            
        def neighbors(id):
            ai = id % len(a_range)
            tmp = id // len(a_range)
            yi = tmp % len(y_range)
            xi = tmp // len(y_range)
            
            nbs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for da in [-1, 0, 1]:
                        if dx==0 and dy==0 and da==0: continue
                        nxi = xi + dx
                        nyi = yi + dy
                        nai = ai + da
                        if 0 <= nxi < len(x_range) and 0 <= nyi < len(y_range) and 0 <= nai < len(a_range):
                            nbs.append(get_id(nxi, nyi, nai))
            return nbs
            
        open_set = []
        heapq.heappush(open_set, (0, start_id))
        came_from = {}
        g_score = {start_id: 0}
        f_score = {start_id: heuristic(start_id, end_id)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert ids to (x, y, a)
                waypoints = []
                for pid in path:
                    ai = pid % len(a_range)
                    tmp = pid // len(a_range)
                    yi = tmp % len(y_range)
                    xi = tmp // len(y_range)
                    waypoints.append([x_range[xi], y_range[yi], a_range[ai]])
                return np.array(waypoints)
                
            for nb in neighbors(current):
                # Check if nb is valid (disk inside corridor)
                ai = nb % len(a_range)
                tmp = nb // len(a_range)
                yi = tmp % len(y_range)
                xi = tmp // len(y_range)
                x, y, a = x_range[xi], y_range[yi], a_range[ai]
                
                if not disk_inside_corridor(cx, cy, r, x, y, a):
                    continue
                    
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score[nb] = tentative_g + heuristic(nb, end_id)
                    heapq.heappush(open_set, (f_score[nb], nb))
                    
        return None

    # 3. Optimize Disk
    best_path = None
    best_area = 0
    
    # Coarse search for center and radius
    # Sofa is likely near the midpoint of the path
    cx_candidates = np.arange(-2.0, 0.0, 0.2)
    cy_candidates = np.arange(-1.0, 1.0, 0.2)
    r_candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for r in r_candidates:
        for cx in cx_candidates:
            for cy in cy_candidates:
                path = find_path_for_disk(cx, cy, r)
                if path is not None:
                    # Evaluate the actual intersection area of this path
                    # (The disk is an approximation)
                    # We can just use the disk area as a proxy for ranking
                    area = np.pi * r**2
                    if area > best_area:
                        best_area = area
                        best_path = path
                        
    if best_path is None:
        # Fallback to seed
        return np.array([[-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0], [-0.3, 0.3, 40.0], [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0], [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0]], dtype=np.float64)

    return best_path

# EVOLVE-BLOCK-END
