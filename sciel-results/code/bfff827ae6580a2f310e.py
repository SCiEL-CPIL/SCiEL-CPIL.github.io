"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0

# Cache for sweep angles to avoid recomputation if possible, 
# though with 50 nodes it's cheap.
# We need depot coordinates. The interface gives distance_matrix.
# We can recover coordinates relative to depot (0,0) assuming Euclidean distances?
# The task says "distance_matrix", usually Euclidean for these generated instances.
# Let's assume Euclidean and recover coords.
# If not Euclidean, angle calculation is invalid. 
# However, "eight generated 50-customer instances" usually implies random Euclidean points.
# Let's try to recover coordinates.
# d(i,j)^2 = (x_i - x_j)^2 + (y_i - y_j)^2
# Set depot at (0,0).
# x_i = d(i,0) * cos(theta_i)
# y_i = d(i,0) * sin(theta_i)
# We need theta. 
# Actually, we can just use the distance matrix to compute angles between vectors?
# No, we need absolute angles relative to depot.
# Let's assume the distance matrix is Euclidean.
# We can compute coordinates once.

_coords_cache = {}

def _get_coords(distance_matrix):
    # Use hash of matrix shape and first few elements as key? 
    # Simpler: just compute every time, it's O(N^2) which is 2500 ops. Negligible.
    n = distance_matrix.shape[0]
    coords = np.zeros((n, 2))
    # Depot is 0 at (0,0)
    # For i > 0:
    # We need to solve for x_i, y_i given d(i,0) and d(i,j) for some j?
    # Actually, we can't uniquely determine 2D coords from distances to depot alone.
    # We need distances between customers to triangulate.
    # Let's use node 1 to fix the axis.
    # d(1,0) is known. Place node 1 at (d(1,0), 0).
    # Then for any i, we have d(i,0), d(i,1), d(1,0).
    # This is a standard trilateration / intersection of two circles.
    
    if n <= 1:
        return coords
    if n <= 2:
        coords[1, 0] = distance_matrix[1, 0]
        coords[1, 1] = 0
        return coords

    d10 = distance_matrix[1, 0]
    coords[1, 0] = d10
    coords[1, 1] = 0.0

    for i in range(2, n):
        d_i0 = distance_matrix[i, 0]
        d_i1 = distance_matrix[i, 1]
        
        # Intersection of circle centered (0,0) radius d_i0 and circle centered (d10,0) radius d_i1
        # x_i^2 + y_i^2 = d_i0^2
        # (x_i - d10)^2 + y_i^2 = d_i1^2
        # x_i^2 - 2*x_i*d10 + d10^2 + y_i^2 = d_i1^2
        # d_i0^2 - 2*x_i*d10 + d10^2 = d_i1^2
        # 2*x_i*d10 = d_i0^2 + d10^2 - d_i1^2
        x_i = (d_i0**2 + d10**2 - d_i1**2) / (2 * d10)
        
        # y_i^2 = d_i0^2 - x_i^2
        y_sq = d_i0**2 - x_i**2
        if y_sq < 0:
            y_sq = 0
        y_i = math.sqrt(y_sq)
        
        # We need to determine the sign of y_i. 
        # We can use a third point, say node 2, to determine orientation.
        # Or simply pick positive y for all, which might flip some points relative to each other.
        # For sweep angle, relative order matters. If we flip some, angles get messed up.
        # Let's use node 2 (if exists) to determine orientation.
        if i == 2:
            coords[2, 0] = x_i
            # Determine sign based on cross product or just pick positive for now?
            # Actually, if we pick positive for all, we assume all points are in upper half plane?
            # No, that's wrong.
            # Let's compute angle of node 2 relative to 0-1 axis.
            # If we assume node 2 is "above" the axis, y>0.
            # But what if node 3 is "below"?
            # We need a consistent orientation.
            # Let's calculate the signed area or use a reference.
            # For simplicity in a heuristic, if we get the sign wrong for some nodes, 
            # the angular sweep might be disrupted.
            # However, usually random points are scattered.
            # Let's try to determine sign using node 2 as reference for subsequent nodes?
            # No, node 2 is being processed now.
            
            # Let's just pick y > 0 for node 2.
            coords[2, 1] = y_i
        else:
            # Determine sign of y_i for node i using node 2 as reference.
            # We know coords of 0, 1, 2.
            # We know distances d(i,0), d(i,1), d(i,2).
            # We have candidate (x_i, y_i) and (x_i, -y_i).
            # Check which one satisfies d(i,2).
            # d(i,2)^2 = (x_i - x_2)^2 + (y_i - y_2)^2
            # Let's check both.
            x2, y2 = coords[2, 0], coords[2, 1]
            d_i2_sq = distance_matrix[i, 2]**2
            
            dist_pos = (x_i - x2)**2 + (y_i - y2)**2
            dist_neg = (x_i - x2)**2 + (-y_i - y2)**2
            
            if abs(dist_pos - d_i2_sq) < abs(dist_neg - d_i2_sq):
                coords[i, 1] = y_i
            else:
                coords[i, 1] = -y_i
            coords[i, 0] = x_i
            
    return coords

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: Sweep-Angle Priority.
    
    Strategy:
    1. Recover 2D coordinates of all nodes (assuming Euclidean distance matrix).
    2. Calculate polar angle of every node relative to depot (0,0).
    3. Determine the "start angle" of the current route. 
       - If current_node is depot, we need to pick a start. We pick the node with the smallest angle (or arbitrary).
       - If current_node is a customer, the "direction" is the angle of the current_node. 
         We want to visit nodes that are "ahead" in the angular sweep from the start of this route.
         However, we don't know the start of the route from this function signature.
         We only have `current_node`.
         
    Refined Strategy:
    Since we don't track the start of the route, we can approximate the "sector" by looking at the current node's angle.
    A good sweep route visits nodes in increasing angle order.
    If we are at node `u` with angle `theta_u`, we should prefer nodes `v` with `theta_v` close to `theta_u` but slightly larger (wrapping around).
    However, if we just pick the nearest in angle, we might backtrack.
    
    Alternative: 
    Pick the node that minimizes the angular difference to the current node, but with a bias towards completing the "fan".
    Actually, a simpler robust sweep heuristic:
    1. Compute angles for all unvisited nodes.
    2. Compute angle of current node.
    3. If current is depot: Pick the node with the smallest angle (start the sweep).
    4. If current is customer: Pick the node with the smallest positive angular difference from current node (i.e., next in counter-clockwise order).
       This creates a non-crossing fan.
       
    To handle capacity and distance:
    The sweep heuristic is primarily for partitioning. Distance is secondary.
    But we must respect capacity.
    
    Let's implement:
    1. Get coords.
    2. Get angles for unvisited nodes.
    3. If current_node == depot:
         Pick feasible node with minimum angle.
    4. Else:
         current_angle = angle[current_node]
         For each feasible node v:
             diff = (angle[v] - current_angle) % (2*pi)
             We want small diff.
             But also consider distance? Pure sweep might ignore distance significantly.
             Let's score = diff + alpha * distance.
             Or just pick min diff.
             
    Let's try pure sweep first (min angular diff). If current is depot, min angle.
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Get coordinates
    coords = _get_coords(distance_matrix)
    
    # Calculate angles for feasible nodes
    # Angle = atan2(y, x)
    # Depot is 0.
    angles = {}
    for n in feasible:
        x = coords[n, 0]
        y = coords[n, 1]
        angles[n] = math.atan2(y, x)
        
    if current_node == depot:
        # Start the route. Pick the node with the smallest angle.
        # To make it deterministic and cover the circle, we can pick min angle.
        # Or we can pick the node that minimizes distance to depot? 
        # Sweep usually starts at an extreme. Min angle is a good extreme.
        best_node = min(feasible, key=lambda n: angles[n])
        return int(best_node)
    else:
        current_angle = math.atan2(coords[current_node, 1], coords[current_node, 0])
        
        # Find node with smallest positive angular difference
        # diff = (angle[v] - current_angle) % (2*pi)
        # We want to minimize this diff.
        # This ensures we sweep counter-clockwise.
        
        best_node = None
        min_diff = float('inf')
        
        for n in feasible:
            diff = (angles[n] - current_angle) % (2 * math.pi)
            # Add a small penalty for distance to avoid very long jumps if angles are similar?
            # For now, pure angle.
            if diff < min_diff:
                min_diff = diff
                best_node = n
                
        return int(best_node)

# EVOLVE-BLOCK-END
