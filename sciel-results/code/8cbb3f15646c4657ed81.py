import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers of 11 hexagons in a honeycomb lattice
    # Center (0,0)
    # Layer 1: 6 neighbors at distance sqrt(3)
    # Layer 2: 4 specific neighbors at distance 3 to minimize max radius
    
    # Axial coordinates (q, r)
    # Center
    centers = [(0, 0)]
    
    # Layer 1
    layer1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    centers.extend(layer1)
    
    # Layer 2: Pick the 4 with smallest distance from origin
    # Based on analysis, these are (2, -1), (1, -2), (-1, 2), (-2, 1)
    # Let's verify distances again to be safe in code logic, or just hardcode if confident.
    # To be safe, generate all 12 Layer 2 candidates and pick 4 with min distance.
    
    layer2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2)
    ]
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    def center_distance(q, r):
        x, y = axial_to_cartesian(q, r)
        return math.sqrt(x*x + y*y)
    
    # Sort Layer 2 candidates by distance
    sorted_layer2 = sorted(layer2_candidates, key=lambda c: center_distance(c[0], c[1]))
    
    # Take the first 4
    selected_layer2 = sorted_layer2[:4]
    centers.extend(selected_layer2)
    
    # Convert centers to Cartesian
    cart_centers = [axial_to_cartesian(q, r) for q, r in centers]
    
    # Calculate max radius (distance from origin to any vertex of any hexagon)
    # Hexagon side length is 1. Circumradius is 1.
    # Max radius = max(center_distance) + 1
    max_center_dist = max(center_distance(q, r) for q, r in centers)
    outer_side = max_center_dist + 1.0
    
    # Construct inner list: (x, y, angle_degrees)
    # We can keep angle 0 for all, as the outer hexagon rotation can be adjusted or just set to 0.
    # The problem says "each inner hexagon may translate and rotate".
    # The outer hexagon rotation is returned as well.
    # If we align the outer hexagon with the main axes of the lattice, angle 0 is fine.
    
    inner = []
    for x, y in cart_centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon center is (0,0)
    # Side length is outer_side
    # Angle 0.0
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
