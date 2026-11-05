import math

def construct():
    # Target number of points
    n = 32
    
    # Helper to get hex distance
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
        
    # Generate points in rings 0 to 3
    # This forms a hexagon of radius 3 with 37 points
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                points.append((a, b))
                
    # We need to remove 5 points to maximize edges.
    # Based on analysis, removing a connected chain of 5 perimeter points
    # that includes 2 corners minimizes the edge loss.
    # One such set is:
    # (3, 0)   [Corner]
    # (3, -1)  [Edge]
    # (3, -2)  [Edge]
    # (3, -3)  [Corner]
    # (2, -3)  [Edge]
    
    points_to_remove = {
        (3, 0),
        (3, -1),
        (3, -2),
        (3, -3),
        (2, -3)
    }
    
    # Filter out the points to remove
    result_points = [p for p in points if p not in points_to_remove]
    
    # Ensure we have exactly 32 points
    # If something went wrong, fallback to the first 32
    if len(result_points) != n:
        result_points = points[:n]
        
    # Convert to list of lists [a, b]
    result = [[a, b] for a, b in result_points]
    
    return result
