import time

def construct():
    # Generate hexagon of radius 3
    hex_points = set()
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex_points.add((a, b))
    
    # Points to remove
    # Chain: (3,0) - (2,1) - (1,2) - (0,3)
    # Isolated Corner: (-3,0)
    to_remove = {(3, 0), (2, 1), (1, 2), (0, 3), (-3, 0)}
    
    points = [p for p in hex_points if p not in to_remove]
    
    if len(points) != 32:
        # Fallback
        return [[a, b] for a in range(16) for b in range(2)]
        
    return [[a, b] for a, b in points]
