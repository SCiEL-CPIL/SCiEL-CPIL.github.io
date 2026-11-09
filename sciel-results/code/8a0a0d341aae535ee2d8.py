import math

def construct():
    points = []
    # Add points with hex distance <= 2
    for a in range(-2, 3):
        for b in range(-2, 3):
            # Check hex distance
            # dist = max(abs(a), abs(b), abs(a+b))
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                points.append((a, b))
    
    # Generate points with hex distance == 3
    ring_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                ring_points.append((a, b))
                
    # Sort ring points by angle
    # x = a + b/2
    # y = b * sqrt(3) / 2
    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    ring_points.sort(key=angle)
    
    # Add first 13 points from the ring
    points_to_add = ring_points[:13]
    points.extend(points_to_add)
    
    # Convert to list of lists as required
    return [[a, b] for a, b in points]
