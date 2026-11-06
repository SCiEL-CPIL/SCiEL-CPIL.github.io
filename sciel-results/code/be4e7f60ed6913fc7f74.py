def construct():
    # Generate points in a spiral order
    points = []
    # We need 32 points.
    # Hexagon of radius 3 has 37 points.
    # So we generate up to radius 3.
    
    # Helper to get points in a ring
    def get_ring(k):
        if k == 0:
            return [(0,0)]
        # The ring k consists of 6k points.
        # We can traverse the perimeter.
        # Start at (k, 0) and move.
        # Directions in axial coordinates (a,b):
        # 1: (1, 0)
        # 2: (0, 1)
        # 3: (-1, 1)
        # 4: (-1, 0)
        # 5: (0, -1)
        # 6: (1, -1)
        
        # Start point: (k, 0)
        # Wait, (k,0) is at distance k.
        # Let's start at (k, 0) and go around.
        # Step 1: move (0, 1) k times? No.
        # The side length is k.
        # From (k, 0), move (0, 1) k times -> (k, k).
        # Then (-1, 1) k times -> (0, 2k).
        # Then (-1, 0) k times -> (-k, 2k).
        # Then (0, -1) k times -> (-k, k).
        # Then (1, -1) k times -> (0, 0)? No.
        # Let's trace carefully.
        
        pts = []
        # Start at (k, 0)
        a, b = k, 0
        for i in range(k):
            pts.append((a, b))
            a += 0; b += 1 # Move (0,1)
        for i in range(k):
            pts.append((a, b))
            a += -1; b += 1 # Move (-1,1)
        for i in range(k):
            pts.append((a, b))
            a += -1; b += 0 # Move (-1,0)
        for i in range(k):
            pts.append((a, b))
            a += 0; b += -1 # Move (0,-1)
        for i in range(k):
            pts.append((a, b))
            a += 1; b += -1 # Move (1,-1)
        for i in range(k):
            pts.append((a, b))
            a += 1; b += 0 # Move (1,0)
            # This brings us back to (k,0)?
            # Let's check k=1.
            # Start (1,0).
            # 1. (1,0) -> (1,1)
            # 2. (1,1) -> (0,2)
            # 3. (0,2) -> (-1,2)
            # 4. (-1,2) -> (-1,1)
            # 5. (-1,1) -> (0,0) -- Wait, this is center.
            # My directions are wrong for a simple perimeter walk.
            
        # Let's use a known spiral generator.
        # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Step lengths: 1, 1, 2, 2, 3, 3, 4, 4...
        # This generates a spiral.
        
    # Simpler: Just generate all points in a box, sort by graph distance, then by angle.
    
    R = 10
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            # Angle approximation
            # x = a + b/2, y = b * sqrt(3)/2
            # angle = atan2(y, x)
            # To avoid math import overhead in tight loop (though R is small), we can just sort by d, then a, then b.
            # But a, b order might not be angular.
            # Let's use atan2.
            import math
            x = a + 0.5 * b
            y = (math.sqrt(3) / 2) * b
            angle = math.atan2(y, x)
            candidates.append((d, angle, a, b))
            
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    selected = []
    for d, angle, a, b in candidates:
        selected.append([a, b])
        if len(selected) == 32:
            break
            
    return selected
