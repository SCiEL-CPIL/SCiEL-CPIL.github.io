def construct():
    # Generate the hexagon of radius 2
    points = set()
    # Add center
    points.add((0, 0))
    
    # Add Shell 1
    shell1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for p in shell1:
        points.add(p)
        
    # Add Shell 2
    # The 12 points at distance 2
    shell2 = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -1), (2, 1), (1, 2) # Wait, (2,1) is dist sqrt(7)?
    ]
    # Let's verify shell2 points again.
    # a^2+ab+b^2 = 4
    # (2,0): 4. OK.
    # (1,1): 3. NO. (1,1) is dist sqrt(3). It is in Shell 1? No, Shell 1 is dist 1.
    # My previous shell1 list: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Distances:
    # (1,0): 1.
    # (0,1): 1.
    # (-1,1): 1.
    # (-1,0): 1.
    # (0,-1): 1.
    # (1,-1): 1.
    # So Shell 1 is correct.
    
    # Now Shell 2 (dist sqrt(4)=2? No, dist squared = 4).
    # (2,0): 4.
    # (0,2): 4.
    # (-2,2): 4-4+4=4.
    # (-2,0): 4.
    # (0,-2): 4.
    # (2,-2): 4-4+4=4.
    # (1,1): 3. (This is dist sqrt(3)).
    # (-1,2): 1-2+4=3.
    # (-2,1): 4-2+1=3.
    # (-1,-1): 3.
    # (1,-2): 1-2+4=3.
    # (2,-1): 4-2+1=3.
    
    # So there are no points with dist squared = 4 except the 6 "corners"?
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # What about the other 6 points of Shell 2?
    # The hexagon of radius 2 has 19 points.
    # Shell 0: 1.
    # Shell 1: 6.
    # Shell 2: 12.
    # So there MUST be 12 points with dist <= 2.
    # Distances are sqrt(a^2+ab+b^2).
    # Values: 1, sqrt(3), 2.
    # Points with dist 1: 6.
    # Points with dist sqrt(3): 6.
    # Points with dist 2: 6.
    # Total 1 + 6 + 6 + 6 = 19.
    # So "Shell 2" in terms of hex distance (steps) includes both dist sqrt(3) and dist 2.
    # Hex distance (steps) is (|a|+|b|+|a+b|)/2.
    # (1,1): (1+1+2)/2 = 2.
    # (2,0): (2+0+2)/2 = 2.
    # So yes, all 12 points have hex distance 2.
    
    shell2 = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -1), (2, 1), (1, 2) # Wait, (2,1) is hex dist (2+1+3)/2 = 3.
    ]
    # Let's list hex distance 2 points properly.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
    # (-1,-1), (0,-2), (1,-2), (2,-1), (2,1) is hex dist 3.
    # (1,2) is hex dist 3.
    # The 12 points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
    # (-1,-1), (0,-2), (1,-2), (2,-1)
    # And?
    # (1,-1) is hex dist 1.
    # (2,1) is hex dist 3.
    # (1,2) is hex dist 3.
    # I am missing 2 points.
    # Let's iterate.
    for a in range(-2, 3):
        for b in range(-2, 3):
            if (a,b) == (0,0): continue
            d = (abs(a) + abs(b) + abs(a+b)) // 2
            if d == 2:
                # print(a,b)
                pass
    # The points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
    # (-1,-1), (0,-2), (1,-2), (2,-1)
    # (2,1) is (2+1+3)/2 = 3.
    # (1,2) is (1+2+3)/2 = 3.
    # (-2,2) is (2+2+0)/2 = 2.
    # (-1,3) is (1+3+2)/2 = 3.
    # So (-2,2) is in Shell 2.
    # And (2,-2) is in Shell 2.
    
    shell2 = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-2, 2), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2)
    ]
    # Let's verify count. 12 points.
    for p in shell2:
        points.add(p)

    # Now add 13 points from Shell 3 (hex distance 3)
    # We want an arc of 13 starting with Side, ending with Side.
    # The S3 points in order around the ring:
    # We identified the pattern C S S C S S ...
    # Let's list S3 points in angular order.
    # (3,0) [C]
    # (2,1) [S]
    # (1,2) [S]
    # (0,3) [C]
    # (-1,3) [S]
    # (-2,2) [S]
    # (-3,1) [C]
    # (-3,0) [C] -- Wait, (-3,0) is C.
    # (-3,-1) [S]
    # (-2,-2) [S]
    # (-1,-3) [C]
    # (0,-3) [C] -- Wait.
    
    # Let's just generate all S3 points and sort them by angle.
    s3_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = (abs(a) + abs(b) + abs(a+b)) // 2
            if d == 3:
                s3_points.append((a,b))
                
    # Sort by angle
    import math
    def angle(p):
        # Map to Cartesian: x = a + b/2, y = b*sqrt(3)/2
        x = p[0] + p[1] / 2.0
        y = p[1] * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    s3_points.sort(key=angle)
    
    # We need to pick a contiguous arc of 13.
    # We want to maximize (2 * #Sides + 1 * #Corners).
    # #Sides = 13 - #Corners.
    # Score = 2(13-C) + C = 26 - C.
    # Maximize by minimizing C.
    # Min C in an arc of 13 is 4 (as derived).
    # So we just need to find an arc of 13 with 4 corners.
    # There are 6 corners in the full ring of 18.
    # Any arc of 13 will have 4 or 5 corners.
    # We want 4.
    
    best_arc = None
    min_corners = 100
    
    # Define which are corners.
    # Corners are (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3).
    corner_set = {(3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)}
    
    n = len(s3_points) # 18
    for i in range(n):
        arc = []
        corners = 0
        for j in range(13):
            p = s3_points[(i+j) % n]
            arc.append(p)
            if p in corner_set:
                corners += 1
        if corners < min_corners:
            min_corners = corners
            best_arc = arc
            
    for p in best_arc:
        points.add(p)
        
    return [[a, b] for a, b in points]
