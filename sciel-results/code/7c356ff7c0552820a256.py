import time
import math

def construct():
    # Harborth's theorem states the maximum number of unit distances for n=32 is 76.
    # A hexagonal spiral of 32 points is a known optimal configuration.
    # We generate the first 32 points of the hexagonal spiral.
    
    points = []
    # Center
    points.append([0, 0])
    
    # Directions for the hexagonal spiral
    # The spiral goes out in rings.
    # Ring k has 6*k points.
    # We need 32 points total.
    # Ring 0: 1 point (0,0)
    # Ring 1: 6 points. Total 7.
    # Ring 2: 12 points. Total 19.
    # Ring 3: 18 points. Total 37.
    # So we need Ring 0, Ring 1, Ring 2, and 32-19=13 points from Ring 3.
    
    # Define directions for the spiral
    # The hexagonal lattice neighbors are:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # But for spiral generation, we move along the perimeter.
    # Let's use the standard hex spiral generation.
    
    # Directions in terms of (a,b) steps for the spiral perimeter
    # 1: (1, 0)
    # 2: (0, 1)
    # 3: (-1, 1)
    # 4: (-1, 0)
    # 5: (0, -1)
    # 6: (1, -1)
    
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate Ring 1
    # Start at (1,0).
    # Move 1 step in dir 0 to (1,0) ? No, (1,0) is the start of the ring.
    # Actually, the points in Ring k are at distance k from center.
    # Let's just list them explicitly or generate carefully.
    
    # Ring 1 points:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Let's add them in spiral order.
    # Start at (1,0).
    # Next is (0,1)? Dist((1,0),(0,1)) = 1^2+1^2-1*1 = 1. Yes.
    # Next is (-1,1)? Dist((0,1),(-1,1)) = 1^2+0^2+0 = 1. Yes.
    # Next is (-1,0)? Dist((-1,1),(-1,0)) = 0+1^2-1*0 = 1. Yes.
    # Next is (0,-1)? Dist((-1,0),(0,-1)) = 1^2+(-1)^2+1(-1) = 1+1-1=1. Yes.
    # Next is (1,-1)? Dist((0,-1),(1,-1)) = 1^2+0+0=1. Yes.
    # Next is (1,0)? Dist((1,-1),(1,0)) = 0+1^2-1*0=1. Yes.
    
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    points.extend([list(p) for p in ring1])
    
    # Ring 2 points
    # Start at (2,0).
    # Path:
    # (2,0)
    # (1,1) -> dist((2,0),(1,1)) = 1+1-1=1.
    # (0,2) -> dist((1,1),(0,2)) = 1+1-2=0? No.
    #   da=1, db=-1. 1+1-1=1.
    #   Wait, (1,1) to (0,2): da=1, db=-1. dist_sq = 1 + 1 - 1 = 1. Correct.
    # (-1,2) -> dist((0,2),(-1,2)) = 1+0+0=1.
    # (-2,1) -> dist((-1,2),(-2,1)) = 1+1+2=4? No.
    #   da=1, db=1. 1+1+1=3. Not 1.
    #   Let's check neighbors of (-1,2).
    #   Neighbors: (0,2), (-1,3), (-2,3), (-2,2), (-2,1), (-1,1).
    #   We want to stay on the ring (dist 2 from origin).
    #   (-2,1): dist_sq((-2,1),(0,0)) = 4+1-2=3. Not 2.
    #   So (-2,1) is NOT on ring 2.
    #   Ring 2 points satisfy a^2+b^2+ab = 4.
    #   Let's list them:
    #   a=2, b=0: 4.
    #   a=1, b=1: 1+1+1=3. No.
    #   a=0, b=2: 4.
    #   a=-1, b=2: 1+4-2=3. No.
    #   a=-2, b=2: 4+4-4=4. Yes.
    #   a=-2, b=1: 4+1-2=3. No.
    #   a=-2, b=0: 4.
    #   a=-1, b=-1: 1+1+1=3. No.
    #   a=0, b=-2: 4.
    #   a=1, b=-2: 1+4-2=3. No.
    #   a=2, b=-2: 4+4-4=4. Yes.
    #   a=2, b=-1: 4+1-2=3. No.
    #   
    #   Wait, my manual listing of ring 2 points is sparse.
    #   Let's use the property that ring k points are at hex distance k.
    #   Hex distance (a,b) = (|a|+|b|+|a+b|)/2.
    #   
    #   Let's generate ring 2 by walking the perimeter.
    #   Start at (2,0).
    #   Next point on perimeter:
    #   The perimeter of a hexagon of radius R consists of 6 segments of length R.
    #   Segment 1: from (R,0) to (0,R). Direction (-1, 1)?
    #     (2,0) -> (1,1) -> (0,2).
    #     Check dist((1,1),(0,0)) = 1+1+1=3. Hex dist = 1.5? No.
    #     Hex dist(1,1) = (1+1+2)/2 = 2. Yes.
    #     So (1,1) is on ring 2.
    #     (0,2) is on ring 2.
    #   Segment 2: from (0,R) to (-R,R). Direction (-1, 0)?
    #     (0,2) -> (-1,2) -> (-2,2).
    #     Hex dist(-1,2) = (1+2+1)/2 = 2. Yes.
    #     Hex dist(-2,2) = (2+2+0)/2 = 2. Yes.
    #   Segment 3: from (-R,R) to (-R,0). Direction (0, -1)?
    #     (-2,2) -> (-2,1) -> (-2,0).
    #     Hex dist(-2,1) = (2+1+1)/2 = 2. Yes.
    #     Hex dist(-2,0) = 2. Yes.
    #   Segment 4: from (-R,0) to (0,-R). Direction (1, -1)?
    #     (-2,0) -> (-1,-1) -> (0,-2).
    #     Hex dist(-1,-1) = (1+1+2)/2 = 2. Yes.
    #     Hex dist(0,-2) = 2. Yes.
    #   Segment 5: from (0,-R) to (R,-R). Direction (1, 0)?
    #     (0,-2) -> (1,-2) -> (2,-2).
    #     Hex dist(1,-2) = (1+2+1)/2 = 2. Yes.
    #     Hex dist(2,-2) = (2+2+0)/2 = 2. Yes.
    #   Segment 6: from (R,-R) to (R,0). Direction (0, 1)?
    #     (2,-2) -> (2,-1) -> (2,0).
    #     Hex dist(2,-1) = (2+1+1)/2 = 2. Yes.
    #     Hex dist(2,0) = 2. Yes.
    
    #   So Ring 2 points in order:
    #   (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)
    #   Total 12 points.
    
    ring2 = [
        [2, 0], [1, 1], [0, 2],
        [-1, 2], [-2, 2], [-2, 1],
        [-2, 0], [-1, -1], [0, -2],
        [1, -2], [2, -2], [2, -1]
    ]
    points.extend(ring2)
    
    # We have 1 + 6 + 12 = 19 points.
    # We need 32 - 19 = 13 more points from Ring 3.
    # Ring 3 perimeter:
    # Segment 1: (3,0) to (0,3). Dir (-1, 1). Points: (3,0), (2,1), (1,2), (0,3).
    # Segment 2: (0,3) to (-3,3). Dir (-1, 0). Points: (-1,3), (-2,3), (-3,3).
    # Segment 3: (-3,3) to (-3,0). Dir (0, -1). Points: (-3,2), (-3,1), (-3,0).
    # ... and so on.
    
    # We need the first 13 points of Ring 3 in spiral order.
    # Start at (3,0).
    # 1. (3,0)
    # 2. (2,1)
    # 3. (1,2)
    # 4. (0,3)
    # 5. (-1,3)
    # 6. (-2,3)
    # 7. (-3,3)
    # 8. (-3,2)
    # 9. (-3,1)
    # 10. (-3,0)
    # 11. (-2,-1)
    # 12. (-1,-2)
    # 13. (0,-3)
    
    ring3_partial = [
        [3, 0], [2, 1], [1, 2], [0, 3],
        [-1, 3], [-2, 3], [-3, 3],
        [-3, 2], [-3, 1], [-3, 0],
        [-2, -1], [-1, -2], [0, -3]
    ]
    points.extend(ring3_partial)
    
    return points
