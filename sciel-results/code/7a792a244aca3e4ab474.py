import math
from itertools import combinations
import time

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def concyclic(p1, p2, p3, p4):
    # Determinant of the circle equation matrix
    # x^2 + y^2 + Dx + Ey + F = 0
    # x1^2+y1^2 x1 y1 1
    # x2^2+y2^2 x2 y2 1
    # x3^2+y3^2 x3 y3 1
    # x4^2+y4^2 x4 y4 1
    a1 = p1[0]**2 + p1[1]**2
    a2 = p2[0]**2 + p2[1]**2
    a3 = p3[0]**2 + p3[1]**2
    a4 = p4[0]**2 + p4[1]**2
    
    # Calculate determinant using expansion or Sarrus
    # For efficiency, we can check if p4 lies on the circle defined by p1, p2, p3
    # Circle through p1, p2, p3:
    # (x - cx)^2 + (y - cy)^2 = r^2
    # 2(x2-x1)x + 2(y2-y1)y = x2^2-x1^2 + y2^2-y1^2
    # 2(x3-x1)x + 2(y3-y1)y = x3^2-x1^2 + y3^2-y1^2
    # Solve for cx, cy
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if d == 0:
        # Points are collinear, so they don't define a unique circle, 
        # but the problem states no three collinear in the set.
        # If p1,p2,p3 are collinear, this case shouldn't happen for valid sets.
        return False 
        
    # Use exact integer arithmetic for the check
    # ux = U/d, uy = V/d
    U = (x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)
    V = (x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)
    
    # Check if (x4, y4) is on the circle defined by (U, V, d)
    # (d*x4 - U)^2 + (d*y4 - V)^2 == (d*x1 - U)^2 + (d*y1 - V)^2
    lhs = (d*x4 - U)**2 + (d*y4 - V)**2
    rhs = (d*x1 - U)**2 + (d*y1 - V)**2
    
    return lhs == rhs

def construct():
    # Start with a known good set of 5 points
    S = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # We will try to expand this set.
    # Strategy: Iterate over a growing box.
    # To ensure we finish within 120s, we need to be careful with the grid size.
    # 1000x1000 = 10^6 points. 
    # For each point, we check distance to 5 points (5 ops), collinearity (10 pairs), concyclicity (10 triples).
    # This is roughly 25 heavy ops per point. 25 * 10^6 = 2.5 * 10^8 ops.
    # Python can do ~10^7 simple ops/sec. So 2.5 * 10^8 might take ~25-50 seconds.
    # We should limit the range or optimize.
    
    # Let's try a range of 500 first, then 1000 if time permits.
    # Actually, the seed found 5 points. Let's see if we can find 6, 7, etc.
    
    start_time = time.time()
    timeout = 110  # Leave 10s for overhead
    
    current_range = 100
    found_new = True
    
    while found_new and time.time() - start_time < timeout:
        found_new = False
        # Expand range if we haven't found a new point in the current range?
        # Or just iterate through a fixed large range?
        # Iterating a fixed large range is safer to ensure we don't miss points that are "far" but necessary.
        # However, a spiral or expanding box is better for finding *new* points.
        
        # Let's stick to a expanding box strategy.
        # If we find a point, we reset the "found_new" logic for the next iteration?
        # No, we add the point and continue searching in the same box? 
        # The seed broke and restarted. This is good because the new point changes the constraints.
        
        # To be more efficient, we can iterate y, then x.
        for y in range(-current_range, current_range + 1):
            if time.time() - start_time > timeout:
                break
            for x in range(-current_range, current_range + 1):
                if time.time() - start_time > timeout:
                    break
                
                P = (x, y)
                if P in S:
                    continue
                
                # Check distances
                valid = True
                for Q in S:
                    d_sq = dist_sq(P, Q)
                    if not is_square(d_sq):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check collinearity
                for Q1, Q2 in combinations(S, 2):
                    if collinear(P, Q1, Q2):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check concyclicity
                for Q1, Q2, Q3 in combinations(S, 3):
                    if concyclic(P, Q1, Q2, Q3):
                        valid = False
                        break
                
                if valid:
                    S.append(P)
                    found_new = True
                    # Break out to restart with new S in the same range?
                    # Or break to expand range?
                    # Restarting in the same range is better to catch nearby points.
                    break
            if found_new:
                break
            if time.time() - start_time > timeout:
                break
        
        # If no new point found in this range, expand the range
        if not found_new:
            current_range *= 2
            if current_range > 2000:
                break
        # If we found a point, we stay in the same range to see if more can be added?
        # The loop structure above breaks out of x,y loops and goes back to while.
        # It will restart scanning the same range with the new point included.
        # This is correct.
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in S]}
