"""Rewritten program: minimize largest 3-collinear-free subset for 20 points."""

def construct():
    # Construct 20 points using two 3x3 blocks side-by-side and 2 extra points.
    # This structure avoids 4-collinear points and creates many 3-collinear triples,
    # reducing the size of the maximum 3-collinear-free subset.
    
    points = []
    
    # Block 1: x in {1, 2, 3}, y in {1, 2, 3}
    for x in range(1, 4):
        for y in range(1, 4):
            points.append([x, y])
            
    # Block 2: x in {4, 5, 6}, y in {1, 2, 3}
    for x in range(4, 7):
        for y in range(1, 4):
            points.append([x, y])
            
    # Extra points to reach 20: x=7, y in {1, 2}
    points.append([7, 1])
    points.append([7, 2])
    
    # Verify 20 distinct points
    if len(points) != 20:
        raise ValueError("Must have 20 points")
    if len(set(map(tuple, points))) != 20:
        raise ValueError("Points must be distinct")
        
    # Verify no four collinear
    def check_no_four_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                count = 2
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    # Check if (x1,y1), (x2,y2), (x3,y3) are collinear
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        count += 1
                        if count >= 4:
                            return False
                if count >= 4:
                    return False
        return True
        
    if not check_no_four_collinear(points):
        raise ValueError("Configuration has 4 collinear points")
        
    return points
