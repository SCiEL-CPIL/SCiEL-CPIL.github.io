def _generate_hexagonal(n):
    """Generate n points in a hexagonal arrangement."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    # Find smallest R such that 1 + 3R(R+1) >= n
    R = 0
    while 1 + 3 * R * (R + 1) < n:
        R += 1
        
    # Generate all points in hexagon of radius R
    # Points (a, b) such that max(|a|, |b|, |a-b|) <= R ? 
    # No, for triangular lattice coords a+b*omega, the hex distance is max(|a|, |b|, |a+b|) is not quite right.
    # Let's just iterate a, b in range [-R, R] and check if dist_sq <= R^2? 
    # No, hexagon shape is defined by hex distance.
    # Hex distance d(a,b) = (|a| + |b| + |a+b|) / 2 ?
    # Let's check (1,0): (1+0+1)/2 = 1. Correct.
    # (1,1): (1+1+2)/2 = 2. Correct.
    # (0,1): (0+1+1)/2 = 1. Correct.
    # So condition: (|a| + |b| + |a+b|) // 2 <= R
    
    points = []
    # We want spiral order. 
    # Simplest robust way: generate all valid points, sort by (dist_sq, angle), take first n.
    # Or just generate layer by layer correctly.
    
    # Let's try layer by layer again, carefully.
    # Layer 0: (0,0)
    # Layer k: 6k points.
    # Directions: 
    # 1: (1, 0) -> (0, 1) ? No.
    # Let's use the standard cube coords mapping or just hardcode the 6 directions for the spiral.
    # Start of layer k is (k, 0).
    # The 6 directions to traverse the perimeter are:
    # 1. (-1, 1)  # towards (0, k)
    # 2. (-1, -1) # towards (-k, 0)
    # 3. (0, -1)  # towards (-k, -k)
    # 4. (1, 0)   # towards (0, -k)
    # 5. (1, 1)   # towards (k, 0)
    # 6. (0, 1)   # towards (k, k) -- wait, this closes the loop?
    
    # Let's trace Layer 1 (k=1) with these directions starting at (1,0):
    # Start (1,0)
    # Dir 1 (-1,1): (0,1)
    # Dir 2 (-1,-1): (-1,0)
    # Dir 3 (0,-1): (-1,-1) -> dist sqrt(3). This is wrong for layer 1.
    
    # The issue is that the "hexagon" of radius 1 in triangular lattice coords (a,b) with metric a^2+ab+b^2=1
    # has points: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Let's check distances:
    # (1,0): 1
    # (0,1): 1
    # (-1,1): 1-1+1=1
    # (-1,0): 1
    # (0,-1): 1
    # (1,-1): 1-1+1=1
    # Yes.
    
    # So for Layer k, we need points at hex-distance k.
    # The perimeter of a hexagon of radius k in this lattice consists of 6k points.
    # The vertices are:
    # (k, 0)
    # (0, k)
    # (-k, k)
    # (-k, 0)
    # (0, -k)
    # (k, -k)
    
    # Let's walk the perimeter from (k, 0).
    # Side 1: (k, 0) to (0, k). Delta (-1, 1). k steps.
    # Points: (k-1, 1), ..., (0, k). (k points)
    # Side 2: (0, k) to (-k, k). Delta (-1, 0). k steps.
    # Points: (-1, k), ..., (-k, k). (k points)
    # Side 3: (-k, k) to (-k, 0). Delta (0, -1). k steps.
    # Points: (-k, k-1), ..., (-k, 0). (k points)
    # Side 4: (-k, 0) to (0, -k). Delta (1, -1). k steps.
    # Points: (-k+1, -1), ..., (0, -k). (k points)
    # Side 5: (0, -k) to (k, -k). Delta (1, 0). k steps.
    # Points: (1, -k), ..., (k, -k). (k points)
    # Side 6: (k, -k) to (k, 0). Delta (0, 1). k steps.
    # Points: (k, -k+1), ..., (k, 0). (k points)
    
    # Total 6k points.
    # Note: The starting point (k,0) is the end of Side 6. We should not add it again if we start the list empty.
    # We start at (k,0). We do NOT add (k,0) to the list initially.
    # We take k steps in Dir 1, k steps in Dir 2, etc.
    
    points = [(0, 0)]
    if n == 1:
        return points
        
    k = 1
    while len(points) < n:
        a, b = k, 0
        # Side 1
        for _ in range(k):
            a -= 1; b += 1
            points.append((a, b))
        # Side 2
        for _ in range(k):
            a -= 1; b += 0
            points.append((a, b))
        # Side 3
        for _ in range(k):
            a -= 0; b -= 1
            points.append((a, b))
        # Side 4
        for _ in range(k):
            a += 1; b -= 1
            points.append((a, b))
        # Side 5
        for _ in range(k):
            a += 1; b += 0
            points.append((a, b))
        # Side 6
        for _ in range(k):
            a += 0; b += 1
            points.append((a, b))
            
        k += 1
        
    return points[:n]
