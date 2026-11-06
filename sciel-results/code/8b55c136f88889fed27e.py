import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Distance squared on triangular lattice
    # Points are (a, b) corresponding to a + b*omega
    # |z|^2 = a^2 + b^2 + ab
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Generate hexagon of radius R
    def gen_hex(R):
        pts = []
        for x in range(-R, R+1):
            for y in range(-R, R+1):
                # Hex distance in axial coords: max(|x|, |y|, |x+y|) is NOT correct for triangular lattice distance 1 neighbors.
                # Triangular lattice neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
                # The hexagon shape in axial coords for radius R is defined by |x|<=R, |y|<=R, |x+y|<=R?
                # Let's check R=1.
                # (0,0) -> 0
                # (1,0) -> max(1,0,1)=1. OK.
                # (0,1) -> max(0,1,1)=1. OK.
                # (-1,1) -> max(1,1,0)=1. OK.
                # (-1,0) -> max(1,0,-1)=1. OK.
                # (0,-1) -> max(0,1,-1)=1. OK.
                # (1,-1) -> max(1,1,0)=1. OK.
                # (1,1) -> max(1,1,2)=2. Excluded. Correct, dist^2 = 1+1+1=3.
                # So condition is max(abs(x), abs(y), abs(x+y)) <= R
                if max(abs(x), abs(y), abs(x+y)) <= R:
                    pts.append([x, y])
        return pts

    # Base set: Hexagon of radius 2 (19 points)
    S = gen_hex(2)
    
    # Candidates: Ring 3 (points with hex distance 3)
    C = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            if max(abs(x), abs(y), abs(x+y)) == 3:
                C.append([x, y])
    
    # Greedy selection to add 13 points
    while len(S) < 32:
        best_gain = -1
        best_pt = None
        current_edges = count_edges(S)
        
        for c in C:
            if c in S:
                continue
            # Calculate gain
            new_S = S + [c]
            new_edges = count_edges(new_S)
            gain = new_edges - current_edges
            if gain > best_gain:
                best_gain = gain
                best_pt = c
        
        if best_pt is None:
            break
            
        S.append(best_pt)
        
        # Time check
        if time.time() > deadline - 5:
            break

    # If we didn't reach 32, pad with arbitrary valid points (should not happen)
    if len(S) < 32:
        # This is a fallback, but greedy should work
        pass

    return S

