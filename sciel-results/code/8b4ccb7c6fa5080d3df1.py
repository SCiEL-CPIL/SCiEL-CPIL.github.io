import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field arithmetic helpers
    # F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # alpha^2 = k
        r0 = (a[0]*b[0] + a[1]*b[1]*k) % p
        r1 = (a[0]*b[1] + a[1]*b[0]) % p
        return (r0, r1)
        
    def inv(a):
        if a == (0, 0):
            raise ValueError("Inv of 0")
        # a = a0 + a1*alpha
        # a * (a0 - a1*alpha) = a0^2 - a1^2*k
        # inv = (a0 - a1*alpha) / (a0^2 - a1^2*k)
        d = (a[0]*a[0] - a[1]*a[1]*k) % p
        d_inv = pow(d, p-2, p)
        r0 = (a[0] * d_inv) % p
        r1 = ((-a[1]) * d_inv) % p
        return (r0, r1)

    # Generate all points
    points = []
    point_coords = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = len(points)
                    points.append([[xa, xb], [ya, yb]])
                    point_coords[idx] = ((xa, xb), (ya, yb))
    
    num_points = len(points) # 2401
    
    # Generate lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Note: (1, m) covers all slopes. (0, 1) is vertical.
    # Total 49 + 1 = 50 directions.
    
    lines = []
    line_points = []
    point_lines = [[] for _ in range(num_points)]
    
    def get_line_id(d, base_point):
        # Find line with direction d passing through base_point
        # We can just check against existing lines? No, we build them.
        pass

    # Build lines for each direction
    # For a direction v, the lines are sets of points {p + t*v}
    # We can identify a line by its "base" point (smallest index?) or by an invariant.
    # For v=(1, m), line is determined by y - m*x.
    # For v=(0, 1), line is determined by x.
    
    # Let's just generate all lines explicitly.
    line_id = 0
    lines_map = {} # (dir_idx, invariant) -> line_id
    
    # Direction 0: (0, 1) (Vertical)
    # Invariant: x
    for x in range(q):
        # x is a point in F_49? No, x is coordinate.
        # Let's iterate over points to form lines.
        pass
        
    # Easier: Iterate over all points, for each direction, find the line.
    # But we need to group them.
    
    # Let's create a list of all lines.
    # There are 50 directions * 49 lines = 2450 lines.
    
    # Direction vectors
    dirs = []
    for m in range(q):
        # m is represented as (ma, mb)
        # We need to iterate m in F_49.
        # Let's generate all elements of F_49.
        pass

    # List of all elements in F_49
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    # Directions
    dir_vectors = []
    for m in field_elements:
        dir_vectors.append(((1, 0), m)) # (1, m) -> (1,0) + (0,1)*m? No.
        # Vector v = (1, m).
        # In coordinates: (1, 0) for x-part? No.
        # Point is (X, Y) where X, Y in F_49.
        # Direction (1, m) means step (1, m).
        # So v = ((1,0), m).
    dir_vectors.append(((0, 0), (0, 1))) # (0, 1)
    
    # Wait, (1, m) for m in F_49.
    # m is a field element.
    # So v = ((1,0), m).
    
    # Let's map field elements to indices 0..48
    elem_to_idx = {e: i for i, e in enumerate(field_elements)}
    
    # Re-define dirs with indices
    # dir 0..48: (1, m)
    # dir 49: (0, 1)
    
    # For each direction, partition points into lines.
    # A line is defined by a point on it.
    # To avoid duplicates, we can use the "first" point in some ordering?
    # Or just generate all q lines for each direction.
    # For direction v, the lines are in 1-1 correspondence with points in a transversal.
    # For v=(1, m), the x-coordinate changes by 1. So lines are determined by Y - m*X.
    # For v=(0, 1), lines are determined by X.
    
    # Let's compute the invariant for each point for each direction.
    # This is O(q^2 * q) = 2401 * 50 * ... ?
    # We can just do it.
    
    # line_id = 0
    # for d_idx in range(50):
    #    ...
    
    # Actually, simpler:
    # For each direction, iterate through all points.
    # If the point is the "start" of a line (e.g., smallest index in the line), create the line.
    
    # To find the start: 
    # For v=(1, m), the line is {p + t*v}.
    # We can just pick the point with the smallest index in the orbit.
    
    # Precompute point indices for fast access
    # We already have `points` list.
    
    # Let's build `point_lines` and `line_points`.
    
    # Initialize
    point_lines = [[] for _ in range(num_points)]
    line_points = []
    
    # Map from (dir_idx, start_point_idx) to line_id
    line_map = {}
    
    for d_idx in range(50):
        if d_idx < 49:
            # v = (1, m) where m = field_elements[d_idx]
            m = field_elements[d_idx]
            v = ((1, 0), m)
        else:
            # v = (0, 1)
            v = ((0, 0), (0, 1))
            
        # For each point, find the start of the line
        # start is the point p such that p - t*v has no valid t?
        # Actually, just find the min index in the orbit.
        
        # To do this efficiently, we can just iterate and group.
        # But q=49 is small.
        
        # Let's just generate lines by picking a representative.
        # For d < 49 (v=(1,m)):
        # The line is determined by Y - mX.
        # We can compute the invariant for each point.
        # But we need to map invariant to a specific line ID.
        
        # Let's just use the set of points.
        # visited = [False] * num_points
        # for p in range(num_points):
        #    if not visited[p]:
        #       create line
        #       walk the line, mark visited
        
        visited = [False] * num_points
        for p in range(num_points):
            if not visited[p]:
                # Start new line
                line_idx = len(line_points)
                line_points.append([])
                
                # Walk the line
                cur = p
                while not visited[cur]:
                    visited[cur] = True
                    line_points[line_idx].append(cur)
                    point_lines[cur].append(line_idx)
                    # Next point: cur + v
                    c_coords = point_coords[cur]
                    x_c, y_c = c_coords
                    nx = add(x_c, v[0])
                    ny = add(y_c, v[1])
                    # Find index of (nx, ny)
                    # We need a fast map from coords to index
                    # coords: ((xa, xb), (ya, yb))
                    # index = (xa*p + xb)*q + (ya*p + yb)
                    n_idx = (nx[0]*p + nx[1])*q + (ny[0]*p + ny[1])
                    cur = n_idx
        
    # Now we have line_points and point_lines.
    # line_count array
    line_count = np.zeros(len(line_points), dtype=np.int64)
    line_single_point = np.full(len(line_points), -1, dtype=np.int64)
    
    # Initialize S from seed
    # Seed: S = U \ B
    # U: 2*xa + ya^2 - yb^2*k == 0
    # B: ya is square mod p
    # y_special: ya in {0,1,2,4} (squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1)
    # Squares: {0, 1, 2, 4}
    
    S = np.zeros(num_points, dtype=bool)
    for idx in range(num_points):
        xa, xb, ya, yb = points[idx][0][0], points[idx][0][1], points[idx][1][0], points[idx][1][1]
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = ya in {0, 1, 2, 4}
        if in_unital and not in_y_special:
            S[idx] = True
            
    # Update line counts and good lines
    s_good = np.zeros(num_points, dtype=np.int64)
    n_good = np.zeros(num_points, dtype=np.int64)
    
    for idx in range(num_points):
        if S[idx]:
            for L in point_lines[idx]:
                if line_count[L] == 0:
                    line_count[L] = 1
                    line_single_point[L] = idx
                elif line_count[L] == 1:
                    line_count[L] = 2
                    line_single_point[L] = -1
                else:
                    line_count[L] += 1
        else:
            # N point
            pass
            
    # Calculate initial good lines
    for L in range(len(line_points)):
        c = line_count[L]
        if c == 0:
            # All points on L are in N
            for p_idx in line_points[L]:
                n_good[p_idx] += 1
        elif c == 1:
            # One point in S
            p_idx = line_single_point[L]
            s_good[p_idx] += 1
            # Other points in N?
            for p_idx2 in line_points[L]:
                if p_idx2 != p_idx:
                    # This point is in N, but L has count 1, so it's not good for N (needs 0)
                    pass
        else:
            # c >= 2
            pass
            
    # Check validity
    def is_valid():
        for idx in range(num_points):
            if S[idx]:
                if s_good[idx] == 0:
                    return False
            else:
                if n_good[idx] == 0:
                    return False
        return True
        
    if not is_valid():
        # Fallback to empty S? Or just return seed N?
        # If seed is invalid, return seed N anyway?
        # The seed is supposed to be valid.
        pass
        
    # Greedy addition
    start_time = time.time()
    limit = 600.0
    
    while time.time() - start_time < limit:
        # Find candidates in N
        candidates = np.where(~S)[0]
        if len(candidates) == 0:
            break
            
        # Pick random candidate
        x_new = candidates[random.randint(0, len(candidates)-1)]
        
        # Tentative addition
        # Save state
        old_line_counts = line_count.copy()
        old_line_single = line_single_point.copy()
        old_s_good = s_good.copy()
        old_n_good = n_good.copy()
        
        # Add x_new
        S[x_new] = True
        valid = True
        
        for L in point_lines[x_new]:
            c = line_count[L]
            if c == 0:
                # Becomes 1
                line_count[L] = 1
                line_single_point[L] = x_new
                
                # Update n_good for all points on L (except x_new which is now S)
                for p_idx in line_points[L]:
                    if p_idx != x_new:
                        # These points were in N (since c was 0)
                        # They lose this good line
                        n_good[p_idx] -= 1
                        if n_good[p_idx] == 0:
                            # Check if this point is still valid
                            # It is in N, so needs n_good > 0
                            valid = False
                            break
                if not valid:
                    break
                # x_new gains s_good
                s_good[x_new] += 1
                
            elif c == 1:
                # Becomes 2
                y = line_single_point[L]
                line_count[L] = 2
                line_single_point[L] = -1
                
                # y loses s_good
                s_good[y] -= 1
                if s_good[y] == 0:
                    valid = False
                    break
                # x_new does not gain s_good
            else:
                # c >= 2
                line_count[L] += 1
                # No change in good lines
                
        if valid:
            # Check x_new itself
            if s_good[x_new] == 0:
                valid = False
                
        if valid:
            # Accept
            continue
        else:
            # Revert
            S[x_new] = False
            line_count[:] = old_line_counts
            line_single_point[:] = old_line_single
            s_good[:] = old_s_good
            n_good[:] = old_n_good
            
    # Construct N
    N = []
    for idx in range(num_points):
        if not S[idx]:
            N.append(points[idx])
            
    return np.array(N, dtype=np.int64)
