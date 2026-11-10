# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3, smallest QR mod 7
    q = p * p  # 49
    total = q * q  # 2401
    deadline = time.time() + 570  # safety margin before 630s

    # Precompute all lines: for each point, list of (line_id, other_points)
    # Line direction (1, m) for m in F_49, or (0, 1)
    # Represent field elements as ints 0..48 where i = i//p + (i%p)*alpha
    
    # Field operations
    def add_f(a, b):
        return ((a // p + b // p) % p, (a % p + b % p) % p)
    
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + a1*b1*3 + (a0*b1 + a1*b0)*alpha
        a0, a1 = a // p, a % p
        b0, b1 = b // p, b % p
        r0 = (a0 * b0 + a1 * b1 * k) % p
        r1 = (a0 * b1 + a1 * b0) % p
        return r0 * p + r1
    
    def to_idx(pt):
        # pt = ((xa, xb), (ya, yb)) -> int 0..2400
        xa, xb, ya, yb = pt[0][0], pt[0][1], pt[1][0], pt[1][1]
        x = xa * p + xb
        y = ya * p + yb
        return x * q + y
    
    # Build lines: for each point, which lines contain it
    # Line (1, m): points {x + t*(1, m) : t in F_49} for fixed x0
    # Actually line through point with direction (1,m): {(x + t, y + t*m) : t in F_49}
    # Line (0,1): {(x, y + t) : t in F_49}
    
    # For each point, compute all lines through it
    # A line is determined by: point + direction
    # For direction (1, m): line contains points (x + t, y + t*m) for t in F_49
    # For direction (0, 1): line contains points (x, y + t) for t in F_49
    
    # Precompute: for each point, list of (direction_type, m) where line through point
    # in that direction has all other points in N
    
    # Actually, let's think differently. For a point to be removable, there must exist
    # a line L through it such that L \ {point} ⊆ N.
    
    # Lines through a point (x, y):
    # - Vertical: {(x, y+t) : t in F_49}
    # - Slope m: {(x+t, y+t*m) : t in F_49} for each m in F_49
    
    # Total 50 lines through each point (49 slopes + 1 vertical)
    
    # For efficiency, precompute for each point and each line direction,
    # the set of other points on that line.
    
    # Let's index points as 0..2400
    # point_idx = x * q + y where x, y in 0..48
    
    # For point idx, x = idx // q, y = idx % q
    # Line with slope m through (x,y): points (x+t, y+t*m) mod field, t in 0..48
    #   other points: t = 1..48
    # Vertical line through (x,y): points (x, y+t), t in 0..48
    #   other points: t = 1..48
    
    # Precompute line membership
    # lines_through[i] = list of frozensets of other points for each line through i
    # This is 2401 * 50 * 48 = ~5.7M entries, might be heavy but let's try
    
    # Actually, let's be smarter. For greedy removal, we need to check:
    # Can point i be removed? Is there a line L through i with L\{i} ⊆ current_N?
    
    # Let's precompute for each point i, for each of its 50 lines, the list of other 48 points.
    
    # Memory: 2401 * 50 * 48 ints = ~5.7M * 4 bytes = ~23 MB. OK.
    
    # But building this takes time. Let's compute on the fly or use a compact representation.
    
    # Alternative: for each point i, store for each line the count of other points still in N.
    # When we remove a point, we decrement counts for all lines through that point.
    
    # Let's precompute:
    # point_lines[i] = list of (line_key, other_points_list) for each of 50 lines through i
    # line_key identifies the line uniquely
    
    # Actually, simpler: just store for each point i, a list of 50 arrays of 48 point indices.
    
    # Let me build this efficiently.
    
    # For point (x, y) with x = i // q, y = i % q:
    # Vertical line: other points are (x, y+t) for t=1..48
    #   point index: x * q + (y + t) % q ... but field addition!
    
    # Wait, y + t is field addition, not integer addition mod q.
    # y = ya * p + yb, t = ta * p + tb
    # y + t = ((ya+ta) % p) * p + ((yb+tb) % p)
    
    # Let me precompute field addition table: add_table[a][b] = a+b in field
    add_table = np.zeros((q, q), dtype=np.int32)
    for a in range(q):
        a0, a1 = a // p, a % p
        for b in range(q):
            b0, b1 = b // p, b % p
            add_table[a, b] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
    
    mul_table = np.zeros((q, q), dtype=np.int32)
    for a in range(q):
        a0, a1 = a // p, a % p
        for b in range(q):
            b0, b1 = b // p, b % p
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_table[a, b] = r0 * p + r1
    
    # Now for point i with x = i // q, y = i % q:
    # Vertical line other points: (x, add_table[y, t]) for t = 1..48
    #   point index: x * q + add_table[y, t]
    # Slope m line other points: (add_table[x, t], add_table[y, mul_table[m, t]]) for t = 1..48
    #   point index: add_table[x, t] * q + add_table[y, mul_table[m, t]]
    
    # Precompute all_line_others: shape (2401, 50, 48) int32
    # Line 0: vertical, lines 1-49: slopes m=0..48
    all_line_others = np.zeros((total, 50, q-1), dtype=np.int32)
    
    for i in range(total):
        x = i // q
        y = i % q
        # Vertical: line 0
        for t in range(1, q):
            all_line_others[i, 0, t-1] = x * q + add_table[y, t]
        # Slopes: lines 1-49
        for m in range(q):
            for t in range(1, q):
                nx = add_table[x, t]
                ny = add_table[y, mul_table[m, t]]
                all_line_others[i, m+1, t-1] = nx * q + ny
    
    # Now greedy removal with random restarts
    best_complement = 0
    best_set = None
    
    rng = np.random.default_rng(42)
    
    num_restarts = 200  # will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random permutation of points
        order = rng.permutation(total)
        
        # in_N: boolean array, initially all True
        in_N = np.ones(total, dtype=bool)
        
        # line_counts[i][l] = number of other points on line l through i that are in N
        # Initialize: for each point i, for each line l, count = 48 (all others in N)
        # But we need to track per-line counts for each point.
        # Actually, when we remove point j, for each point i on the same line as j,
        # we decrement the count. But that's expensive.
        
        # Better: for each point i, for each line l through i, maintain count.
        # When removing point j, for each line l' through j, and for each point i on line l',
        # if i != j, decrement count[i][l_of_i_containing_j].
        
        # This is complex. Alternative: just check on the fly.
        
        # For point i to be removable: exists line l through i such that all 48 others are in N.
        # Check: for each of 50 lines, check if all 48 others are in N.
        # That's 50 * 48 = 2400 checks per point. For 2401 points, ~5.7M checks per pass.
        # With numpy, we can vectorize.
        
        # Let's do it iteratively, trying to remove points in random order.
        # For each point in order, check if removable, if so remove it.
        
        # To speed up: maintain for each point i, for each line l, the count of others in N.
        # line_count[i, l] = sum over others on line l of in_N[other]
        
        # Initialize
        line_count = np.full((total, 50), q-1, dtype=np.int32)
        
        # When we remove point j, for each line l through j:
        #   for each other point i on line l:
        #     line_count[i, l_corresponding] -= 1
        # where l_corresponding is the line index for i that corresponds to line l.
        
        # This requires knowing, for line l through j, what line index it has for each other point i.
        # This is complicated. Let's just do the naive check with numpy vectorization.
        
        removed_count = 0
        for idx in order:
            if time.time() > deadline - 15:
                break
            if not in_N[idx]:
                continue
            # Check if any line through idx has all others in N
            # all_line_others[idx] has shape (50, 48)
            others = all_line_others[idx]  # (50, 48)
            # Check: for each line, are all 48 others in N?
            # in_N[others] has shape (50, 48), check if any row is all True
            all_in = np.all(in_N[others], axis=1)  # (50,)
            if np.any(all_in):
                # Remove idx
                in_N[idx] = False
                removed_count += 1
        
        complement = removed_count
        if complement > best_complement:
            best_complement = complement
            best_set = in_N.copy()
    
    # Also try the seed construction as a baseline
    # Build seed set
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    seed_in_N = np.ones(total, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = xa * p + xb
                    y = ya * p + yb
                    idx = x * q + y
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        seed_in_N[idx] = False
    
    seed_complement = np.sum(~seed_in_N)
    if seed_complement > best_complement:
        best_complement = seed_complement
        best_set = seed_in_N.copy()
    
    # Build result
    points = []
    for i in range(total):
        if best_set[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
