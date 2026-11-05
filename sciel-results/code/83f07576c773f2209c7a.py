"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _try_construct_50(deadline):
    """
    Attempts to construct a 4-coloring of K50 by finding a sum-free partition
    of Z5 x Z5 (excluding 0) into 4 sets, then doubling.
    """
    # Elements of Z5 x Z5
    elements = [(i, j) for i in range(5) for j in range(5)]
    zero = (0, 0)
    non_zero = [e for e in elements if e != zero]
    
    # Map elements to indices 0..23
    elem_to_idx = {e: i for i, e in enumerate(non_zero)}
    
    # Precompute negation and sum
    # neg(a, b) = (-a, -b) mod 5
    # sum(a, b) = (a1+b1, a2+b2) mod 5
    
    def neg(e):
        return ((-e[0]) % 5, (-e[1]) % 5)
        
    def add(e1, e2):
        return ((e1[0] + e2[0]) % 5, (e1[1] + e2[1]) % 5)

    # We want to assign each non_zero element to one of 4 colors (0,1,2,3)
    # such that for each color c, no triple (a,b,c) in S_c sums to 0.
    # Condition: For any a, b in S_c (a != b), add(a, b) should not be in S_c?
    # Wait, a+b+c=0 => c = -add(a,b).
    # So if a, b in S_c, then neg(add(a,b)) must NOT be in S_c.
    
    # Initialize with random assignment
    rng = random.Random(42)
    assignment = [rng.randint(0, 3) for _ in range(24)]
    
    def count_violations(assign):
        """Count the number of zero-sum triples in each color class."""
        # Group elements by color
        groups = [[] for _ in range(4)]
        for idx, c in enumerate(assign):
            groups[c].append(non_zero[idx])
        
        violations = 0
        for c in range(4):
            S = groups[c]
            S_set = set(S)
            # Check all pairs a, b in S
            # If a == b, check if -2a is in S
            # If a != b, check if -(a+b) is in S
            # To avoid double counting, we can just check all ordered pairs or unordered?
            # A triangle is defined by 3 distinct vertices.
            # In the difference set, the vectors are distinct?
            # If a=b, then the vectors are a, a, -2a.
            # If a != b, vectors are a, b, -(a+b).
            # We just need to ensure no such triple exists in S.
            
            # Efficient check:
            # For each a in S:
            #   if -2a in S: violation (type 1)
            # For each pair a, b in S with a < b (lexicographically or index):
            #   if -(a+b) in S: violation (type 2)
            
            for a in S:
                if neg(add(a, a)) in S_set:
                    violations += 1
            
            # Unordered pairs
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    a = S[i]
                    b = S[j]
                    if neg(add(a, b)) in S_set:
                        violations += 1
        return violations

    best_assign = assignment[:]
    best_violations = count_violations(best_assign)
    
    # Local Search
    start_time = time.time()
    while time.time() < deadline - 10.0:
        if best_violations == 0:
            break
        
        # Pick a random element to reassign
        idx = rng.randint(0, 23)
        old_color = best_assign[idx]
        
        # Try all other colors
        best_new_color = old_color
        min_violations = best_violations
        
        for new_color in range(4):
            if new_color == old_color:
                continue
            # Tentatively assign
            best_assign[idx] = new_color
            v = count_violations(best_assign)
            if v < min_violations:
                min_violations = v
                best_new_color = new_color
        
        # Commit the best move
        best_assign[idx] = best_new_color
        best_violations = min_violations
        
        # If stuck for a long time, random restart?
        # For now, simple hill climbing.
        
    if best_violations == 0:
        # Construct K25 coloring
        # Colors for K25 edges
        # Vertex i corresponds to non_zero[i] if i < 24, and zero is vertex 24?
        # Actually, vertices are 0..24.
        # Let's map vertex v to element elem_to_idx?
        # Better: vertex v is element elements[v].
        # Color of edge (u, v) is determined by u - v.
        
        k25_colors = []
        for u in range(25):
            for v in range(u + 1, 25):
                eu = elements[u]
                ev = elements[v]
                diff = add(eu, neg(ev)) # u - v
                if diff == zero:
                    # Should not happen for u != v
                    c = 0
                else:
                    idx = elem_to_idx[diff]
                    c = best_assign[idx]
                k25_colors.append(c)
        
        # Double to K50
        n = 50
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i < 25 and j < 25:
                    # First copy
                    # Edge (i, j) in K25
                    # Index in k25_colors:
                    # Row i has 25-1-i edges? No, 25 vertices.
                    # Edges for row i: (i, i+1) ... (i, 24). Count: 24-i.
                    # Total edges before row i: sum_{k=0}^{i-1} (24-k)
                    # We can just re-iterate
                    pass
        
        # Re-iterate to build K50 colors correctly
        colors = []
        for i in range(50):
            for j in range(i + 1, 50):
                if i < 25 and j < 25:
                    # Both in first copy
                    # Find index in k25_colors
                    # k25_colors is built for vertices 0..24
                    # We need the color of edge (i, j) in K25
                    # We can just look it up by rebuilding the index or storing in a dict
                    # Let's store K25 colors in a 2D list for easy lookup
                    pass
        
        # Let's rebuild K25 lookup
        C25 = [[-1]*25 for _ in range(25)]
        idx = 0
        for u in range(25):
            for v in range(u + 1, 25):
                c = k25_colors[idx]
                C25[u][v] = c
                C25[v][u] = c
                idx += 1
        
        for i in range(50):
            for j in range(i + 1, 50):
                if i < 25 and j < 25:
                    colors.append(C25[i][j])
                elif i >= 25 and j >= 25:
                    # Second copy
                    i2 = i - 25
                    j2 = j - 25
                    colors.append(C25[i2][j2])
                else:
                    # Cross edge
                    colors.append(3)
        
        return {'n': 50, 'colors': colors}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try to find n=50
    try:
        res = _try_construct_50(deadline)
        if res:
            return res
    except Exception:
        pass
    
    # Fallback to n=32
    return _baseline32()

# EVOLVE-BLOCK-END
