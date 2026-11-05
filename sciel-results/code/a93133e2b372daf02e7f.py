import time

def construct():
    N = 128
    start_time = time.time()
    time_limit = 85.0  # Safety margin

    def is_valid(pos, c, colors, k):
        # Check Constraint 1: pos is the 3rd term
        # a = pos - 2d
        for d in range(1, (pos - 1) // 2 + 1):
            a = pos - 2 * d
            if colors[a] == colors[a + d] and colors[a] == c:
                return False
        
        # Check Constraint 2: pos is the 4th term
        # a = pos - 3d
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            # distinct colors among a, a+d, a+2d, c
            # We can optimize this check
            c1 = colors[a]
            c2 = colors[a + d]
            c3 = colors[a + 2 * d]
            
            # If c1, c2, c3 are all same, say u:
            # Set is {u, c}. Size is 1 or 2. Always < 3.
            # So this case is always invalid.
            if c1 == c2 and c2 == c3:
                return False
            
            # If c1, c2, c3 are not all same:
            # We need {c1, c2, c3, c} to have size >= 3.
            # This fails only if {c1, c2, c3, c} has size 1 or 2.
            # Size 1: c1=c2=c3=c. But we already handled c1=c2=c3.
            #        If c1=c2=c3=u, and c=u, size 1.
            #        If c1=c2=c3=u, and c!=u, size 2.
            #        So if c1=c2=c3, it's always invalid.
            
            # Size 2:
            # The set {c1, c2, c3} has size 2. Let the values be u, v.
            # Then c must be u or v to keep size 2.
            # If c is neither u nor v, size becomes 3.
            # So we fail if c is in {c1, c2, c3}.
            
            # Let's just compute the set size, it's small.
            if len({c1, c2, c3, c}) < 3:
                return False

        return True

    # Try k=3, 4, ...
    for k in range(3, 15):
        colors = [None] * (N + 1)
        node_count = 0
        max_nodes = 1000000  # Abort if search space is too big for this k
        
        def solve(pos):
            nonlocal node_count
            if time.time() - start_time > time_limit:
                return False
            if node_count > max_nodes:
                return False
            
            if pos > N:
                return True
            
            node_count += 1
            
            # Heuristic: try colors in random order?
            # Or just 0..k-1
            for c in range(k):
                if is_valid(pos, c, colors, k):
                    colors[pos] = c
                    if solve(pos + 1):
                        return True
                    colors[pos] = None
            
            return False

        if solve(1):
            return [colors[i] for i in range(1, N + 1)]
            
    # Fallback: Greedy with large k
    colors = [None] * (N + 1)
    for pos in range(1, N + 1):
        for c in range(128):
            if is_valid(pos, c, colors, 128):
                colors[pos] = c
                break
    return [colors[i] for i in range(1, N + 1)]
