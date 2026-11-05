import time
import random

DEADLINE = 110  # seconds, safety margin before 120s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    # ---- Build the n=32 seed colouring (Greenwood-Gleason doubled) ----
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

    n_seed = 32
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if i // 16 == j // 16:
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_seed.append(3)

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        # Efficient check using bitmasks
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        for c in range(4):
            edges = adj[c]
            if not edges:
                continue
            # Build adjacency list for this color
            adj_list = {i: set() for i in range(n)}
            for (i, j) in edges:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        # Check if i-j-k is a triangle in color c
                        # Common neighbors of i and j in color c
                        common = adj_list[i] & adj_list[j]
                        if common:
                            return False
        return True

    if not _validate(colors_seed, n_seed):
        return {'n': 1, 'colors': []}

    best_n = n_seed
    best_colors = colors_seed[:]

    # Attempt to extend to n=33
    # New vertex index 32. Edges (32, i) for i in 0..31.
    # Let x[i] be color of (32, i).
    # Constraint: for any i < j, if x[i] == x[j] == c, then colors_seed[edge(i,j)] != c.
    
    def try_extend_to_33():
        if not _time_left():
            return None
        
        # Precompute constraints for each color c and pair (i,j)
        # If x[i]=c and x[j]=c, then seed_color(i,j) must not be c.
        # This is equivalent to: for each color c, the set of indices i where x[i]=c must be an independent set in the graph G_c,
        # where G_c has edges (i,j) if seed_color(i,j) == c.
        
        # Graphs G_c are defined by colors_seed.
        # We need to partition {0..31} into 4 sets S_0, S_1, S_2, S_3 such that
        # S_c is an independent set in G_c.
        # This is a 4-coloring problem of a specific type? No, it's a constraint satisfaction.
        
        # Heuristic: Randomized Local Search
        # Start with a random assignment x[0..31].
        # Cost = number of violated constraints.
        # Violation for pair (i,j) with x[i]=x[j]=c and seed_color(i,j)=c.
        
        # To speed up, we can just try a few random restarts with a simple hill-climb.
        # Given the tight time, maybe just a few random tries?
        # Actually, for n=32, the graphs G_c are quite dense/structured.
        
        # Let's try a backtracking with forward checking, but limited depth/time.
        # Or simply: return None if we don't find it quickly.
        
        # A simpler approach: Try to find a valid assignment for x[0..31].
        # We can use a SAT-like solver logic manually.
        
        # Let's use a simple randomized greedy with restarts.
        for restart in range(1000):
            if not _time_left():
                return None
            
            x = [random.randint(0, 3) for _ in range(32)]
            
            # Check validity
            def is_valid(x):
                for i in range(32):
                    for j in range(i + 1, 32):
                        if x[i] == x[j]:
                            c = x[i]
                            # Get seed color for (i,j)
                            # Index in colors_seed
                            # i, j < 32, so they are in the seed part.
                            # Index calculation for (i,j) in lexicographic order of 0..31
                            # idx = i * 32 - i*(i+1)//2 + (j - i - 1)
                            # Simpler: precompute a lookup table for seed colors
                            pass
                return True
            # This nested loop is O(N^2) = 1024 checks. 1000 restarts = 1M checks. Very fast.
            
            # Precompute seed color lookup for speed
            # seed_color[i][j]
            seed_lookup = [[-1]*32 for _ in range(32)]
            idx = 0
            for i in range(32):
                for j in range(i + 1, 32):
                    seed_lookup[i][j] = colors_seed[idx]
                    seed_lookup[j][i] = colors_seed[idx]
                    idx += 1
            
            def check_x(x):
                for i in range(32):
                    for j in range(i + 1, 32):
                        if x[i] == x[j]:
                            if seed_lookup[i][j] == x[i]:
                                return False
                return True
            
            if check_x(x):
                # Found a valid extension!
                new_colors = colors_seed[:]
                for i in range(32):
                    new_colors.append(x[i])
                return new_colors
            
            # Simple hill climb: flip a random x[i] if it reduces violations
            # But random restarts might be enough for small N?
            # Let's add a quick local search.
            current_violations = 0
            # Count violations
            for i in range(32):
                for j in range(i + 1, 32):
                    if x[i] == x[j] and seed_lookup[i][j] == x[i]:
                        current_violations += 1
            
            for step in range(100):
                if not _time_left():
                    return None
                if current_violations == 0:
                    new_colors = colors_seed[:]
                    for i in range(32):
                        new_colors.append(x[i])
                    return new_colors
                    
                # Pick a random index to flip
                i = random.randint(0, 31)
                old_c = x[i]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                # Calculate change in violations
                delta = 0
                for j in range(32):
                    if j == i:
                        continue
                    # Violation involving (i, j)
                    # Old: x[i]=old_c, x[j]=x[j]
                    if old_c == x[j] and seed_lookup[i][j] == old_c:
                        delta -= 1 # Remove violation
                    # New: x[i]=new_c, x[j]=x[j]
                    if new_c == x[j] and seed_lookup[i][j] == new_c:
                        delta += 1 # Add violation
                
                if delta <= 0:
                    x[i] = new_c
                    current_violations += delta
                else:
                    # Accept with small probability (Simulated Annealing lite)
                    if random.random() < 0.1:
                        x[i] = new_c
                        current_violations += delta
                    else:
                        # Try another index or restart
                        pass
            
    ext_colors = try_extend_to_33()
    if ext_colors:
        if _validate(ext_colors, 33):
            return {'n': 33, 'colors': ext_colors}
    
    return {'n': best_n, 'colors': best_colors}
