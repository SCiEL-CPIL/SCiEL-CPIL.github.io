import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Leave 5s margin for return/verification overhead
    MARGIN = 5.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    def verify(n, colors):
        """Verify no C4 in any color class."""
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[c][u] & adj[c][v]).bit_count() >= 2:
                        return False
        return True

    def extract(color, n):
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                out.append(color[i][j])
        return out

    def lazyebnik_woldar_25():
        """
        Construct a valid 5-coloring of K_25 using AG(2,5).
        Vertices are (x, y) in GF(5)^2.
        Edges are colored by the slope of the line connecting them.
        Slopes: 0, 1, 2, 3, infinity.
        If dx == 0, slope is infinity (color 4).
        Else slope = dy * dx^(-1) mod 5.
        """
        n = 25
        color = [[0] * n for _ in range(n)]
        # Precompute modular inverses for 1..4 mod 5
        # 1^-1=1, 2^-1=3, 3^-1=2, 4^-1=4
        inv = [0, 1, 3, 2, 4]
        
        for i in range(n):
            x1, y1 = i % 5, i // 5
            for j in range(i + 1, n):
                x2, y2 = j % 5, j // 5
                dx = (x2 - x1) % 5
                dy = (y2 - y1) % 5
                if dx == 0:
                    c = 4
                else:
                    c = (dy * inv[dx]) % 5
                color[i][j] = c
                color[j][i] = c
        return extract(color, n)

    def fast_local_search(n, init_colors, budget_time):
        """
        Fast local search to fix violations or extend coloring.
        Uses bitmasks for adjacency and common neighbor counts.
        """
        if time_left() < MARGIN:
            return None

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total_pairs = len(pairs)
        
        # Initialize state
        color = [[0] * n for _ in range(n)]
        idx = 0
        for (i, j) in pairs:
            color[i][j] = init_colors[idx]
            color[j][i] = init_colors[idx]
            idx += 1

        adj = [[0] * n for _ in range(5)]
        for (i, j) in pairs:
            c = color[i][j]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)

        # common[c][u][v] = number of common neighbors of u and v in color c
        # We only care if it's >= 2 (violation) or == 1 (risky)
        # To save memory/time, we can just recompute delta on the fly or maintain counts.
        # Given n<=28, n^2 * 5 is small (28^2*5 = 3920). Let's maintain counts.
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        viol = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cc = (adj[c][u] & adj[c][v]).bit_count()
                    common[c][u][v] = cc
                    common[c][v][u] = cc
                    if cc >= 2:
                        viol += 1

        if viol == 0:
            return extract(color, n)

        T = 1.0
        T_end = 0.01
        t_start = time.time()
        
        # Precompute list of vertices for fast iteration
        vertices = list(range(n))

        while time_left() > MARGIN and (time.time() - t_start) < budget_time:
            # Pick a random edge
            a, b = pairs[random.randrange(total_pairs)]
            c1 = color[a][b]
            c2 = random.randrange(5)
            if c2 == c1:
                continue

            # Calculate delta in violations
            delta = 0
            
            # Remove edge (a,b) from color c1
            Na = adj[c1][a]
            Nb = adj[c1][b]
            
            # Neighbors of a in c1 (excluding b)
            x = Nb & ~(1 << a)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c1][a][xi]
                new = old - 1
                common[c1][a][xi] = new
                common[c1][xi][a] = new
                if old == 2:
                    delta -= 1
                # if old > 2, delta doesn't change (still violation)
                # if old < 2, no change in violation count

            # Neighbors of b in c1 (excluding a)
            x = Na & ~(1 << b)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c1][b][xi]
                new = old - 1
                common[c1][b][xi] = new
                common[c1][xi][b] = new
                if old == 2:
                    delta -= 1

            adj[c1][a] &= ~(1 << b)
            adj[c1][b] &= ~(1 << a)

            # Add edge (a,b) to color c2
            Na = adj[c2][a]
            Nb = adj[c2][b]

            # Neighbors of a in c2 (excluding b)
            x = Nb & ~(1 << a)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c2][a][xi]
                new = old + 1
                common[c2][a][xi] = new
                common[c2][xi][a] = new
                if old == 1:
                    delta += 1

            # Neighbors of b in c2 (excluding a)
            x = Na & ~(1 << b)
            while x:
                lsb = x & (-x)
                xi = lsb.bit_length() - 1
                x ^= lsb
                old = common[c2][b][xi]
                new = old + 1
                common[c2][b][xi] = new
                common[c2][xi][b] = new
                if old == 1:
                    delta += 1

            adj[c2][a] |= (1 << b)
            adj[c2][b] |= (1 << a)

            viol += delta

            # Accept if improves or random
            if delta <= 0:
                color[a][b] = c2
                color[b][a] = c2
            else:
                # Simulated annealing acceptance
                # Use a simple acceptance probability
                if random.random() < (2.71828 ** (-delta / T)):
                    color[a][b] = c2
                    color[b][a] = c2
                else:
                    # Revert
                    # Remove from c2
                    Na = adj[c2][a]
                    Nb = adj[c2][b]
                    x = Nb & ~(1 << a)
                    while x:
                        lsb = x & (-x)
                        xi = lsb.bit_length() - 1
                        x ^= lsb
                        old = common[c2][a][xi]
                        new = old - 1
                        common[c2][a][xi] = new
                        common[c2][xi][a] = new
                        if old == 2:
                            viol -= 1
                    x = Na & ~(1 << b)
                    while x:
                        lsb = x & (-x)
                        xi = lsb.bit_length() - 1
                        x ^= lsb
                        old = common[c2][b][xi]
                        new = old - 1
                        common[c2][b][xi] = new
                        common[c2][xi][b] = new
                        if old == 2:
                            viol -= 1
                    adj[c2][a] &= ~(1 << b)
                    adj[c2][b] &= ~(1 << a)
                    
                    # Add back to c1
                    Na = adj[c1][a]
                    Nb = adj[c1][b]
                    x = Nb & ~(1 << a)
                    while x:
                        lsb = x & (-x)
                        xi = lsb.bit_length() - 1
                        x ^= lsb
                        old = common[c1][a][xi]
                        new = old + 1
                        common[c1][a][xi] = new
                        common[c1][xi][a] = new
                        if old == 1:
                            viol += 1
                    x = Na & ~(1 << b)
                    while x:
                        lsb = x & (-x)
                        xi = lsb.bit_length() - 1
                        x ^= lsb
                        old = common[c1][b][xi]
                        new = old + 1
                        common[c1][b][xi] = new
                        common[c1][xi][b] = new
                        if old == 1:
                            viol += 1
                    adj[c1][a] |= (1 << b)
                    adj[c1][b] |= (1 << a)

            # Cool down
            T = max(T_end, T * 0.999)

            if viol == 0:
                return extract(color, n)
        
        return None

    # 1. Generate the guaranteed n=25 solution
    colors_25 = lazyebnik_woldar_25()
    best_n = 25
    best_colors = colors_25
    
    # Verify n=25 just in case (should be instant)
    if not verify(25, colors_25):
        # Fallback to simple cyclic if algebraic fails (unlikely)
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    # 2. Try to extend to n=26, 27, 28
    # Strategy: Start with the n=25 coloring on the first 25 vertices.
    # Assign random colors to edges involving the new vertex.
    # Run local search to fix violations.
    
    for target_n in [26, 27, 28]:
        if time_left() < MARGIN + 10:
            break
            
        found = False
        # Try a few times with different random initializations for the new edges
        for attempt in range(5):
            if time_left() < MARGIN + 5:
                break
            
            # Construct initial coloring for target_n
            init_colors = [0] * (target_n * (target_n - 1) // 2)
            
            # Copy first 25 vertices' colors
            idx = 0
            for i in range(25):
                for j in range(i + 1, 25):
                    init_colors[idx] = colors_25[idx]
                    idx += 1
            
            # Randomly color edges involving vertices 25..target_n-1
            for i in range(25, target_n):
                for j in range(i + 1, target_n):
                    init_colors[idx] = random.randrange(5)
                    idx += 1
            
            # Edges between old (0-24) and new (25+)
            # Note: The lexicographic order is (0,1)..(0,24), (0,25).., (1,2)..(1,24), (1,25)..
            # My simple copy above only handled i<25, j<25.
            # I need to properly map the indices.
            
            # Let's rebuild init_colors properly to avoid index errors
            init_colors = [0] * (target_n * (target_n - 1) // 2)
            idx = 0
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if i < 25 and j < 25:
                        # Map to 25-coloring index
                        # Index in 25-coloring: i*(25-1) - i*(i-1)//2 + (j-i-1)
                        # Simpler: just use the list we made if we made it in the same order.
                        # Let's just re-generate the 25 colors in a flat list in lex order.
                        pass # We'll do this differently below for safety
            
            # Safer approach:
            flat_25 = lazyebnik_woldar_25()
            idx25 = 0
            idx_full = 0
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if i < 25 and j < 25:
                        init_colors[idx_full] = flat_25[idx25]
                        idx25 += 1
                    else:
                        init_colors[idx_full] = random.randrange(5)
                    idx_full += 1
            
            # Run local search with a time budget
            # Give it 20 seconds per attempt
            result = fast_local_search(target_n, init_colors, 20.0)
            if result is not None:
                best_n = target_n
                best_colors = result
                found = True
                break
        
        if found:
            continue # Try next larger n
        else:
            # If we failed to extend to this n, we might still try the next one
            # but it's less likely. We'll break if we run out of time.
            pass

    return {'n': best_n, 'colors': best_colors}
