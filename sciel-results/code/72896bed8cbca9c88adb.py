"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Build the 32-vertex base coloring (Greenwood-Gleason: two copies of 16-vertex Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors(n=32):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def color_index(n, i, j):
        """Get index in colors array for edge (i,j), i<j in lexicographic order."""
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    # Start with 32-vertex base
    n = 32
    colors = base_colors(n)
    best_n = n
    best_colors = colors[:]

    def build_adj(colors, n):
        """Build adjacency: adj[i][c] = set of vertices j such that edge (i,j) has color c."""
        adj = [[set() for _ in range(4)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[color_index(n, i, j)]
                adj[i][c].add(j)
                adj[j][c].add(i)
        return adj

    def try_add_vertex(colors, n, adj, deadline, max_restarts=200):
        """
        Try to add one new vertex (vertex n) to the coloring.
        For the new vertex, assign colors to edges (n, i) for i in 0..n-1.
        Constraint: for each color c, S_c = {i : color(n,i) = c} must be
        an independent set in the color-c subgraph.
        
        Uses randomized backtracking with constraint propagation.
        Returns (success, assignment) or (False, None).
        """
        if time.time() > deadline:
            return False, None

        for restart in range(max_restarts):
            if time.time() > deadline:
                return False, None
            
            assignment = [-1] * n
            
            # Choose vertex ordering
            if restart == 0:
                order = list(range(n))
            elif restart == 1:
                order = list(range(n))
                random.shuffle(order)
            elif restart % 3 == 0:
                order = sorted(range(n), key=lambda i: sum(len(adj[i][c]) for c in range(4)), reverse=True)
            elif restart % 3 == 1:
                order = sorted(range(n), key=lambda i: sum(len(adj[i][c]) for c in range(4)))
            else:
                order = list(range(n))
                random.shuffle(order)
            
            # Forward-checking backtracking
            def backtrack(idx):
                nonlocal assignment
                if time.time() > deadline:
                    return False
                if idx == len(order):
                    return True
                
                i = order[idx]
                
                # Determine available colors for vertex i
                available = []
                for c in range(4):
                    valid = True
                    for k in range(idx):
                        j = order[k]
                        if assignment[j] == c and j in adj[i][c]:
                            valid = False
                            break
                    if valid:
                        available.append(c)
                
                if not available:
                    return False
                
                # Shuffle available colors for randomization
                random.shuffle(available)
                
                for c in available:
                    assignment[i] = c
                    if backtrack(idx + 1):
                        return True
                    assignment[i] = -1
                
                return False

            if backtrack(0):
                return True, assignment
        
        return False, None

    def local_search_coloring(target_n, deadline):
        """
        Try to find a valid 4-coloring of K_{target_n} with no monochromatic triangles
        using randomized local search (simulated annealing style).
        Returns colors list or None.
        """
        if time.time() > deadline - 1:
            return None
        
        # Initialize: use the base coloring for first 32 vertices, random for rest
        cur_colors = base_colors(32)
        for i in range(32, target_n):
            for j in range(i + 1, target_n):
                cur_colors.append(random.randint(0, 3))
        
        # Precompute all edges
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                edges.append((i, j))
        
        def count_mono_triangles(colors):
            """Count monochromatic triangles."""
            count = 0
            # Build adjacency for quick lookup
            for a in range(target_n):
                for b in range(a + 1, target_n):
                    ca = colors[color_index(target_n, a, b)]
                    for c in range(b + 1, target_n):
                        cb = colors[color_index(target_n, a, c)]
                        if cb != ca:
                            continue
                        cc = colors[color_index(target_n, b, c)]
                        if cc == ca:
                            count += 1
            return count
        
        # Simulated annealing
        cur_energy = count_mono_triangles(cur_colors)
        best_colors = cur_colors[:]
        best_energy = cur_energy
        
        temperature = 10.0
        cooling = 0.995
        max_iterations = 200000
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 1:
                break
            
            # Pick a random edge to flip
            ei = random.randint(0, len(edges) - 1)
            i, j = edges[ei]
            idx = color_index(target_n, i, j)
            old_color = cur_colors[idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Compute delta energy: how many mono triangles change
            # A triangle (i,j,k) is affected if it includes edge (i,j)
            # Count mono triangles containing (i,j) before and after
            delta = 0
            for k in range(target_n):
                if k == i or k == j:
                    continue
                # Triangle (i, j, k)
                # Get colors of edges (i,k) and (j,k)
                c_ik = cur_colors[color_index(target_n, i, k)]
                c_jk = cur_colors[color_index(target_n, j, k)]
                
                # Before flip: colors are old_color, c_ik, c_jk
                before_mono = 1 if (old_color == c_ik == c_jk) else 0
                
                # After flip: colors are new_color, c_ik, c_jk
                after_mono = 1 if (new_color == c_ik == c_jk) else 0
                
                delta += after_mono - before_mono
            
            cur_energy += delta
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_colors = cur_colors[:]
            
            # Accept or reject
            if delta < 0:
                cur_colors[idx] = new_color
            else:
                # Accept with probability exp(-delta / temperature)
                if random.random() < pow(2.718281828, -delta / temperature):
                    cur_colors[idx] = new_color
                else:
                    cur_energy -= delta  # revert
            
            temperature *= cooling
        
        if best_energy == 0:
            return best_colors
        return None

    # Main loop: try to extend the coloring
    adj = build_adj(colors, n)
    
    for attempt in range(15):  # Try to add up to 15 more vertices
        if time.time() > deadline:
            break
        
        # Rebuild adj each time
        adj = build_adj(colors, n)
        
        success, assignment = try_add_vertex(colors, n, adj, deadline, max_restarts=50)
        
        if success:
            new_n = n + 1
            # Append colors for edges (new_n-1, i) for i in 0..n-1
            for i in range(n):
                colors.append(assignment[i])
            n = new_n
            best_n = max(best_n, n)
            best_colors = colors[:]
        else:
            break

    # If we couldn't extend much, try local search for larger n
    if best_n < 45 and time.time() < deadline - 5:
        for target_n in [36, 40, 44, 48, 50]:
            if time.time() > deadline - 2:
                break
            if target_n <= best_n:
                continue
            
            result = local_search_coloring(target_n, deadline)
            if result is not None:
                best_n = target_n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
