import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # --- GF(16) based 32-vertex coloring (Greenwood-Gleason) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_tbl, x = {}, 1
    for e in range(15):
        log_tbl[x] = e
        x = mul(x, 2)

    def gf16_color(i, j):
        # Color based on GF(16) difference
        return log_tbl[(i % 16) ^ (j % 16)] % 3

    def make_adj_32():
        adj = [[-1] * 32 for _ in range(32)]
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    c = gf16_color(i, j)
                else:
                    c = 3
                adj[i][j] = adj[j][i] = c
        return adj

    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(adj[i][j])
        return colors

    def colors_to_adj(colors, n):
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj

    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def is_valid(adj, n):
        return count_mono_triangles(adj, n) == 0

    def add_vertex_backtrack(adj, n, time_limit):
        """
        Try to add vertex n (0-indexed) to a valid coloring of n vertices.
        Returns assignment list of length n if successful, else None.
        Uses backtracking with forward checking and MRV heuristic.
        """
        start = time.time()
        
        # For each vertex u in 0..n-1, and each color c,
        # check if assigning c to the new vertex creates a monochromatic triangle with u and some other vertex.
        # Actually, the new vertex is connected to all existing vertices.
        # A monochromatic triangle involving the new vertex v_new, u, w occurs if:
        #   color(v_new, u) == color(v_new, w) == color(u, w)
        # So for a fixed color c assigned to (v_new, u), we must ensure that for all w != u,
        #   it is NOT the case that color(v_new, w) == c AND color(u, w) == c.
        # Wait, that's not quite right. The triangle is (v_new, u, w).
        # The edges are (v_new, u), (v_new, w), (u, w).
        # They are monochromatic if all three have the same color.
        # Let c_u = color(v_new, u) and c_w = color(v_new, w).
        # If c_u == c_w == c, then we need color(u, w) != c.
        
        # Let's define domains. Initially, each vertex u can have any color 0..3.
        # But if we assign color c to edge (v_new, u), then for every other vertex w,
        # if we assign color c to edge (v_new, w), then color(u, w) must NOT be c.
        # This is a global constraint.
        
        # Alternative view:
        # For each color c, let S_c be the set of vertices u such that color(v_new, u) == c.
        # Then S_c must be an independent set in the graph G_c, where G_c has edges (u, w)
        # such that color(u, w) == c.
        # Because if u, w in S_c, then color(v_new, u) = color(v_new, w) = c.
        # If color(u, w) = c, then triangle (v_new, u, w) is monochromatic.
        
        # So the problem reduces to:
        # Partition the set of vertices {0..n-1} into 4 sets S_0, S_1, S_2, S_3
        # such that each S_c is an independent set in G_c.
        
        # This is a 4-coloring problem of a hypergraph or a constraint satisfaction problem.
        # Specifically, it's equivalent to finding a 4-coloring of the vertices such that
        # no edge of G_c connects two vertices of color c.
        
        # This is exactly the problem of "coloring the vertices with 4 colors such that
        # for each c, the subgraph induced by vertices of color c has no edges from G_c".
        
        # Let's use backtracking with forward checking.
        
        # Precompute for each vertex u and color c, the set of vertices w that are "forbidden"
        # if u is assigned color c. i.e., if color(v_new, u) = c, then color(v_new, w) cannot be c
        # if color(u, w) = c.
        
        # Actually, simpler:
        # Let's maintain for each vertex u, the set of allowed colors.
        # Initially all {0,1,2,3}.
        # When we assign color c to vertex u, we must remove color c from the domain of
        # every vertex w such that color(u, w) == c.
        
        # This is standard arc consistency / forward checking.
        
        # MRV heuristic: pick the vertex with the smallest domain.
        
        # Let's implement this.
        
        domains = [set([0, 1, 2, 3]) for _ in range(n)]
        assignment = [-1] * n
        
        # Precompute adjacency lists for each color
        # neighbors_c[u] = list of w such that color(u, w) == c
        neighbors = [[[] for _ in range(4)] for _ in range(n)]
        for u in range(n):
            for w in range(u + 1, n):
                c = adj[u][w]
                neighbors[u][c].append(w)
                neighbors[w][c].append(u)
        
        def backtrack(idx, unassigned):
            if time.time() - start > time_limit:
                return False
            
            if not unassigned:
                return True
            
            # MRV: pick unassigned vertex with smallest domain
            min_size = 5
            min_idx = -1
            for u in unassigned:
                if len(domains[u]) < min_size:
                    min_size = len(domains[u])
                    min_idx = u
            
            if min_idx == -1 or min_size == 0:
                return False
            
            # Try each color in the domain of min_idx
            for c in list(domains[min_idx]):
                # Assign c to min_idx
                assignment[min_idx] = c
                domains[min_idx] = {c}
                
                # Check if this assignment is consistent
                # For each neighbor w of min_idx in color c, remove c from domains[w]
                # If any domain becomes empty, backtrack
                removed = []
                consistent = True
                for w in neighbors[min_idx][c]:
                    if c in domains[w]:
                        domains[w].remove(c)
                        removed.append(w)
                        if not domains[w]:
                            consistent = False
                            break
                
                if consistent:
                    new_unassigned = unassigned - {min_idx}
                    if backtrack(0, new_unassigned):
                        return True
                
                # Undo
                assignment[min_idx] = -1
                domains[min_idx] = set([0, 1, 2, 3])
                for w in removed:
                    domains[w].add(c)
            
            return False
        
        if backtrack(0, set(range(n))):
            return assignment
        return None

    def add_vertex_sa(adj, n, time_limit):
        """
        Try to add vertex n using simulated annealing.
        Returns assignment list of length n if successful, else None.
        """
        start = time.time()
        
        # Energy function: number of monochromatic triangles involving the new vertex.
        # For a given assignment of colors to edges (v_new, u) for u in 0..n-1,
        # the energy is the number of pairs (u, w) such that
        #   color(v_new, u) == color(v_new, w) == color(u, w).
        
        def energy(assign):
            e = 0
            for u in range(n):
                cu = assign[u]
                for w in range(u + 1, n):
                    if assign[w] == cu and adj[u][w] == cu:
                        e += 1
            return e
        
        # Initialize
        assign = [random.randint(0, 3) for _ in range(n)]
        e = energy(assign)
        best_assign = assign[:]
        best_e = e
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() - start < time_limit and T > T_min:
            u = random.randint(0, n - 1)
            old_c = assign[u]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Compute delta
            delta = 0
            for w in range(n):
                if w == u:
                    continue
                cw = assign[w]
                # Old contribution: if old_c == cw and adj[u][w] == old_c, then 1
                if old_c == cw and adj[u][w] == old_c:
                    delta -= 1
                # New contribution: if new_c == cw and adj[u][w] == new_c, then 1
                if new_c == cw and adj[u][w] == new_c:
                    delta += 1
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                assign[u] = new_c
                e += delta
                if e < best_e:
                    best_e = e
                    best_assign = assign[:]
                    if best_e == 0:
                        return best_assign
            
            T *= cooling
        
        return best_assign if best_e == 0 else None

    # Start with 32-vertex coloring
    adj = make_adj_32()
    best_n = 32
    best_colors = adj_to_colors(adj, best_n)
    
    # Try to expand
    for target_n in range(33, 51):
        if time_left() < 10:
            break
        
        n = len(adj)
        found = False
        
        # Try backtracking first (exact, but may be slow)
        if time_left() > 5:
            bt_limit = min(5.0, time_left() - 5)
            assignment = add_vertex_backtrack(adj, n, bt_limit)
            if assignment is not None:
                # Success
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
        
        # If backtracking didn't work, try SA
        if not found and time_left() > 5:
            sa_limit = min(10.0, time_left() - 5)
            assignment = add_vertex_sa(adj, n, sa_limit)
            if assignment is not None:
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
        
        if not found:
            break
    
    # Final validation
    adj_final = colors_to_adj(best_colors, best_n)
    if not is_valid(adj_final, best_n):
        return {'n': 32, 'colors': adj_to_colors(make_adj_32(), 32)}
    
    return {'n': best_n, 'colors': best_colors}
