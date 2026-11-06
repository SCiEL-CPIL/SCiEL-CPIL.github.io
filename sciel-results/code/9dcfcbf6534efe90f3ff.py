import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build the base K32 coloring (two copies of K16, color 3 between)
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
    
    def build_k32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors
    
    # Precompute edge index for K_n: index of (i,j) in lex order
    # idx(i,j,n) = i*(2n-1-i)//2 + (j-i-1)
    
    def get_edge_idx(i, j, n):
        return i * (2 * n - 1 - i) // 2 + (j - i - 1)
    
    # Incremental validity check: adding vertex v (index = n) with edges to 0..n-1
    # For each color c, the set S_c = {u : color(v,u)=c} must be independent in color-c graph
    def is_valid_new_vertex(base_colors, base_n, new_edges):
        """
        Check if adding a new vertex (index base_n) with given edge colors to existing vertices
        creates any monochromatic triangle.
        new_edges[u] = color of edge (base_n, u) for u in 0..base_n-1
        """
        # For each color c, collect vertices connected to new vertex with color c
        # Then check no two such vertices are connected by color c in base
        for c in range(4):
            verts_c = []
            for u in range(base_n):
                if new_edges[u] == c:
                    verts_c.append(u)
            # Check all pairs in verts_c
            for a in range(len(verts_c)):
                i = verts_c[a]
                for b in range(a + 1, len(verts_c)):
                    j = verts_c[b]
                    # Get color of edge (i,j) in base
                    idx = get_edge_idx(i, j, base_n)
                    if base_colors[idx] == c:
                        return False
        return True
    
    # Local search: try to fix a vertex by reassigning its edges
    def try_fix_vertex(colors, n, v, max_iters=200):
        """
        Try to reassign edges of vertex v to make the coloring valid.
        Returns True if fixed, modifies colors in place.
        """
        if time.time() > deadline - 2:
            return False
        
        # Get current edges of v
        # v connects to all u != v
        # Edges (u,v) for u < v: indices get_edge_idx(u, v, n)
        # Edges (v,u) for u > v: indices get_edge_idx(v, u, n)
        
        def get_v_edges():
            edges = {}
            for u in range(v):
                idx = get_edge_idx(u, v, n)
                edges[u] = colors[idx]
            for u in range(v + 1, n):
                idx = get_edge_idx(v, u, n)
                edges[u] = colors[idx]
            return edges
        
        def set_v_edges(new_edges):
            for u in range(v):
                idx = get_edge_idx(u, v, n)
                colors[idx] = new_edges[u]
            for u in range(v + 1, n):
                idx = get_edge_idx(v, u, n)
                colors[idx] = new_edges[u]
        
        def check_vertex_v():
            """Check if vertex v creates any monochromatic triangle."""
            edges = get_v_edges()
            for c in range(4):
                verts_c = [u for u in range(n) if u != v and edges[u] == c]
                for a in range(len(verts_c)):
                    i = verts_c[a]
                    for b in range(a + 1, len(verts_c)):
                        j = verts_c[b]
                        if i < j:
                            idx = get_edge_idx(i, j, n)
                        else:
                            idx = get_edge_idx(j, i, n)
                        if colors[idx] == c:
                            return False
            return True
        
        current_edges = get_v_edges()
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                return False
            
            if check_vertex_v():
                return True
            
            # Pick a random vertex u connected to v and change color
            u = random.randint(0, n - 1)
            if u == v:
                continue
            old_c = current_edges[u]
            new_c = random.choice([c for c in range(4) if c != old_c])
            current_edges[u] = new_c
            set_v_edges(current_edges)
            
            # Accept if it doesn't make things worse (or with some probability)
            # Simple: just keep going, we'll check at the end
            
        return check_vertex_v()
    
    # Main: try to extend K32 to larger n
    current_n = 32
    current_colors = build_k32()
    
    # Precompute for faster incremental checks
    # For each color c and each vertex pair, we need quick access
    
    # Try to add vertices one by one
    max_target = 50
    
    for target_n in range(33, min(max_target + 1, 63)):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple restarts with random assignments
        num_restarts = 50000
        restarts_done = 0
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            restarts_done += 1
            
            # Random assignment for new vertex (index = current_n)
            new_edges = [random.randint(0, 3) for _ in range(current_n)]
            
            if is_valid_new_vertex(current_colors, current_n, new_edges):
                # Valid! Add the vertex
                # New edges go at the end of the colors list
                current_colors.extend(new_edges)
                current_n = target_n
                found = True
                break
        
        if not found:
            # Try local search: randomly assign, then try to fix
            for attempt in range(200):
                if time.time() > deadline - 10:
                    break
                
                new_edges = [random.randint(0, 3) for _ in range(current_n)]
                
                # Try to fix by local search on the new vertex
                # We'll simulate: add the vertex, then try to fix it
                temp_colors = current_colors + new_edges
                temp_n = current_n + 1
                
                # Check if valid
                if is_valid_new_vertex(current_colors, current_n, new_edges):
                    current_colors = temp_colors
                    current_n = temp_n
                    found = True
                    break
                
                # Try local search: reassign some edges of the new vertex
                # The new vertex is at index current_n
                v = current_n
                edges = new_edges[:]
                
                for it in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    if is_valid_new_vertex(current_colors, current_n, edges):
                        current_colors = current_colors + edges
                        current_n = temp_n
                        found = True
                        break
                    
                    # Change a random edge
                    u = random.randint(0, current_n - 1)
                    edges[u] = random.randint(0, 3)
                
                if found:
                    break
        
        if not found:
            # Can't extend to this n, break
            break
    
    # Verify the result
    if current_n > 32:
        # Do a full validation
        n = current_n
        colors = current_colors
        # Build adjacency per color
        adj = [[set() for _ in range(4)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        
        valid = True
        for c in range(4):
            for i in range(n):
                neighbors = adj[i][c]
                for j in neighbors:
                    for k in neighbors:
                        if k > j and k in adj[j][c]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {'n': n, 'colors': colors}
    
    # Fallback to K32
    return {'n': 32, 'colors': build_k32()}
