import time
import random
import sys
from itertools import combinations

def _valid_4col(colors, n):
    """Check that no colour class contains a triangle."""
    # Build adjacency per colour
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if 0 <= c < 4:
                adj[c].add((i, j))
                adj[c].add((j, i))
            idx += 1
    # Check triangles per colour
    for c in range(4):
        edges = adj[c]
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        for u in verts:
            neigh_u = [v for (a, b) in edges if a == u for v in [b]]
            # Build adjacency list
        # More efficient: build adj list
    adj_list = [[] for _ in range(4)]
    adj_set = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if 0 <= c < 4:
                adj_list[c].append((i, j))
                adj_set[c].add((i, j))
                adj_set[c].add((j, i))
            idx += 1
    for c in range(4):
        # For each pair of neighbours of a vertex, check if they're connected
        # Build adjacency list per colour
        adjlst = [[] for _ in range(n)]
        for (u, v) in adj_list[c]:
            adjlst[u].append(v)
            adjlst[v].append(u)
        for u in range(n):
            for idx_i, v in enumerate(adjlst[u]):
                for w in adjlst[u][idx_i+1:]:
                    if w in adj_set[c]:
                        return False
    return True

def _check_no_mono_triangle(colors, n):
    """Fast check: no monochromatic triangle."""
    # For each triple (i,j,k) with i<j<k, check if all three edges same colour
    edge_color = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_color[(i, j)] = colors[idx]
            idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c1 = edge_color.get((i, j))
                c2 = edge_color.get((i, k))
                c3 = edge_color.get((j, k))
                if c1 is not None and c2 is not None and c3 is not None:
                    if c1 == c2 == c3:
                        return False
    return True

def _build_seed_coloring():
    """Build the seed 32-vertex colouring from the current program."""
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
    return colors

def _try_extend(colors, n, target_n, deadline, rng):
    """
    Try to extend a valid n-vertex 4-colouring to target_n vertices.
    Uses backtracking with constraint checking.
    Returns (success, new_colors, new_n) or (False, colors, n).
    """
    if n >= target_n:
        return True, colors[:], n
    
    # Current edge colours as dict for fast lookup
    edge_colors = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_colors[(i, j)] = colors[idx]
            idx += 1
    
    # For each new vertex, we need to assign colours to edges to all existing vertices
    # such that no monochromatic triangle is formed.
    # A triangle (u, v, new) is monochromatic if edge(u,v) == edge(u,new) == edge(v,new).
    # So for new vertex x, for each existing vertex i, the colour c_i of edge(i,x) must
    # not equal edge(i,j) for any j that also has edge(j,x) == c_i.
    
    # More precisely: for each colour c, the set of existing vertices connected to x
    # with colour c must form an independent set in the subgraph of colour c.
    
    def can_assign(existing_colors_for_new, x_idx):
        """Check if assigning colours in existing_colors_for_new (list of colours for edges from x to 0..n-1) is valid."""
        # For each colour c, get the set of vertices connected to x with colour c
        for c in range(4):
            verts = [i for i, col in enumerate(existing_colors_for_new) if col == c]
            # Check that no pair in verts has edge colour c
            for ii in range(len(verts)):
                for jj in range(ii + 1, len(verts)):
                    u, v = verts[ii], verts[jj]
                    if u > v:
                        u, v = v, u
                    if edge_colors.get((u, v)) == c:
                        return False
        return True
    
    # Backtracking to assign vertex n, n+1, ..., target_n-1
    result_colors = list(colors)
    
    def backtrack(current_n):
        if current_n >= target_n:
            return True
        if time.time() > deadline:
            return False
        
        x = current_n
        # Try all assignments of colours to edges (x, 0), (x, 1), ..., (x, current_n-1)
        # That's 4^current_n possibilities - too many for large current_n
        # Use constraint propagation / backtracking per edge
        
        # Build assignment incrementally
        assignment = [-1] * current_n
        
        # For each colour c, maintain the set of vertices already assigned colour c to x
        # and check independence
        
        def is_partial_valid(assigned_up_to):
            """Check partial assignment up to assigned_up_to-1 is valid."""
            for c in range(4):
                verts = [i for i in range(assigned_up_to) if assignment[i] == c]
                for ii in range(len(verts)):
                    for jj in range(ii + 1, len(verts)):
                        u, v = verts[ii], verts[jj]
                        if u > v:
                            u, v = v, u
                        if edge_colors.get((u, v)) == c:
                            return False
            return True
        
        def backtrack_edge(pos):
            if pos == current_n:
                if is_partial_valid(current_n):
                    # Commit this vertex
                    for i in range(current_n):
                        result_colors.append(assignment[i])
                        edge_colors[(i, x)] = assignment[i]
                        edge_colors[(x, i)] = assignment[i]
                    return True
                return False
            if time.time() > deadline:
                return False
            
            # Symmetry breaking: try colours in some order
            for c in range(4):
                assignment[pos] = c
                if is_partial_valid(pos + 1):
                    if backtrack_edge(pos + 1):
                        return True
            assignment[pos] = -1
            return False
        
        # Shuffle for diversity
        # Actually, let's use a smarter approach: try to find a valid assignment
        # using randomized backtracking
        if backtrack_edge(0):
            return backtrack(current_n + 1)
        
        return False
    
    if backtrack(n):
        return True, result_colors, target_n
    return False, colors, n

def _local_search(colors, n, deadline, rng):
    """
    Try to extend the colouring by one vertex at a time using local search.
    Returns (colors, n) of the best found.
    """
    best_n = n
    best_colors = colors[:]
    
    # Try adding one vertex at a time
    current_n = n
    current_colors = colors[:]
    
    # Build edge colour lookup
    edge_colors = {}
    idx = 0
    for i in range(current_n):
        for j in range(i + 1, current_n):
            edge_colors[(i, j)] = current_colors[idx]
            idx += 1
    
    while current_n < 50 and time.time() < deadline:
        x = current_n
        # Assign colours to edges (x, 0), ..., (x, current_n-1)
        # Use local search: random initialisation, then fix violations
        
        # Random initialisation
        assignment = [rng.randint(0, 3) for _ in range(current_n)]
        
        best_cost = float('inf')
        best_assignment = None
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Compute violations: for each colour c, count pairs (i,j) where 
            # assignment[i]==c, assignment[j]==c, and edge_colors[(i,j)]==c
            cost = 0
            for c in range(4):
                verts = [i for i in range(current_n) if assignment[i] == c]
                for ii in range(len(verts)):
                    for jj in range(ii + 1, len(verts)):
                        u, v = verts[ii], verts[jj]
                        if u > v:
                            u, v = v, u
                        if edge_colors.get((u, v)) == c:
                            cost += 1
            
            if cost == 0:
                best_assignment = assignment[:]
                break
            
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment[:]
            
            # Perturbation: change one random edge's colour
            pos = rng.randint(0, current_n - 1)
            old = assignment[pos]
            new_c = rng.randint(0, 3)
            if new_c != old:
                assignment[pos] = new_c
        
        if best_assignment is not None and best_cost == 0:
            # Successfully added vertex
            for i in range(current_n):
                current_colors.append(best_assignment[i])
                edge_colors[(i, x)] = best_assignment[i]
                edge_colors[(x, i)] = best_assignment[i]
            current_n += 1
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
        else:
            # Failed to add vertex, break
            break
    
    return best_colors, best_n

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic triangles.
    Maximize n. Returns {'n': n, 'colors': [...]} where colors has one entry
    for each pair (i,j), i<j, in lexicographic order.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    # Start with the seed 32-vertex colouring
    colors = _build_seed_coloring()
    n = 32
    
    # Verify it's valid
    if not _check_no_mono_triangle(colors, n):
        # Fallback: use a simple valid construction
        # Just return a smaller valid one
        return {'n': 1, 'colors': [0]}
    
    best_colors = colors[:]
    best_n = n
    
    # Try to extend using local search
    extended_colors, extended_n = _local_search(colors, n, deadline, rng)
    if extended_n > best_n:
        best_n = extended_n
        best_colors = extended_colors
    
    # Try more random restarts with different initial perturbations
    attempt = 0
    while time.time() < deadline and attempt < 20:
        attempt += 1
        # Perturb the seed slightly and try to extend
        perturbed = colors[:]
        # Randomly flip a few colours
        for _ in range(rng.randint(1, 5)):
            pos = rng.randint(0, len(perturbed) - 1)
            perturbed[pos] = rng.randint(0, 3)
        
        if _check_no_mono_triangle(perturbed, n):
            ec, en = _local_search(perturbed, n, deadline, rng)
            if en > best_n:
                best_n = en
                best_colors = ec
    
    # Final verification
    if not _check_no_mono_triangle(best_colors, best_n):
        # Return the seed
        return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}
