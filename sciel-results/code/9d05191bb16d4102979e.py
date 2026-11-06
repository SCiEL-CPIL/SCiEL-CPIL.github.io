"""Vertex-by-vertex construction with backtracking for C6-free 4-colorings of K_n."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: n=9 seed
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best = {'n': n_seed, 'colors': seed_colors}
    
    # Try n=17 first (known to be possible), then n=18
    for target_n in [17, 18]:
        if time.time() > deadline - 30.0:
            break
        try:
            result = _vertex_backtrack(target_n, deadline - 20.0)
            if result is not None:
                return result
        except Exception:
            pass
    
    # If backtracking failed, try local search for n=17
    if time.time() < deadline - 10.0:
        try:
            result = _local_search(17, deadline - 5.0)
            if result is not None:
                return result
        except Exception:
            pass
    
    return best


def _vertex_backtrack(n, deadline):
    """
    Build K_n vertex by vertex. When adding vertex v, assign colors to edges (v, u) for u < v.
    Check that no C6 is formed in any color class.
    """
    if time.time() > deadline - 5.0:
        return None
    
    # colors[i][j] = color of edge (i,j), i<j, or -1 if unassigned
    # We'll store as a 2D list
    colors = [[-1] * n for _ in range(n)]
    
    # Precompute: for each vertex v, which pairs (a,b) with a,b < v could form a C6 with v?
    # A C6 involving v has form: v - a - b - c - d - e - v
    # So we need 5 other vertices a,b,c,d,e < v
    # The edges are: (v,a), (a,b), (b,c), (c,d), (d,e), (e,v)
    # All these edges must have the same color
    
    # For efficiency, we'll check C6 violations incrementally
    
    def check_c6_with_new_vertex(v, new_colors):
        """
        Check if adding vertex v with given edge colors creates any C6.
        new_colors: dict mapping u -> color for edge (v,u), u < v
        Returns True if no C6 violation.
        """
        if time.time() > deadline - 2.0:
            return False
        
        # For C6: v - a - b - c - d - e - v
        # We need to find a,b,c,d,e < v such that all 6 edges have same color
        # Edges: (v,a), (a,b), (b,c), (c,d), (d,e), (e,v)
        
        # Group edges incident to v by color
        edges_by_color = defaultdict(list)
        for u in range(v):
            c = new_colors.get(u, colors[v][u])
            if c >= 0:
                edges_by_color[c].append(u)
        
        for color, neighbors in edges_by_color.items():
            if len(neighbors) < 2:
                continue
            
            # For each pair (a, e) in neighbors, check if there's a path a-b-c-d-e of length 4
            # where all edges have this color
            # This is expensive, so let's be smarter
            
            # Build adjacency for this color class among vertices < v
            adj = defaultdict(list)
            for i in range(v):
                for j in range(i + 1, v):
                    if colors[i][j] == color:
                        adj[i].append(j)
                        adj[j].append(i)
            
            # For each a in neighbors, do BFS to find if there's a path of length 4 to any e in neighbors
            # Path: a - b - c - d - e where a,e in neighbors, b,c,d < v
            # We need exactly 4 edges, so depth 4 BFS from a, but we must end at e in neighbors
            
            neighbor_set = set(neighbors)
            
            for a in neighbors:
                # BFS from a, depth 4, looking for any e in neighbor_set (e != a)
                # State: (current_vertex, depth, visited_set)
                # But visited_set is expensive. Instead, just do BFS up to depth 4.
                
                # BFS level by level
                level = {a}
                prev_level = {a}
                
                for depth in range(4):
                    next_level = set()
                    for node in level:
                        for nb in adj[node]:
                            if nb not in prev_level:  # Avoid going back
                                if depth == 3 and nb in neighbor_set and nb != a:
                                    return False  # Found C6
                                next_level.add(nb)
                    prev_level = level
                    level = next_level
                
                if not level:
                    break
        
        return True
    
    def backtrack(v):
        """Assign colors to edges (v, u) for u < v."""
        if time.time() > deadline - 2.0:
            return False
        
        if v == 0:
            return True
        
        if not backtrack(v - 1):
            return False
        
        # Now assign colors to edges (v, u) for u < v
        # Try all 4^v assignments? Too many. Use backtracking.
        
        # Order edges by constraint: edges to vertices that are in many C6s first
        # For simplicity, just go in order u = 0, 1, ..., v-1
        
        def assign_edges(u_idx):
            if time.time() > deadline - 2.0:
                return False
            
            if u_idx == v:
                # All edges assigned, check for C6
                new_colors = {}
                for u in range(v):
                    new_colors[u] = colors[v][u]
                return check_c6_with_new_vertex(v, new_colors)
            
            u = u_idx
            for color in range(4):
                colors[v][u] = color
                if assign_edges(u_idx + 1):
                    return True
                colors[v][u] = -1
            
            return False
        
        return assign_edges(0)
    
    if backtrack(n - 1):
        # Convert to required format
        result_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                result_colors.append(colors[i][j])
        return {'n': n, 'colors': result_colors}
    
    return None


def _local_search(n, deadline):
    """Local search with conflict-directed moves."""
    if time.time() > deadline - 5.0:
        return None
    
    # Initialize with random coloring
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    
    # Precompute edge to vertex mapping
    edge_vertices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_vertices.append((i, j))
            idx += 1
    
    num_edges = len(edge_vertices)
    
    # Precompute all C6 cycles
    cycles = []
    edge_to_cycles = defaultdict(set)
    
    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            min_idx = perm.index(min(perm))
            rot = perm[min_idx:] + perm[:min_idx]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)
            
            edges = []
            for k in range(6):
                u = rot[k]
                v = rot[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                # Find edge index
                for e_idx, (a, b) in enumerate(edge_vertices):
                    if a == u and b == v:
                        edges.append(e_idx)
                        break
            
            cycle_id = len(cycles)
            cycles.append(tuple(edges))
            for e in edges:
                edge_to_cycles[e].add(cycle_id)
    
    num_cycles = len(cycles)
    
    # Count violations
    def count_violations(colors):
        violations = 0
        for cycle in cycles:
            color_counts = [0] * 4
            for e in cycle:
                color_counts[colors[e]] += 1
            if any(c == 6 for c in color_counts):
                violations += 1
        return violations
    
    total_violations = count_violations(colors)
    
    # Local search
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 1.0:
        if total_violations == 0:
            return {'n': n, 'colors': colors}
        
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
        
        # Calculate delta
        delta = 0
        for c_id in edge_to_cycles[e]:
            cycle = cycles[c_id]
            old_counts = [0] * 4
            new_counts = [0] * 4
            for ee in cycle:
                if ee == e:
                    old_counts[old_color] += 1
                    new_counts[new_color] += 1
                else:
                    old_counts[colors[ee]] += 1
                    new_counts[colors[ee]] += 1
            
            old_v = 1 if any(c == 6 for c in old_counts) else 0
            new_v = 1 if any(c == 6 for c in new_counts) else 0
            delta += new_v - old_v
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colors[e] = new_color
            total_violations += delta
        T *= cooling
        if T < min_T:
            T = min_T
    
    if total_violations == 0:
        return {'n': n, 'colors': colors}
    
    return None
# EVOLVE-BLOCK-END
