"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, time_limit=0.5):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation."""
    start_time = time.time()
    m = len(edges)
    if m == 0:
        return True
    
    # Precompute edge membership: for each vertex, list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: color[v] = -1 (unassigned), 0, or 1
    # For each edge, track count of 0s and 1s
    edge_zero = [0] * m
    edge_one = [0] * m
    
    # Unit propagation: if an edge has 4 vertices of same color, the 5th must be different
    # If an edge has 4 vertices of same color and the 5th is unassigned, assign it to the other color
    
    def propagate(queue):
        """Process queue of (vertex, color) assignments. Returns False if contradiction."""
        while queue:
            v, c = queue.pop()
            if color[v] != -1:
                if color[v] != c:
                    return False
                continue
            color[v] = c
            for ei in vertex_edges[v]:
                if c == 0:
                    edge_zero[ei] += 1
                    if edge_zero[ei] == 5:
                        # All 5 are 0, contradiction if we need both colors
                        # Wait, 2-coloring means each edge has both colors
                        # So if all 5 are same color, it's NOT a valid 2-coloring
                        # But we're checking if there EXISTS a 2-coloring
                        # So monochromatic edge means this path fails
                        return False
                else:
                    edge_one[ei] += 1
                    if edge_one[ei] == 5:
                        return False
        return True
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return None  # timeout
        
        # Find unassigned vertex with most constraints
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            # All assigned, check if valid
            return True
        
        # Choose vertex with most edges
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        # Try color 0
        for c in [0, 1]:
            # Check if assigning v to c creates immediate contradiction
            ok = True
            for ei in vertex_edges[v]:
                if c == 0 and edge_zero[ei] == 4:
                    ok = False
                    break
                if c == 1 and edge_one[ei] == 4:
                    ok = False
                    break
            if not ok:
                continue
            
            # Save state
            color[v] = c
            changes = []
            for ei in vertex_edges[v]:
                if c == 0:
                    edge_zero[ei] += 1
                    changes.append((ei, 'zero', 1))
                else:
                    edge_one[ei] += 1
                    changes.append((ei, 'one', 1))
            
            # Check for unit propagation
            queue = []
            for ei in vertex_edges[v]:
                if edge_zero[ei] == 4:
                    # One vertex left unassigned in this edge, must be 1
                    for u in edges[ei]:
                        if color[u] == -1:
                            queue.append((u, 1))
                            break
                elif edge_one[ei] == 4:
                    for u in edges[ei]:
                        if color[u] == -1:
                            queue.append((u, 0))
                            break
            
            if propagate(queue):
                if backtrack():
                    return True
            
            # Undo
            for ei, kind, delta in changes:
                if kind == 'zero':
                    edge_zero[ei] -= delta
                else:
                    edge_one[ei] -= delta
            color[v] = -1
            
            # Undo propagation - this is tricky, need to re-propagate
            # For simplicity, let's use a different approach: full state copy
            
        return False
    
    # Simpler approach: use state copying for correctness
    def backtrack_simple():
        if time.time() - start_time > time_limit:
            return None
        
        # Find unassigned vertex
        unassigned = [v for v in range(n) if color[v] == -1]
        if not unassigned:
            return True
        
        # Choose vertex with most constraints
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        for c in [0, 1]:
            # Try assigning v to c
            new_color = color[:]
            new_edge_zero = edge_zero[:]
            new_edge_one = edge_one[:]
            
            new_color[v] = c
            valid = True
            for ei in vertex_edges[v]:
                if c == 0:
                    new_edge_zero[ei] += 1
                    if new_edge_zero[ei] == 5:
                        valid = False
                        break
                else:
                    new_edge_one[ei] += 1
                    if new_edge_one[ei] == 5:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Unit propagation
            queue = [(v, c)]
            while queue:
                uv, uc = queue.pop()
                for ei in vertex_edges[uv]:
                    if uc == 0 and new_edge_zero[ei] == 4:
                        for u in edges[ei]:
                            if new_color[u] == -1:
                                if u != uv:
                                    new_color[u] = 1
                                    new_edge_one[ei] += 1
                                    if new_edge_one[ei] == 5:
                                        valid = False
                                    queue.append((u, 1))
                                break
                    elif uc == 1 and new_edge_one[ei] == 4:
                        for u in edges[ei]:
                            if new_color[u] == -1:
                                if u != uv:
                                    new_color[u] = 0
                                    new_edge_zero[ei] += 1
                                    if new_edge_zero[ei] == 5:
                                        valid = False
                                    queue.append((u, 0))
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check if all assigned
                if all(new_color[i] != -1 for i in range(n)):
                    return True
                # Recurse
                color_copy = new_color
                ez_copy = new_edge_zero
                eo_copy = new_edge_one
                color[:] = color_copy
                edge_zero[:] = ez_copy
                edge_one[:] = eo_copy
                if backtrack_simple():
                    return True
                # Restore
                color[:] = color
                # Actually we need to restore properly - this is getting complex
        
        return False
    
    # Let me use a cleaner recursive approach with explicit state
    def solve():
        if time.time() - start_time > time_limit:
            return None
        
        color = [-1] * n
        edge_zero = [0] * m
        edge_one = [0] * m
        
        def rec(depth):
            if time.time() - start_time > time_limit:
                return None
            
            # Find unassigned vertex with most constraints
            best_v = -1
            best_count = -1
            for v in range(n):
                if color[v] == -1:
                    cnt = len(vertex_edges[v])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
            
            if best_v == -1:
                return True  # All assigned
            
            for c in [0, 1]:
                # Try assigning best_v to c
                ok = True
                for ei in vertex_edges[best_v]:
                    if c == 0 and edge_zero[ei] == 4:
                        ok = False
                        break
                    if c == 1 and edge_one[ei] == 4:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Apply assignment
                color[best_v] = c
                if c == 0:
                    edge_zero[best_v_ei] = edge_zero[best_v_ei]  # placeholder
                
                # This is getting too complex. Let me use a simpler but correct approach.
                pass
        
        return False
    
    # Let me just use a straightforward backtracking without unit propagation for simplicity
    # and rely on the small size
    color = [-1] * n
    
    def bt():
        if time.time() - start_time > time_limit:
            return None
        
        # Find first unassigned
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        if v == -1:
            return True
        
        for c in [0, 1]:
            color[v] = c
            # Check all edges containing v
            ok = True
            for ei in vertex_edges[v]:
                e = edges[ei]
                zeros = sum(1 for u in e if color[u] == 0)
                ones = sum(1 for u in e if color[u] == 1)
                unassigned = sum(1 for u in e if color[u] == -1)
                # If already 5 of one color, invalid
                if zeros == 5 or ones == 5:
                    ok = False
                    break
                # If 4 of one color and 1 unassigned, that unassigned must be other color
                # But we don't enforce that here, just check current state
            if ok:
                if bt():
                    return True
            color[v] = -1
        return False
    
    result = bt()
    if result is None:
        return None  # timeout
    return result

def construct():
    """
    Find a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Best known: 51 edges.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with a known good construction and try to improve
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5) = 126 edges
    # We want to minimize this
    
    # Strategy: Use a random search / local search approach
    # Start with a random set of edges and try to find a minimal non-2-colorable subgraph
    
    best_n = 9
    best_edges = None
    best_count = float('inf')
    
    # First, get a valid solution to start with
    # Use the complete hypergraph on 9 vertices, then try to remove edges
    all_edges_9 = list(combinations(range(9), 5))
    random.shuffle(all_edges_9)
    
    # Greedy: start with all edges, remove edges that don't break 2-colorability
    current_edges = all_edges_9[:]
    
    # Try to minimize by removing edges
    for i in range(len(current_edges) - 1, -1, -1):
        if time.time() > deadline:
            break
        # Try removing edge i
        test_edges = current_edges[:i] + current_edges[i+1:]
        if _is_2_colorable(9, test_edges, time_limit=0.1):
            # Removing this edge makes it 2-colorable, so we must keep it
            continue
        else:
            # Removing this edge still leaves it non-2-colorable, so remove it
            current_edges = test_edges
    
    if current_edges:
        best_edges = current_edges
        best_count = len(current_edges)
        best_n = 9
    
    # Try with different vertex counts
    for n in range(10, 16):
        if time.time() > deadline - 5:
            break
        all_edges_n = list(combinations(range(n), 5))
        random.shuffle(all_edges_n)
        
        # Random subset approach: pick random subsets and check
        # Start with a large random subset
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            # Pick a random subset of edges
            k = min(len(all_edges_n), 200)
            subset = random.sample(all_edges_n, k)
            
            # Check if this subset is non-2-colorable
            if not _is_2_colorable(n, subset, time_limit=0.1):
                # Try to minimize
                current = subset[:]
                for i in range(len(current) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    test = current[:i] + current[i+1:]
                    if _is_2_colorable(n, test, time_limit=0.05):
                        continue
                    else:
                        current = test
                
                if len(current) < best_count:
                    best_count = len(current)
                    best_edges = current
                    best_n = n
    
    # If we found a good solution, return it
    if best_edges is not None:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    
    # Fallback: return the complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
