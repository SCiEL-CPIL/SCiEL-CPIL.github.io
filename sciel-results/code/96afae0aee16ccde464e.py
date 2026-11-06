import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges, timeout=5.0):
    """Fast bitmask-based backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no proper 2-coloring exists)."""
    start = time.time()
    
    # Precompute edge masks
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
    
    m = len(edges)
    all_mask = (1 << n) - 1
    
    # For each edge, precompute the list of vertices
    edge_verts = [list(e) for e in edges]
    
    # State: color_mask (bit v set means vertex v is color 1), uncolored_mask
    # We use a stack-based approach with undo
    
    def check_not_2colorable():
        """Try to find a 2-coloring. Return False if none exists (not 2-colorable)."""
        nonlocal start
        color1 = [0] * n  # 1 if vertex is color 1, 0 if color 0, -1 if unassigned
        for i in range(n):
            color1[i] = -1
        
        # Count how many vertices of each color are in each edge
        # For efficiency, we track per-edge: count0, count1, unassigned
        edge_count0 = [0] * m
        edge_count1 = [0] * m
        edge_unassigned = [5] * m
        
        def propagate():
            """Unit propagation. Return False on conflict."""
            nonlocal start
            if time.time() - start > timeout:
                return None  # timeout
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    if edge_unassigned[i] == 0:
                        # All assigned, check for monochromatic
                        if edge_count0[i] == 5 or edge_count1[i] == 5:
                            return False
                    elif edge_unassigned[i] == 1:
                        # One unassigned, four assigned
                        if edge_count0[i] == 4:
                            # The unassigned must be color 1
                            # Find the unassigned vertex
                            for v in edge_verts[i]:
                                if color1[v] == -1:
                                    color1[v] = 1
                                    edge_count1[i] += 1
                                    edge_unassigned[i] -= 1
                                    changed = True
                                    break
                        elif edge_count1[i] == 4:
                            # The unassigned must be color 0
                            for v in edge_verts[i]:
                                if color1[v] == -1:
                                    color1[v] = 0
                                    edge_count0[i] += 1
                                    edge_unassigned[i] -= 1
                                    changed = True
                                    break
            return True
        
        # Order vertices by degree (most constrained first)
        vertex_deg = [0] * n
        for i in range(m):
            for v in edge_verts[i]:
                vertex_deg[v] += 1
        vertex_order = sorted(range(n), key=lambda v: -vertex_deg[v])
        
        def backtrack(pos):
            nonlocal start
            if time.time() - start > timeout:
                return None
            if pos == n:
                return True  # Found valid coloring
            
            v = vertex_order[pos]
            
            # Try color 0
            color1[v] = 0
            # Update edge counts
            affected = []
            for i in range(m):
                if v in edge_verts[i]:
                    edge_count0[i] += 1
                    edge_unassigned[i] -= 1
                    affected.append(i)
            
            result = propagate()
            if result is True:
                res = backtrack(pos + 1)
                if res is not None:
                    # Undo and return
                    color1[v] = -1
                    for i in affected:
                        edge_count0[i] -= 1
                        edge_unassigned[i] += 1
                    return res
            elif result is None:
                # Timeout
                color1[v] = -1
                for i in affected:
                    edge_count0[i] -= 1
                    edge_unassigned[i] += 1
                return None
            
            # Undo color 0
            color1[v] = -1
            for i in affected:
                edge_count0[i] -= 1
                edge_unassigned[i] += 1
            
            # Try color 1
            color1[v] = 1
            affected = []
            for i in range(m):
                if v in edge_verts[i]:
                    edge_count1[i] += 1
                    edge_unassigned[i] -= 1
                    affected.append(i)
            
            result = propagate()
            if result is True:
                res = backtrack(pos + 1)
                if res is not None:
                    color1[v] = -1
                    for i in affected:
                        edge_count1[i] -= 1
                        edge_unassigned[i] += 1
                    return res
            elif result is None:
                color1[v] = -1
                for i in affected:
                    edge_count1[i] -= 1
                    edge_unassigned[i] += 1
                return None
            
            # Undo color 1
            color1[v] = -1
            for i in affected:
                edge_count1[i] -= 1
                edge_unassigned[i] += 1
            
            return None
        
        # Initial propagation
        result = propagate()
        if result is False:
            return True  # Not 2-colorable (conflict in initial state)
        if result is None:
            return False  # Timeout, assume 2-colorable
        
        result = backtrack(0)
        if result is True:
            return False  # 2-colorable
        elif result is None:
            return False  # Timeout, assume 2-colorable
        else:
            return True  # Not 2-colorable
    
    return check_not_2colorable()

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    
    best_n = None
    best_edges = None
    best_count = float('inf')
    
    # Try different n values, starting from smaller n where dense hypergraphs
    # are more likely to be non-2-colorable
    for n in range(9, 27):
        if time.time() > deadline - 10:
            break
        
        all_possible = list(combinations(range(n), 5))
        total_edges = len(all_possible)
        
        # Multiple random restarts
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset of edges
            # Use a moderate number to increase chance of non-2-colorability
            num_edges = min(total_edges, max(20, n * 2))
            random.shuffle(all_possible)
            selected = all_possible[:num_edges]
            
            # Check if non-2-colorable
            if _is_not_2_colorable(n, selected, timeout=3.0):
                # Found a non-2-colorable hypergraph! Now minimize via greedy removal
                current_edges = list(selected)
                
                # Greedy edge removal: try removing each edge, keep if still non-2-colorable
                i = 0
                while i < len(current_edges):
                    if time.time() > deadline - 5:
                        break
                    
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if _is_not_2_colorable(n, test_edges, timeout=2.0):
                        current_edges = test_edges
                        # Don't increment i
                    else:
                        i += 1
                
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_n = n
                    best_edges = [list(e) for e in current_edges]
                    if best_count <= 51:
                        break
            
            # Try fewer edges in subsequent attempts
            if attempt > 5:
                num_edges = max(15, num_edges - 3)
        
        if best_count <= 51:
            break
    
    # Fallback: complete 5-uniform on 9 vertices (126 edges, definitely non-2-colorable)
    if best_edges is None or best_count > 126:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}
