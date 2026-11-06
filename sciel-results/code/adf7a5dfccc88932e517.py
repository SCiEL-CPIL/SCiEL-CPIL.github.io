import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not."""
    if not edges:
        return True
    
    # Build edge lists as tuples of vertex indices
    edge_tuples = [tuple(e) for e in edges]
    m = len(edge_tuples)
    
    # For each edge, precompute bitmask of vertices
    edge_masks = [0] * m
    for i, e in enumerate(edge_tuples):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
    
    # Adjacency: vertex -> list of edge indices containing it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vert_edges[v].append(i)
    
    # Order vertices by degree descending
    vert_order = sorted(range(n), key=lambda v: -len(vert_edges[v]))
    # Map from position in order to vertex
    order_to_vert = vert_order
    # Map from vertex to position in order
    vert_to_pos = [0] * n
    for pos, v in enumerate(vert_order):
        vert_to_pos[v] = pos
    
    # State: colors array indexed by vertex
    colors = [-1] * n
    
    # For efficient propagation, track for each edge:
    # count of 0s, count of 1s, list of unassigned vertices
    edge_cnt0 = [0] * m
    edge_cnt1 = [0] * m
    edge_unassigned = [list(e) for e in edge_tuples]
    
    def assign_vertex(v, c):
        """Assign color c to vertex v. Return False if conflict."""
        if colors[v] != -1:
            return colors[v] == c
        colors[v] = c
        for ei in vert_edges[v]:
            if c == 0:
                edge_cnt0[ei] += 1
            else:
                edge_cnt1[ei] += 1
            # Remove v from unassigned list of this edge
            lst = edge_unassigned[ei]
            for i, u in enumerate(lst):
                if u == v:
                    lst[i] = lst[-1]
                    lst.pop()
                    break
        return True
    
    def unassign_vertex(v):
        """Unassign vertex v."""
        if colors[v] == -1:
            return
        c = colors[v]
        colors[v] = -1
        for ei in vert_edges[v]:
            if c == 0:
                edge_cnt0[ei] -= 1
            else:
                edge_cnt1[ei] -= 1
            edge_unassigned[ei].append(v)
    
    def propagate():
        """Apply unit propagation. Return False if conflict detected."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                cnt0 = edge_cnt0[ei]
                cnt1 = edge_cnt1[ei]
                unass = edge_unassigned[ei]
                
                if cnt0 == 5 or cnt1 == 5:
                    return False
                
                if cnt0 == 4:
                    if len(unass) == 1:
                        v = unass[0]
                        if colors[v] == 0:
                            return False
                        if colors[v] == -1:
                            if not assign_vertex(v, 1):
                                return False
                            changed = True
                    elif len(unass) == 0:
                        if cnt1 == 0:
                            return False
                elif cnt1 == 4:
                    if len(unass) == 1:
                        v = unass[0]
                        if colors[v] == 1:
                            return False
                        if colors[v] == -1:
                            if not assign_vertex(v, 0):
                                return False
                            changed = True
                    elif len(unass) == 0:
                        if cnt0 == 0:
                            return False
        return True
    
    # Stack-based DFS to avoid recursion limit issues
    # Each stack frame: (pos, choice) where choice is 0 or 1
    # We'll use iterative DFS with explicit state saving
    
    # Actually, let's use recursion with a higher limit since n <= 26
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    def dfs(pos):
        if pos == n:
            return True
        
        v = order_to_vert[pos]
        
        # Try color 0
        if assign_vertex(v, 0):
            if propagate():
                if dfs(pos + 1):
                    return True
            unassign_vertex(v)
        
        # Try color 1
        if assign_vertex(v, 1):
            if propagate():
                if dfs(pos + 1):
                    return True
            unassign_vertex(v)
        
        return False
    
    result = dfs(0)
    sys.setrecursionlimit(old_limit)
    return result

def minimize_hypergraph(n, edges, deadline):
    """Minimize a non-2-colorable hypergraph by removing redundant edges."""
    current = list(edges)
    m = len(current)
    
    # Multiple passes of edge removal
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        if m == 0:
            break
        
        removed_any = False
        # Try removing each edge
        new_edges = []
        for i, e in enumerate(current):
            if time.time() > deadline - 15:
                new_edges.append(e)
                continue
            
            # Check if H \ {e} is still non-2-colorable
            h_minus = current[:i] + current[i+1:]
            if not h_minus:
                new_edges.append(e)
            else:
                if is_2_colorable(n, h_minus):
                    # Still 2-colorable without e, so e is needed
                    new_edges.append(e)
                else:
                    # Non-2-colorable without e, so e is redundant
                    removed_any = True
        
        current = new_edges
        m = len(current)
        
        if not removed_any:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_m = 200
    best_n = 0
    best_edges = []
    
    random.seed(42)
    
    # Known small constructions to try first
    # Try various n values and random edge sets
    
    for n in range(10, 17):
        if time.time() > deadline - 30:
            break
        
        # Number of trials for this n
        max_trials = 30
        trials_done = 0
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            trials_done += 1
            
            # Start with a dense random hypergraph
            # m_start should be high enough to likely be non-2-colorable
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Try different starting densities
            m_start = min(40 + trial * 2, len(all_possible))
            if m_start > len(all_possible):
                m_start = len(all_possible)
            
            edges = [list(e) for e in all_possible[:m_start]]
            
            # Check if non-2-colorable
            if not is_2_colorable(n, edges):
                # It is non-2-colorable. Now minimize.
                min_edges = minimize_hypergraph(n, edges, deadline)
                min_m = len(min_edges)
                
                if min_m < best_m:
                    best_m = min_m
                    best_n = n
                    best_edges = min_edges
                    
                    # If we find something very good, we can stop early
                    if best_m <= 35:
                        break
        
        if best_m <= 35:
            break
    
    # Fallback: use complete hypergraph on small n
    if best_m > 126:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_m = len(best_edges)
    
    return {'n': best_n, 'edges': best_edges}
