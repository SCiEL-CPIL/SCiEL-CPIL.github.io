import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges):
    """
    Fast check: return True if hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation on bitmasks.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge vertex lists and bitmasks
    edge_verts = [list(e) for e in edges]
    edge_masks = [0] * m
    for i, e in enumerate(edge_verts):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
    
    # Vertex to edges mapping
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_verts):
        for v in e:
            vert_to_edges[v].append(i)
    
    # Colors: -1 unassigned, 0, 1
    colors = [-1] * n
    # For each edge, count of color 0 and color 1 assigned
    cnt0 = [0] * m
    cnt1 = [0] * m
    
    # Precompute for each edge the list of vertices for quick access
    # Unit propagation: if an edge has 4 of one color and 1 unassigned, force that vertex
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0 = cnt0[i]
                c1 = cnt1[i]
                if c0 + c1 == 5:
                    if c0 == 5 or c1 == 5:
                        return False
                    continue
                e = edge_verts[i]
                unassigned = [v for v in e if colors[v] == -1]
                if not unassigned:
                    if c0 == 5 or c1 == 5:
                        return False
                    continue
                # If all assigned are same color, force unassigned to other
                if c1 == 0 and c0 > 0:
                    for v in unassigned:
                        colors[v] = 1
                        cnt1[i] += 1
                        for j in vert_to_edges[v]:
                            if j != i:
                                cnt1[j] += 1
                        changed = True
                elif c0 == 0 and c1 > 0:
                    for v in unassigned:
                        colors[v] = 0
                        cnt0[i] += 1
                        for j in vert_to_edges[v]:
                            if j != i:
                                cnt0[j] += 1
                        changed = True
        return True
    
    # Backtracking with variable ordering heuristic
    # Pick vertex involved in most "tight" edges (edges with 3+ assigned)
    def pick_vertex():
        best_v = -1
        best_score = -1
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for j in vert_to_edges[v]:
                c0 = cnt0[j]
                c1 = cnt1[j]
                if c0 + c1 >= 3:
                    score += 10
                elif c0 + c1 == 2:
                    score += 3
                else:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        return best_v
    
    def backtrack():
        if not propagate():
            return False
        
        # Check if all assigned
        all_assigned = True
        for v in range(n):
            if colors[v] == -1:
                all_assigned = False
                break
        if all_assigned:
            # Verify no monochromatic edge
            for i in range(m):
                if cnt0[i] == 5 or cnt1[i] == 5:
                    return False
            return True
        
        v = pick_vertex()
        if v == -1:
            # No unassigned vertex but not all edges checked? Shouldn't happen
            for i in range(m):
                if cnt0[i] == 5 or cnt1[i] == 5:
                    return False
            return True
        
        # Try color 0
        old_colors = colors[:]
        old_cnt0 = cnt0[:]
        old_cnt1 = cnt1[:]
        
        colors[v] = 0
        for j in vert_to_edges[v]:
            cnt0[j] += 1
        if backtrack():
            return True
        
        # Restore
        colors[:] = old_colors
        cnt0[:] = old_cnt0
        cnt1[:] = old_cnt1
        
        # Try color 1
        colors[v] = 1
        for j in vert_to_edges[v]:
            cnt1[j] += 1
        if backtrack():
            return True
        
        # Restore for caller
        colors[:] = old_colors
        cnt0[:] = old_cnt0
        cnt1[:] = old_cnt1
        
        return False
    
    return not backtrack()


def _greedy_delete(n, edges, deadline):
    """Greedy edge deletion: try removing each edge, keep if still non-2-colorable."""
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        i = 0
        while i < len(edges) and time.time() < deadline:
            if time.time() > deadline:
                break
            candidate = edges[:i] + edges[i+1:]
            if _is_non_2colorable(n, candidate):
                edges.pop(i)
                removed_any = True
            else:
                i += 1
    return edges


def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_m = 126
    best_graph = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    # Try different vertex counts, starting from complete hypergraphs
    # n=10: C(10,5)=252, n=11: C(11,5)=462, n=12: C(12,5)=792
    for n in [10, 11, 12, 9]:
        if time.time() > deadline - 10:
            break
        
        all_edges = [list(e) for e in combinations(range(n), 5)]
        total = len(all_edges)
        
        # Try several random orderings for greedy deletion
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            
            edges = all_edges[:]
            random.shuffle(edges)
            
            # First, check if the complete hypergraph is non-2-colorable
            # (it should be for n>=9, but let's verify for smaller edge sets)
            # Actually, let's just start greedy deletion directly
            
            edges = _greedy_delete(n, edges, deadline - 3)
            
            if len(edges) < best_m:
                best_m = len(edges)
                best_graph = {'n': n, 'edges': edges}
                
                if best_m <= 35:
                    break
        
        if best_m <= 35:
            break
    
    # Also try random sparse initializations for larger n
    for n in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        all_edges = [list(e) for e in combinations(range(n), 5)]
        total = len(all_edges)
        
        for density in [0.15, 0.1, 0.08]:
            if time.time() > deadline - 5:
                break
            
            init_m = int(total * density)
            if init_m >= best_m:
                continue
            if init_m < 25:
                continue
                
            edges = [list(e) for e in random.sample(all_edges, init_m)]
            
            if not _is_non_2colorable(n, edges):
                continue
            
            edges = _greedy_delete(n, edges, deadline - 3)
            
            if len(edges) < best_m:
                best_m = len(edges)
                best_graph = {'n': n, 'edges': edges}
    
    return best_graph
