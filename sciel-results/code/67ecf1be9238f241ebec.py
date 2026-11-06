import time
import random
from itertools import combinations
import sys

def _is_not_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is NOT 2-colorable (i.e., no proper 2-coloring exists).
    Returns True if NOT 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    m = len(edges)
    
    # Convert edges to bitmasks and build incidence structure
    edge_masks = []
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        for v in e:
            vertex_edges[v].append(i)
    
    # State: color[v] in {-1, 0, 1}, c0[i] = count of 0s in edge i, c1[i] = count of 1s
    color = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    
    # Unit propagation: if an edge has 4 of one color and 0 of the other,
    # the remaining uncolored vertex must be the other color.
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    # All 5 must be 0, but one is uncolored -> contradiction unless we force it
                    # Actually, if c0[i]==4 and c1[i]==0, the 5th vertex must be 0
                    # But that would make c0[i]==5, which is a contradiction (monochromatic)
                    # Wait: proper 2-coloring means no edge is monochromatic.
                    # So if an edge has 4 zeros and 0 ones, the 5th must be 1 to avoid mono.
                    # But then c1[i] becomes 1, c0[i] stays 4. That's fine.
                    # Let me re-read: "no colouring of the vertices leaves every edge with both colours"
                    # This means we want NO coloring where every edge has both colors.
                    # So we're looking for a coloring where SOME edge is monochromatic.
                    # Actually, "not 2-colorable" means there is no proper 2-coloring.
                    # A proper 2-coloring is one where every edge has both colors (no mono edge).
                    # So "not 2-colorable" = every coloring has at least one mono edge.
                    
                    # For backtracking: we try to find a proper 2-coloring.
                    # If we can't, the hypergraph is not 2-colorable.
                    
                    # In unit propagation for finding a proper 2-coloring:
                    # If an edge has 4 vertices colored 0, the 5th MUST be 1 (to avoid mono-0).
                    # If an edge has 4 vertices colored 1, the 5th MUST be 0 (to avoid mono-1).
                    # If an edge has 5 vertices colored 0 or 5 colored 1, contradiction.
                    
                    # So: c0[i]==4, c1[i]==0 -> uncolored vertex must be 1
                    #     c1[i]==4, c0[i]==0 -> uncolored vertex must be 0
                    #     c0[i]==5 or c1[i]==5 -> contradiction
                    
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                                if c1[other_e] == 5:
                                    return False
                            changed = True
                            break
                elif c1[i] == 4 and c0[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                                if c0[other_e] == 5:
                                    return False
                            changed = True
                            break
        return True
    
    def backtrack():
        # Check for immediate contradictions
        for i in range(m):
            if c0[i] == 5 or c1[i] == 5:
                return False
        
        # Find the most constrained edge (fewest uncolored vertices)
        best_edge = -1
        best_uncolored = 6
        for i in range(m):
            uncolored = 5 - c0[i] - c1[i]
            if uncolored < best_uncolored:
                best_uncolored = uncolored
                best_edge = i
        
        if best_edge == -1:
            return True  # All edges are fully colored and no contradictions
        
        if best_uncolored == 0:
            # Edge is fully colored but not mono (since we checked c0!=5 and c1!=5)
            # This edge is satisfied. But we need to check all edges.
            # Actually, if best_uncolored==0 for all edges, we're done.
            # Let's verify no edge is mono
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            return True
        
        # Get uncolored vertices in best_edge
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
        
        if not candidates:
            return True
        
        # Try each candidate with color 0, then color 1
        for v in candidates:
            saved_color = color[v]
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            # Try color 0
            color[v] = 0
            c0[best_edge] += 1
            conflict = False
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    conflict = True
                    break
            
            if not conflict and propagate():
                if backtrack():
                    return True
            
            # Restore
            color[v] = saved_color
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            color[v] = 1
            c1[best_edge] += 1
            conflict = False
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    conflict = True
                    break
            
            if not conflict and propagate():
                if backtrack():
                    return True
            
            # Restore
            color[v] = saved_color
            c0[:] = saved_c0
            c1[:] = saved_c1
        
        return False
    
    # Try to find a proper 2-coloring
    if propagate():
        if backtrack():
            return False  # Found a proper 2-coloring, so it IS 2-colorable
    return True  # No proper 2-coloring found, so it is NOT 2-colorable

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Try different n values
    for n in [11, 12, 10, 9, 13]:
        if time.time() > deadline - 15:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        # Quick check: is the full hypergraph not 2-colorable?
        if not _is_not_2_colorable(n, all_possible):
            continue
        
        # Phase 1: Greedy removal with multiple random orderings
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            curr = [list(e) for e in all_possible]
            order = list(range(len(curr)))
            random.shuffle(order)
            
            removed_any = True
            while removed_any and time.time() < deadline - 25:
                removed_any = False
                for i in order:
                    if time.time() > deadline - 25:
                        break
                    if i >= len(curr):
                        continue
                    
                    new_edges = curr[:i] + curr[i+1:]
                    if _is_not_2_colorable(n, new_edges):
                        curr = new_edges
                        order = list(range(len(curr)))
                        random.shuffle(order)
                        removed_any = True
                        break
            
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
        
        # Phase 2: Local search to reduce edge count further
        if best_edges and time.time() < deadline - 10:
            curr_edges = best_edges[:]
            curr_n = best_n
            all_edges_set = set(tuple(e) for e in all_possible)
            curr_set = set(tuple(e) for e in curr_edges)
            
            T = 5.0
            cooling = 0.97
            steps = 0
            max_steps = 5000
            
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1
                move_type = random.random()
                new_set = set(curr_set)
                
                if move_type < 0.4:
                    if not curr_set:
                        continue
                    e_to_remove = random.choice(list(curr_set))
                    new_set.remove(e_to_remove)
                elif move_type < 0.7:
                    available = list(all_edges_set - curr_set)
                    if not available:
                        continue
                    e_to_add = random.choice(available)
                    new_set.add(e_to_add)
                else:
                    if not curr_set:
                        continue
                    e_to_remove = random.choice(list(curr_set))
                    available = list(all_edges_set - curr_set)
                    if not available:
                        continue
                    e_to_add = random.choice(available)
                    new_set.remove(e_to_remove)
                    new_set.add(e_to_add)
                
                new_edges_list = [list(e) for e in new_set]
                is_valid = _is_not_2_colorable(curr_n, new_edges_list)
                
                if is_valid:
                    delta = len(new_set) - len(curr_set)
                    if delta < 0:
                        curr_set = new_set
                        if len(curr_set) < best_m:
                            best_m = len(curr_set)
                            best_edges = [list(e) for e in curr_set]
                            best_n = curr_n
                    else:
                        if random.random() < (2.718281828 ** (-delta / T)):
                            curr_set = new_set
                # If not valid, reject
                
                T *= cooling
        
        if best_m <= 51:
            break
    
    if best_edges is None:
        # Fallback: all 5-subsets of 9 vertices
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
