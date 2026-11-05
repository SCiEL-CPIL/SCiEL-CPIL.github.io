"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_non_2colorable_fast(n, edges, deadline):
    """Fast check if hypergraph is NOT 2-colorable using optimized backtracking."""
    if time.time() > deadline:
        return None
    edge_set = [tuple(sorted(e)) for e in edges]
    color = [-1] * n
    
    # Build incidence: for each vertex, list of edge indices
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_edges[v].append(i)
    
    # For each edge, track count of 0s, 1s, and unassigned
    edge_state = []
    for e in edge_set:
        c0 = c1 = un = 0
        for v in e:
            if color[v] == 0:
                c0 += 1
            elif color[v] == 1:
                c1 += 1
            else:
                un += 1
        edge_state.append([c0, c1, un])
    
    def propagate():
        """Unit propagation with edge state tracking."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                c0, c1, un = edge_state[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 1
                            edge_state[i][1] += 1
                            edge_state[i][2] -= 1
                            changed = True
                            # Update other edges containing v
                            for j in vert_edges[v]:
                                if j != i:
                                    edge_state[j][1] += 1
                                    edge_state[j][2] -= 1
                elif c1 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 0
                            edge_state[i][0] += 1
                            edge_state[i][2] -= 1
                            changed = True
                            for j in vert_edges[v]:
                                if j != i:
                                    edge_state[j][0] += 1
                                    edge_state[j][2] -= 1
        return True
    
    # Find unassigned vertex with most constraints
    def get_next_vertex():
        best_v = -1
        best_count = -1
        for v in range(n):
            if color[v] == -1:
                # Count unassigned neighbors in edges
                count = len(vert_edges[v])
                if count > best_count:
                    best_count = count
                    best_v = v
        return best_v
    
    def backtrack():
        if time.time() > deadline:
            return 'timeout'
        v = get_next_vertex()
        if v == -1:
            # All assigned - check if valid 2-coloring
            for i, e in enumerate(edge_set):
                c0, c1, un = edge_state[i]
                if un == 0 and (c0 == 5 or c1 == 5):
                    return False
            return True
        
        saved_color = color[:]
        saved_state = [s[:] for s in edge_state]
        
        for c in [0, 1]:
            color[v] = c
            # Update edge states
            for j in vert_edges[v]:
                if c == 0:
                    edge_state[j][0] += 1
                else:
                    edge_state[j][1] += 1
                edge_state[j][2] -= 1
            
            if propagate():
                result = backtrack()
                if result == 'timeout':
                    color[:] = saved_color
                    edge_state[:] = saved_state
                    return 'timeout'
                if result == True:
                    color[:] = saved_color
                    edge_state[:] = saved_state
                    return True
            color[:] = saved_color
            edge_state[:] = saved_state
        return False
    
    if not propagate():
        return True  # Already contradictory, so not 2-colorable
    
    result = backtrack()
    if result == 'timeout':
        return None
    if result == True:
        return False
    else:
        return True

def construct():
    deadline = time.time() + 145
    
    # Try to find a small non-2-colorable 5-uniform hypergraph
    # Start with known good candidates
    
    best_edges = None
    best_n = 26
    
    # Strategy: Try n=10 to n=16 with various edge counts
    # Use a smarter initialization
    
    for n in range(10, 17):
        if time.time() > deadline:
            break
        all_possible = list(combinations(range(n), 5))
        total_edges = len(all_possible)
        
        # Try different target edge counts
        for target_e in range(30, min(60, total_edges + 1), 5):
            if time.time() > deadline:
                break
            
            # Try multiple random samples
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                sample = random.sample(all_possible, target_e)
                sample = [list(e) for e in sample]
                
                is_non_2col = _is_non_2colorable_fast(n, sample, deadline)
                if is_non_2col is True:
                    if best_edges is None or len(sample) < len(best_edges):
                        best_edges = sample
                        best_n = n
                        # Try to reduce further
                        current_edges = sample[:]
                        current_n = n
                        for _ in range(20):
                            if time.time() > deadline:
                                break
                            if not current_edges:
                                break
                            # Try removing a random edge
                            idx = random.randint(0, len(current_edges) - 1)
                            temp_edges = current_edges[:idx] + current_edges[idx+1:]
                            is_non_2col2 = _is_non_2colorable_fast(current_n, temp_edges, deadline)
                            if is_non_2col2 is True:
                                current_edges = temp_edges
                                if best_edges is None or len(current_edges) < len(best_edges):
                                    best_edges = current_edges[:]
                                    best_n = current_n
                    break  # Found one for this n, target_e
    
    # Fallback: if nothing found, use a denser construction
    if best_edges is None:
        # Use K_9^5 minus some edges as fallback
        n = 9
        all_edges = list(combinations(range(n), 5))
        # Remove edges greedily
        current_edges = [list(e) for e in all_edges]
        random.seed(42)
        edge_indices = list(range(len(current_edges)))
        random.shuffle(edge_indices)
        
        for idx in edge_indices:
            if time.time() > deadline:
                break
            temp_edges = current_edges[:idx] + current_edges[idx+1:]
            is_non_2col = _is_non_2colorable_fast(n, temp_edges, deadline)
            if is_non_2col is True:
                current_edges = temp_edges
        
        best_edges = current_edges
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
