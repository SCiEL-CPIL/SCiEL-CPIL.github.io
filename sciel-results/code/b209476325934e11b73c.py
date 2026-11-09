import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """Exact check: returns True iff the hypergraph IS 2-colorable."""
    if not edges:
        return True
    
    m = len(edges)
    # Precompute edge vertex lists
    edge_verts = [list(e) for e in edges]
    
    # For unit propagation, track for each edge:
    # - count of vertices assigned 0
    # - count of vertices assigned 1
    # - list of unassigned vertices
    
    # State: assignment array (-1 unassigned, 0 or 1)
    assignment = [-1] * n
    unassigned_count = n
    
    # Edge state: for each edge, track zero_count, one_count, unassigned_verts
    edge_state = []
    for e in edge_verts:
        edge_state.append({'zeros': 0, 'ones': 0, 'unassigned': list(e)})
    
    def propagate(assignment, edge_state, unassigned_count):
        """Unit propagation. Returns None if contradiction, else (assignment, edge_state, unassigned_count)."""
        changed = True
        while changed:
            changed = False
            for i, es in enumerate(edge_state):
                if len(es['unassigned']) == 0:
                    # Edge fully assigned: check if monochromatic
                    if es['zeros'] == 5 or es['ones'] == 5:
                        return None
                    continue
                # Check if forced: if zeros == 4, remaining must be 1; if ones == 4, remaining must be 0
                if es['zeros'] == 4:
                    # All remaining must be 1
                    for v in es['unassigned']:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            unassigned_count -= 1
                            es['unassigned'].remove(v)
                            es['ones'] += 1
                            changed = True
                elif es['ones'] == 4:
                    # All remaining must be 0
                    for v in es['unassigned']:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            unassigned_count -= 1
                            es['unassigned'].remove(v)
                            es['zeros'] += 1
                            changed = True
                # Also check: if zeros + unassigned == 5 and zeros == 5-1, forced
                # Already handled above
        return assignment, edge_state, unassigned_count
    
    def backtrack(assignment, edge_state, unassigned_count):
        # Propagate
        result = propagate(assignment, edge_state, unassigned_count)
        if result is None:
            return False
        assignment, edge_state, unassigned_count = result
        
        if unassigned_count == 0:
            # All assigned, check no monochromatic edge
            for es in edge_state:
                if es['zeros'] == 5 or es['ones'] == 5:
                    return False
            return True
        
        # Choose next unassigned vertex
        # Heuristic: pick vertex in edge with most constrained state
        best_v = -1
        best_score = -1
        for v in range(n):
            if assignment[v] != -1:
                continue
            # Score: number of edges where this vertex is critical
            score = 0
            for es in edge_state:
                if v in es['unassigned']:
                    if es['zeros'] >= 3 or es['ones'] >= 3:
                        score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            return True
        
        # Try 0
        assignment[best_v] = 0
        new_es = []
        for es in edge_state:
            new_es.append({'zeros': es['zeros'], 'ones': es['ones'], 'unassigned': list(es['unassigned'])})
        for es in new_es:
            if best_v in es['unassigned']:
                es['unassigned'].remove(best_v)
                es['zeros'] += 1
        if backtrack(assignment, new_es, unassigned_count - 1):
            return True
        
        # Try 1
        assignment[best_v] = 1
        new_es = []
        for es in edge_state:
            new_es.append({'zeros': es['zeros'], 'ones': es['ones'], 'unassigned': list(es['unassigned'])})
        for es in new_es:
            if best_v in es['unassigned']:
                es['unassigned'].remove(best_v)
                es['ones'] += 1
        if backtrack(assignment, new_es, unassigned_count - 1):
            return True
        
        assignment[best_v] = -1
        return False
    
    return backtrack(assignment, edge_state, n)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    best_edges = None
    best_count = float('inf')
    
    # Try to find small non-2-colorable hypergraphs
    # Start with small edge counts and increase
    # For each n, try random sets of edges
    
    for n in range(9, 15):
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        num_all = len(all_edges)
        
        # Try increasing edge counts
        for target_count in range(20, min(60, num_all + 1)):
            if time.time() > deadline:
                break
            
            # Try multiple random samples
            max_tries = 50
            found = False
            for trial in range(max_tries):
                if time.time() > deadline:
                    break
                
                # Random sample of edges
                sample = random.sample(all_edges, min(target_count, num_all))
                
                # Check if non-2-colorable
                if not _is_2_colorable(n, sample):
                    # Found a non-2-colorable hypergraph with target_count edges
                    if target_count < best_count:
                        best_count = target_count
                        best_edges = [list(e) for e in sample]
                        found = True
                    break  # Found one for this count, move to next count
        
        # If we found a good one, try to reduce further
        if best_edges and best_count <= 30:
            # Try to remove edges while maintaining non-2-colorability
            reduced = True
            while reduced and time.time() < deadline:
                reduced = False
                current_edges = best_edges
                for i in range(len(current_edges)):
                    if time.time() > deadline:
                        break
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if test_edges and not _is_2_colorable(n, test_edges):
                        best_edges = test_edges
                        best_count = len(test_edges)
                        reduced = True
                        break
    
    # Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable)
    if best_edges is None:
        fallback_edges = [list(e) for e in combinations(range(9), 5)]
        return {'n': 9, 'edges': fallback_edges}
    
    return {'n': n, 'edges': best_edges}
