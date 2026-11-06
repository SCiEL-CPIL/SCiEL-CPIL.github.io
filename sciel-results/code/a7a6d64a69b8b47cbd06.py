"""Construct a graph with no regular induced subgraph on 6+ vertices."""
import time
import random
from itertools import combinations

DEADLINE = 110  # seconds, safety margin before 120s limit

def _is_regular_subgraph(vertices, adj_matrix, n):
    """Check if the induced subgraph on 'vertices' is regular."""
    k = len(vertices)
    if k < 6:
        return False
    # Count degrees within the subgraph
    vset = set(vertices)
    degrees = []
    for v in vertices:
        deg = 0
        for u in vertices:
            if u != v and adj_matrix[v][u]:
                deg += 1
        degrees.append(deg)
    # Check if all degrees are equal
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _check_new_subsets_with_vertex(new_v, adj_matrix, n):
    """Check all subsets of size >= 6 that include new_v for regularity."""
    other_vertices = list(range(new_v))  # only previous vertices
    # We need to check subsets of size 6, 7, ..., new_v+1 that include new_v
    # The subset is {new_v} union S where S is a subset of other_vertices
    # |S| must be at least 5 (so total >= 6)
    max_size = len(other_vertices)
    for size_s in range(5, max_size + 1):
        if size_s + 1 < 6:
            continue
        for S in combinations(other_vertices, size_s):
            subset = list(S) + [new_v]
            if _is_regular_subgraph(subset, adj_matrix, n):
                return False
    return True

def construct():
    start_time = time.time()
    
    # Try to build a graph on n vertices, starting from n=21 and working down
    for n_target in range(21, 4, -1):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Random restarts for this n
        best_graph = None
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            random.seed(attempt * 1000 + n_target)
            
            # Build graph incrementally
            adj_matrix = [[False] * n_target for _ in range(n_target)]
            valid = True
            
            for v in range(n_target):
                if time.time() - start_time > DEADLINE - 30:
                    valid = False
                    break
                
                # Decide edges from v to previous vertices
                # Use random but biased towards creating irregularity
                for u in range(v):
                    # Random edge with some probability
                    if random.random() < 0.5:
                        adj_matrix[v][u] = True
                        adj_matrix[u][v] = True
                
                # Check if adding v creates any regular induced subgraph of size >= 6
                if v + 1 >= 6:
                    if not _check_new_subsets_with_vertex(v, adj_matrix, n_target):
                        valid = False
                        break
            
            if valid:
                # Verify the entire graph is valid
                all_valid = True
                for size in range(6, n_target + 1):
                    if time.time() - start_time > DEADLINE - 30:
                        all_valid = False
                        break
                    for subset in combinations(range(n_target), size):
                        if _is_regular_subgraph(list(subset), adj_matrix, n_target):
                            all_valid = False
                            break
                    if not all_valid:
                        break
                
                if all_valid:
                    # Build edge list
                    edges = []
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            if adj_matrix[i][j]:
                                edges.append([i, j])
                    best_graph = {'n': n_target, 'edges': edges}
                    break  # Found a valid graph for this n, no need to try smaller
        
        if best_graph is not None:
            return best_graph
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
