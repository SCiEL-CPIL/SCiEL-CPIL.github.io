"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _is_not_2_colorable(n, edges):
    """Check if hypergraph has no proper 2-coloring via backtracking with unit propagation."""
    # Precompute: for each edge, list of vertices
    edge_set = set()
    for e in edges:
        edge_set.add(frozenset(e))
    
    # Map vertex -> list of edge indices containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # Edge state: 0 = unassigned, 1 = satisfied (has both colors), 2 = violated (all same color)
    # For unit propagation: track counts of each color per edge
    edge_pos_count = [0] * len(edges)  # count of vertices colored 1 in edge
    edge_neg_count = [0] * len(edges)  # count of vertices colored 0 in edge
    edge_unassigned = [5] * len(edges)  # count of unassigned vertices in edge
    edge_satisfied = [False] * len(edges)
    
    color = [-1] * n  # -1 = unassigned, 0 = color A, 1 = color B
    
    def propagate(start_color, start_vertex):
        """Unit propagation: assign colors and check for contradictions."""
        stack = [(start_color, start_vertex)]
        while stack:
            c, v = stack.pop()
            if color[v] != -1:
                if color[v] != c:
                    return False
                continue
            color[v] = c
            for ei in vertex_edges[v]:
                if edge_satisfied[ei]:
                    continue
                if c == 1:
                    edge_pos_count[ei] += 1
                else:
                    edge_neg_count[ei] += 1
                edge_unassigned[ei] -= 1
                # Check if edge is satisfied (has both colors)
                if edge_pos_count[ei] > 0 and edge_neg_count[ei] > 0:
                    edge_satisfied[ei] = True
                # Check if edge is violated (all assigned vertices same color, and no unassigned left)
                elif edge_unassigned[ei] == 0:
                    if edge_pos_count[ei] == 0 or edge_neg_count[ei] == 0:
                        # Edge has only one color -> violated
                        return False
                # Unit propagation: if all but one vertex in edge have same color,
                # the last must have the other color
                elif edge_unassigned[ei] == 1:
                    if edge_pos_count[ei] > 0 and edge_neg_count[ei] == 0:
                        # All assigned are color 1, last must be color 0
                        for w in edges[ei]:
                            if color[w] == -1:
                                if not propagate(0, w):
                                    return False
                                break
                    elif edge_neg_count[ei] > 0 and edge_pos_count[ei] == 0:
                        # All assigned are color 0, last must be color 1
                        for w in edges[ei]:
                            if color[w] == -1:
                                if not propagate(1, w):
                                    return False
                                break
        return True
    
    def backtrack(vertex_idx):
        """Backtracking search."""
        if vertex_idx >= n:
            # All vertices assigned, check if all edges are satisfied
            for ei in range(len(edges)):
                if not edge_satisfied[ei]:
                    if edge_pos_count[ei] > 0 and edge_neg_count[ei] > 0:
                        continue
                    else:
                        return False
            return True
        
        # Try both colors
        for c in [0, 1]:
            # Save state
            color_backup = color[:]
            pos_backup = edge_pos_count[:]
            neg_backup = edge_neg_count[:]
            unasg_backup = edge_unassigned[:]
            sat_backup = edge_satisfied[:]
            
            if propagate(c, vertex_idx):
                if backtrack(vertex_idx + 1):
                    return True
            
            # Restore state
            color[:] = color_backup
            edge_pos_count[:] = pos_backup
            edge_neg_count[:] = neg_backup
            edge_unassigned[:] = unasg_backup
            edge_satisfied[:] = sat_backup
        
        return False
    
    # Try both initial colorings (symmetry breaking: vertex 0 = color 0)
    if not propagate(0, 0):
        return True
    result = backtrack(1)
    return not result

def construct():
    """Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges."""
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Known small constructions to try
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5)=126 edges
    # We need to find smaller ones
    
    # Strategy: try known small hypergraphs that are not 2-colorable
    # For 5-uniform, the minimum is known to be around 51 edges
    
    # Try a few candidate constructions
    candidates = []
    
    # Candidate 1: Complete 5-graph on 9 vertices (126 edges) - fallback
    edges_9 = [list(e) for e in itertools.combinations(range(9), 5)]
    candidates.append((9, edges_9))
    
    # Candidate 2: Try to construct smaller hypergraphs
    # Use a random search approach with pruning
    
    best_n = 9
    best_edges = edges_9
    best_count = len(best_edges)
    
    # Try n=10 to n=15 with random edge sets
    import random
    random.seed(42)
    
    for n in range(9, 16):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random constructions
        all_edges = list(itertools.combinations(range(n), 5))
        
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            # Start with a base set and add random edges
            # Try to find a small non-2-colorable hypergraph
            
            # Use a greedy approach: start with random edges, check if 2-colorable,
            # if yes, add more edges until not 2-colorable
            
            edge_set = set()
            # Add some random edges initially
            num_initial = min(20, len(all_edges))
            initial_edges = random.sample(all_edges, num_initial)
            for e in initial_edges:
                edge_set.add(frozenset(e))
            
            # Check if already not 2-colorable (Valid)
            current_edges = [list(e) for e in edge_set]
            if _is_not_2_colorable(n, current_edges):
                if len(current_edges) < best_count:
                    best_n = n
                    best_edges = current_edges
                    best_count = len(current_edges)
                continue
            
            # If invalid, add more edges until not 2-colorable
            remaining = [e for e in all_edges if frozenset(e) not in edge_set]
            random.shuffle(remaining)
            
            for e in remaining:
                if time.time() > deadline - 30:
                    break
                edge_set.add(frozenset(e))
                current_edges = [list(x) for x in edge_set]
                if _is_not_2_colorable(n, current_edges):
                    if len(current_edges) < best_count:
                        best_n = n
                        best_edges = current_edges
                        best_count = len(current_edges)
                    break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
