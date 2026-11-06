import time
import random
import itertools
from collections import deque

def _now():
    return time.time()

def _is_not_2_colorable(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring (Property B holds).
    Uses backtracking with unit propagation.
    Returns True if no 2-coloring exists.
    """
    if not edges:
        return False
    
    # Precompute edge indices for each vertex
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    m = len(edges)
    
    def propagate(assign, edge_state):
        """
        assign: list of -1, 0, 1
        edge_state: list of (count0, count1) for each edge
        Returns (success, assign, edge_state, forced_count)
        """
        queue = deque()
        # Initialize queue with all unassigned vertices? No, unit propagation works on edges.
        # An edge forces a value if it has 4 of one color and 1 unassigned.
        # We need to check edges after each assignment.
        
        # To be efficient, we can just check all edges in a loop until stable.
        # n <= 26, m <= 200. This is fast enough.
        
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0, c1 = edge_state[i]
                # Count unassigned
                unassigned_in_edge = 5 - c0 - c1
                if unassigned_in_edge == 0:
                    if c0 == 5 or c1 == 5:
                        return False, assign, edge_state, 0
                    continue
                
                if c0 == 4 and unassigned_in_edge == 1:
                    # The unassigned vertex in this edge MUST be 1
                    for v in e:
                        if assign[v] == -1:
                            assign[v] = 1
                            edge_state[i] = (c0, c1 + 1)
                            # Update other edges incident to v
                            for j in vertex_edges[v]:
                                if j != i:
                                    c0j, c1j = edge_state[j]
                                    edge_state[j] = (c0j, c1j + 1)
                            changed = True
                            break
                elif c1 == 4 and unassigned_in_edge == 1:
                    # The unassigned vertex in this edge MUST be 0
                    for v in e:
                        if assign[v] == -1:
                            assign[v] = 0
                            edge_state[i] = (c0 + 1, c1)
                            # Update other edges incident to v
                            for j in vertex_edges[v]:
                                if j != i:
                                    c0j, c1j = edge_state[j]
                                    edge_state[j] = (c0j + 1, c1j)
                            changed = True
                            break
        
        return True, assign, edge_state, 0

    def backtrack(assign, edge_state):
        # Check for contradictions
        for i in range(m):
            c0, c1 = edge_state[i]
            if c0 == 5 or c1 == 5:
                return True # Failed to find coloring, so this branch is dead (Property B holds for this branch)
        
        # Find unassigned vertex
        v = -1
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned, no contradictions. So a valid coloring exists.
            return False
        
        # Branch
        # Try 0
        assign_copy = assign[:]
        state_copy = [s for s in edge_state]
        assign_copy[v] = 0
        state_copy[v]  # dummy
        # Update state for v
        for j in vertex_edges[v]:
            c0, c1 = state_copy[j]
            state_copy[j] = (c0 + 1, c1)
        
        ok, assign_new, state_new, _ = propagate(assign_copy, state_copy)
        if not ok:
            # Contradiction found, so this branch is dead.
            # Wait, propagate returns False on contradiction.
            # If propagate fails, it means no coloring extends this assignment.
            # So we return True (Property B holds for this sub-branch).
            return True
        
        if backtrack(assign_new, state_new):
            return True
        
        # Try 1
        assign_copy2 = assign[:]
        state_copy2 = [s for s in edge_state]
        assign_copy2[v] = 1
        for j in vertex_edges[v]:
            c0, c1 = state_copy2[j]
            state_copy2[j] = (c0, c1 + 1)
        
        ok2, assign_new2, state_new2, _ = propagate(assign_copy2, state_copy2)
        if not ok2:
            return True
        
        if backtrack(assign_new2, state_new2):
            return True
        
        return False

    assign = [-1] * n
    edge_state = [(0, 0) for _ in range(m)]
    
    # Initial propagation (none needed since all 0)
    return backtrack(assign, edge_state)

def construct():
    start_time = _now()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_count = float('inf')
    best_n = 0
    
    # We know the answer is 51. Let's try to find a 51-edge construction.
    # The known construction is on 11 vertices? Or more?
    # Actually, the smallest 5-uniform non-2-colorable hypergraph is on 11 vertices?
    # No, the first one was found with more vertices.
    # Let's try random search on n=11, 12, 13, 14.
    
    for n in [11, 12, 13, 14, 15]:
        all_possible = list(itertools.combinations(range(n), 5))
        
        for attempt in range(100):
            if _now() > deadline:
                break
            
            random.seed(attempt * 100 + n)
            
            # Start with a dense random hypergraph
            # Probability of an edge being present.
            # To be non-2-colorable, it needs to be fairly dense.
            # For n=11, total edges = 462.
            # Let's start with 200 edges.
            num_edges = min(len(all_possible), 200)
            edges = random.sample(all_possible, num_edges)
            edges = [list(e) for e in edges]
            
            # Check if it is non-2-colorable
            if not _is_not_2_colorable(n, edges):
                continue
            
            # It is non-2-colorable. Now minimize.
            # Greedy deletion.
            random.shuffle(edges)
            
            i = 0
            while i < len(edges) and _now() < deadline:
                # Try to remove edges[i]
                new_edges = edges[:i] + edges[i+1:]
                if _is_not_2_colorable(n, new_edges):
                    edges = new_edges
                else:
                    i += 1
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
                best_n = n
                if best_count <= 51:
                    # We found a very good one. We can break early if we want, but let's try a bit more.
                    pass
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Complete hypergraph on 9 vertices (126 edges)
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}

# EVOLVE-BLOCK-START
def construct():
    # The complete 5-uniform hypergraph on 9 vertices: any 2-colouring has 5 vertices of one colour.
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
