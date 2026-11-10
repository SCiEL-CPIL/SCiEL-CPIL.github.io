"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, deadline=None):
    """
    Check if the 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False if not.
    Uses backtracking with unit propagation and bitset-based edge filtering.
    """
    if deadline is not None and time.time() > deadline:
        return True  # assume colorable to avoid false negatives under time pressure
    
    # Precompute for each vertex the list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for ei, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(ei)
    
    m = len(edges)
    
    # State: color[v] = 0 or 1 for assigned, -1 for unassigned
    # edge_state[ei] = bitmask of colors present in that edge (0, 1, 2, 3)
    # We track for each edge: count of 0s and count of 1s assigned
    # An edge is "satisfied" (monochromatic) if all assigned vertices are same color
    # Unit propagation: if an edge has 4 vertices of color 0 and 1 unassigned, the last must be 1
    #                    if an edge has 4 vertices of color 1 and 1 unassigned, the last must be 0
    
    # For efficiency, use a recursive backtracking with propagation
    
    # Precompute edge membership as bitmasks for fast checking
    edge_bitmasks = [0] * m
    for ei, edge in enumerate(edges):
        mask = 0
        for v in edge:
            mask |= (1 << v)
        edge_bitmasks[ei] = mask
    
    # For each edge, store the list of vertices
    edge_vertices = [list(e) for e in edges]
    
    # Counters: for each edge, how many vertices are color 0, how many color 1
    edge_zero_count = [0] * m
    edge_one_count = [0] * m
    
    color = [-1] * n
    assigned_count = [0] * m  # number of assigned vertices in each edge
    
    # Propagation queue
    def propagate(queue):
        """Process unit propagation. Returns False if contradiction."""
        while queue:
            v = queue.pop()
            c = color[v]
            if c == -1:
                continue
            for ei in vertex_edges[v]:
                if c == 0:
                    edge_zero_count[ei] += 1
                else:
                    edge_one_count[ei] += 1
                assigned_count[ei] += 1
                
                # Check for contradiction: edge is monochromatic
                if edge_zero_count[ei] == 5 or edge_one_count[ei] == 5:
                    return False
                
                # Unit propagation: if 4 are one color, the 5th must be the other
                if assigned_count[ei] == 4:
                    # Find the unassigned vertex
                    for u in edge_vertices[ei]:
                        if color[u] == -1:
                            if edge_zero_count[ei] == 4:
                                # Must be color 1
                                if color[u] == 0:
                                    return False
                                if color[u] == -1:
                                    color[u] = 1
                                    queue.append(u)
                            elif edge_one_count[ei] == 4:
                                # Must be color 0
                                if color[u] == 1:
                                    return False
                                if color[u] == -1:
                                    color[u] = 0
                                    queue.append(u)
                            break
        return True
    
    # Undo function
    def undo_changes(changes):
        """Undo color assignments and counter changes."""
        for v, old_c, old_zero, old_one, old_assigned in changes:
            color[v] = old_c
            # We need to restore edge counts, but this is complex.
            # Instead, let's use a simpler approach: snapshot the state.
            pass
    
    # Simpler approach: use a stack of states
    # Actually, let's just do a clean recursive backtracking with state copying
    # Since n <= 26 and m <= 200, this should be fast enough with good heuristics.
    
    # Better: use a proper backtracking with incremental state management
    
    # Let me rewrite with a cleaner state management
    
    # State: color array, edge_zero_count, edge_one_count, assigned_count
    # We'll use a list to track changes for undo
    
    def backtrack(vertex_order, idx):
        """Backtrack through vertices in given order."""
        if idx == len(vertex_order):
            # All vertices assigned, check if any edge is monochromatic
            for ei in range(m):
                if edge_zero_count[ei] == 5 or edge_one_count[ei] == 5:
                    return False
            return True  # 2-colorable
        
        v = vertex_order[idx]
        if color[v] != -1:
            # Already assigned by propagation
            return backtrack(vertex_order, idx + 1)
        
        # Try both colors
        for c in [0, 1]:
            # Save state
            changes = []
            
            # Assign color
            color[v] = c
            queue = [v]
            
            if propagate(queue):
                if backtrack(vertex_order, idx + 1):
                    return True
            
            # Undo: we need to restore state
            # This is tricky with the propagation. Let's use a different approach.
            pass
        
        return False
    
    # The above is getting complicated. Let's use a simpler but effective approach:
    # Just do recursive backtracking with full state snapshots for small n,
    # and for larger n use a smarter heuristic.
    
    # Actually, for n <= 26 and m <= 200, a well-implemented backtracker should work.
    # Let me use a cleaner implementation.
    
    # State representation:
    # color: list of length n, -1/0/1
    # For each edge: zero_count, one_count, assigned_count
    
    # Use a list of (vertex, color, edge_updates) for undo
    # edge_updates: list of (edge_idx, old_zero, old_one, old_assigned)
    
    # This is getting complex. Let me just implement a straightforward backtracker
    # with memoization on the assigned set for small cases.
    
    # For the purpose of this task, let's use a simpler check:
    # Try all 2^n colorings? No, n can be up to 26.
    
    # Let me implement a proper DPLL-style solver.
    
    # Reset state
    color = [-1] * n
    edge_zero = [0] * m
    edge_one = [0] * m
    edge_assigned = [0] * m
    
    # Order vertices by degree (most constrained first)
    vertex_degrees = [len(vertex_edges[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -vertex_degrees[v])
    
    # For propagation, we need to be able to undo. Let's use a stack.
    # Each assignment pushes a record of what changed.
    
    # I'll implement this more carefully.
    
    # Given the complexity, let me use a simpler but correct approach:
    # Recursive backtracking with state copying (slower but correct)
    # For n up to 26, this might be slow, but with good heuristics it should be okay.
    
    # Actually, let me just implement it properly with undo.
    
    # I'll use a different strategy: since the hypergraph is small,
    # I'll implement a simple backtracking solver.
    
    def solve():
        color = [-1] * n
        ez = [0] * m
        eo = [0] * m
        ea = [0] * m
        
        # Propagation function that returns list of changes or None if contradiction
        def propagate(changes):
            # changes is a list of (v, c) that need to be processed
            # We process them and add new forced assignments
            # Returns None if contradiction, else list of all changes made
            made = list(changes)
            i = 0
            while i < len(made):
                v, c = made[i]
                i += 1
                if color[v] != -1:
                    if color[v] != c:
                        return None  # contradiction
                    continue
                color[v] = c
                for ei in vertex_edges[v]:
                    if c == 0:
                        ez[ei] += 1
                    else:
                        eo[ei] += 1
                    ea[ei] += 1
                    
                    if ez[ei] == 5 or eo[ei] == 5:
                        return None
                    if ea[ei] == 4:
                        # Find unassigned vertex
                        for u in edge_vertices[ei]:
                            if color[u] == -1:
                                if ez[ei] == 4:
                                    made.append((u, 1))
                                elif eo[ei] == 4:
                                    made.append((u, 0))
                                break
            return made
        
        def undo(made):
            for v, c in reversed(made):
                for ei in vertex_edges[v]:
                    if c == 0:
                        ez[ei] -= 1
                    else:
                        eo[ei] -= 1
                    ea[ei] -= 1
                color[v] = -1
        
        def bt(idx):
            if deadline is not None and time.time() > deadline:
                return None  # timeout
            if idx == len(vertex_order):
                # Check if any edge is monochromatic
                for ei in range(m):
                    if ez[ei] == 5 or eo[ei] == 5:
                        return False  # not 2-colorable
                return True  # 2-colorable
            
            v = vertex_order[idx]
            if color[v] != -1:
                return bt(idx + 1)
            
            for c in [0, 1]:
                # Try assigning v = c
                made = propagate([(v, c)])
                if made is not None:
                    result = bt(idx + 1)
                    if result is not None:
                        return result
                    undo(made)
                # Check deadline
                if deadline is not None and time.time() > deadline:
                    return None
            
            return False  # not 2-colorable (this branch)
        
        # Initial propagation with no assignments
        # Just start backtracking
        result = bt(0)
        if result is None:
            return True  # assume colorable on timeout
        return result
    
    return solve()


def construct():
    """
    Construct a non-2-colorable 5-uniform hypergraph with minimal edges.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Strategy: Start with a known non-2-colorable hypergraph and try to minimize edges.
    # We'll use a greedy removal approach: start with a non-2-colorable hypergraph,
    # and try to remove edges while maintaining non-2-colorability.
    
    # First, let's generate candidate hypergraphs.
    # We'll try different n values and use random constructions.
    
    best_edges = None
    best_n = None
    
    # Try n from 9 to 20
    for n in range(9, 21):
        if time.time() > deadline:
            break
        
        # Generate a random 5-uniform hypergraph
        # Start with a dense random hypergraph and try to remove edges
        all_possible = list(combinations(range(n), 5))
        random.shuffle(all_possible)
        
        # Start with a random subset
        # For non-2-colorability, we need enough edges
        # Let's start with about 50-100 edges and try to reduce
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random subset
            num_edges = min(len(all_possible), random.randint(40, 80))
            edges = [list(e) for e in all_possible[:num_edges]]
            
            # Check if non-2-colorable
            if not _is_2_colorable(n, edges, deadline):
                # Try to remove edges greedily
                current_edges = list(edges)
                i = 0
                while i < len(current_edges) and time.time() < deadline:
                    # Try removing edge i
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if not _is_2_colorable(n, test_edges, deadline):
                        current_edges = test_edges
                        # Don't increment i, try removing the next edge
                    else:
                        i += 1
                
                if len(current_edges) < (best_edges is None or len(best_edges)):
                    best_edges = current_edges
                    best_n = n
    
    # Fallback: complete hypergraph on 9 vertices
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    
    return {'n': best_n, 'edges': best_edges}
