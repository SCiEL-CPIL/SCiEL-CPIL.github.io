import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Uses backtracking with unit propagation.
    """
    # Precompute adjacency: for each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    colors = [-1] * n
    
    # For fast checking, we track counts of 0s and 1s in each edge
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def propagate():
        # Unit propagation loop
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if colors[v] != -1:
                    continue
                # Check if any edge forces v
                forced_val = -1
                for e_idx in vertex_edges[v]:
                    e = edges[e_idx]
                    # Count assigned colors in this edge
                    c0 = count_0[e_idx]
                    c1 = count_1[e_idx]
                    # Number of unassigned in this edge
                    unassigned = 5 - c0 - c1
                    # If v is in this edge, it is one of the unassigned
                    # If c0 == 4, then all other 4 are 0. v must be 1.
                    if c0 == 4:
                        if forced_val != -1 and forced_val != 1:
                            return False # Conflict: forced to be different values
                        forced_val = 1
                    # If c1 == 4, then all other 4 are 1. v must be 0.
                    elif c1 == 4:
                        if forced_val != -1 and forced_val != 0:
                            return False
                        forced_val = 0
                if forced_val != -1:
                    colors[v] = forced_val
                    # Update counts
                    for e_idx in vertex_edges[v]:
                        if forced_val == 0:
                            count_0[e_idx] += 1
                        else:
                            count_1[e_idx] += 1
                    changed = True
                    # Check for conflict in edges
                    for e_idx in vertex_edges[v]:
                        if count_0[e_idx] == 5 or count_1[e_idx] == 5:
                            return False
        return True

    def backtrack(start_v):
        if start_v == n:
            return False # All vertices assigned, no conflict found -> NOT 2-colorable? 
                         # Wait, the function is is_2_colorable.
                         # If we reach the end without conflict, it IS 2-colorable.
                         # So return True.
            
        # Find next unassigned vertex
        v = start_v
        while v < n and colors[v] != -1:
            v += 1
        if v == n:
            return True # All assigned, valid coloring found
        
        # Try color 0
        if propagate():
            if backtrack(v + 1):
                return True
            # Undo
            # We need to undo the assignments made in this branch.
            # This is tricky with the global state.
            # Better to use a stack or pass state.
            # Given the small size, we can just re-initialize or use a simple stack.
            # Let's use a simple recursive structure with state passing or careful undo.
            # Actually, for simplicity and speed, let's use a standard DFS with state copying or careful undo.
            # Since n <= 26, copying is cheap.
            
        # This implementation is getting complex with global state and undo.
        # Let's use a simpler, slightly slower but correct approach:
        # Iterate through all 2^n colorings? No, 2^26 is too big.
        # But we only search until we find one.
        # The backtracking above needs proper undo.
        
        # Let's restart with a cleaner backtracking implementation.
        pass

    # Clean implementation
    def solve():
        colors = [-1] * n
        count_0 = [0] * len(edges)
        count_1 = [0] * len(edges)
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    if colors[v] != -1:
                        continue
                    forced_val = -1
                    for e_idx in vertex_edges[v]:
                        c0 = count_0[e_idx]
                        c1 = count_1[e_idx]
                        if c0 == 4:
                            if forced_val != -1 and forced_val != 1:
                                return False
                            forced_val = 1
                        elif c1 == 4:
                            if forced_val != -1 and forced_val != 0:
                                return False
                            forced_val = 0
                    if forced_val != -1:
                        colors[v] = forced_val
                        for e_idx in vertex_edges[v]:
                            if forced_val == 0:
                                count_0[e_idx] += 1
                            else:
                                count_1[e_idx] += 1
                        if count_0[e_idx] == 5 or count_1[e_idx] == 5:
                             # Wait, checking conflict here is redundant if we check after update
                             pass
                        changed = True
                # Check for conflicts after propagation step
                for e_idx in range(len(edges)):
                    if count_0[e_idx] == 5 or count_1[e_idx] == 5:
                        return False
            return True

        def backtrack(start_v):
            # Find next unassigned
            v = start_v
            while v < n and colors[v] != -1:
                v += 1
            if v == n:
                return True # Found a valid coloring
            
            # Try 0
            if colors[v] == -1:
                # Save state? No, we modify and undo.
                # But propagate modifies many.
                # Easiest is to copy the state.
                # n=10..14, edges<100. Copying is fast.
                
                # Try color 0
                colors_copy = colors[:]
                c0_copy = count_0[:]
                c1_copy = count_1[:]
                colors[v] = 0
                for e_idx in vertex_edges[v]:
                    count_0[e_idx] += 1
                if propagate():
                    if backtrack(v + 1):
                        return True
                # Restore
                colors[:] = colors_copy
                count_0[:] = c0_copy
                count_1[:] = c1_copy
                
                # Try color 1
                colors[v] = 1
                for e_idx in vertex_edges[v]:
                    count_1[e_idx] += 1
                if propagate():
                    if backtrack(v + 1):
                        return True
                # Restore
                colors[:] = colors_copy
                count_0[:] = c0_copy
                count_1[:] = c1_copy
                
            return False

        return backtrack(0)

    return solve()

def construct():
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Heuristic: Randomly remove edges while maintaining non-2-colorability
    # We want to minimize edges.
    
    best_edges = all_edges
    start_time = time.time()
    timeout = 100 # Leave margin
    
    # Try to find a smaller set
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
            
        # Shuffle edges
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        # Try to remove each edge
        # We iterate through the list and try to remove
        # If removal makes it 2-colorable, keep it.
        
        # To make it faster, we can try to remove in random order
        # and stop if we get stuck? No, we want to minimize.
        
        # Let's try a greedy removal:
        # Sort edges randomly.
        # For each edge, try removing it.
        # Check if remaining is NOT 2-colorable.
        # If yes, remove it.
        # If no, keep it.
        
        candidate = current_edges[:]
        for i in range(len(candidate) - 1, -1, -1):
            # Try removing candidate[i]
            test_edges = candidate[:i] + candidate[i+1:]
            if not is_2_colorable(n, test_edges):
                # It is still not 2-colorable, so we can remove the edge
                candidate = test_edges
            else:
                # It became 2-colorable, so we must keep the edge
                pass
        
        if len(candidate) < len(best_edges):
            best_edges = candidate
            # If we get very small, we can stop early?
            # Theoretical min is 29.
            # If we find < 51, it's a good improvement.
            
    return {'n': n, 'edges': best_edges}
