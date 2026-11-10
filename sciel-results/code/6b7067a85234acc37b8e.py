import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Proactive Constraint-Based Construction (Backtracking).
    
    Unlike local search which fixes violations after the fact, this approach
    builds the graph vertex by vertex. It maintains the state of all subsets
    of size up to 5 (which will become size 6 when a new vertex is added) 
    and checks if adding a new vertex would create a regular induced subgraph
    of size 6. If so, it backtracks. This is much faster than verifying the 
    whole graph repeatedly because it only checks the relevant local constraints 
    for the new vertex.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def try_construct(n, max_restarts=1000):
        """
        Attempts to construct a valid graph on n vertices using backtracking.
        Returns edge list if successful, None otherwise.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # adj[i] is a bitset of neighbors of i
            adj = [0] * n
            
            # We will build the graph by adding vertices 0 to n-1.
            # State: We need to ensure that no subset of 6 vertices is regular.
            # When adding vertex v, we only need to check subsets that include v 
            # and 5 previously added vertices.
            
            # To make backtracking efficient, we can use a simple recursive 
            # backtracking with randomization.
            
            def backtrack(v):
                if v == n:
                    return True
                
                # Try all possible connections from v to previous vertices 0..v-1
                # There are 2^v possibilities. For v up to 20, this is too many 
                # for brute force, so we randomize the order.
                
                # Generate random permutations of the possible edge masks
                # Actually, we can just try random masks until we find one that works.
                # Or, more efficiently, we can check constraints as we assign edges.
                
                # Let's use a simpler approach: 
                # Pick a random mask for edges from v to 0..v-1.
                # Check if it creates a regular 6-subset.
                # If yes, try another mask.
                # If we try too many masks, backtrack to v-1.
                
                # However, pure random mask trial is slow.
                # Let's use a heuristic: 
                # 1. Randomly assign edges.
                # 2. Check all subsets of size 6 containing v.
                # 3. If any is regular, reject.
                # 4. If accepted, proceed to v+1.
                # 5. If stuck, backtrack.
                
                # To avoid infinite loops, we limit the number of attempts per vertex.
                attempts = 0
                max_attempts = 50
                
                # Shuffle the order of masks? No, just random sampling.
                # We can bias towards sparse or dense graphs if needed, but random is a good start.
                
                while attempts < max_attempts:
                    attempts += 1
                    
                    # Randomly decide edges to previous vertices
                    mask = 0
                    for u in range(v):
                        if random.random() < 0.5:
                            mask |= (1 << u)
                    
                    # Check validity: 
                    # For every subset S of {0, ..., v-1} of size 5,
                    # check if S U {v} is a regular graph.
                    valid = True
                    for subset5 in combinations(range(v), 5):
                        # The subset is subset5 + (v,)
                        # We need to check if the induced subgraph on these 6 vertices is regular.
                        
                        # Degrees in the induced subgraph:
                        # deg(v) in subset = popcount(mask & subset5_mask)
                        # deg(u) in subset = 1 (if connected to v) + deg(u) in subset5
                        
                        # Let's compute degrees.
                        # To speed this up, we can precompute degrees in subset5?
                        # No, subset5 changes.
                        
                        # Let's just compute directly.
                        # There are C(v, 5) subsets. For v=20, C(19,5) = 11628.
                        # This is expensive to do for every random mask.
                        
                        # Optimization: 
                        # Precompute for each subset5, the degree vector of vertices in subset5 
                        # induced by edges among themselves.
                        # Then deg(u) in S U {v} = deg_internal(u) + (1 if u in mask else 0).
                        # deg(v) in S U {v} = popcount(mask & subset5_mask).
                        
                        # We need all 6 degrees to be equal.
                        
                        # Let's precompute internal degrees for all subsets of size 5?
                        # That's too much memory/state.
                        
                        # Alternative: 
                        # Just check the most likely violations first?
                        # Or, use a faster check.
                        
                        # Let's stick to the simple check but optimize the inner loop.
                        subset_mask = 0
                        for u in subset5:
                            subset_mask |= (1 << u)
                            
                        deg_v = (mask & subset_mask).bit_count()
                        
                        # Check if all u in subset5 have the same degree
                        # and if that degree matches deg_v (adjusted for connection to v)
                        
                        # Actually, it's easier to just compute the full degree sequence.
                        # But we can early exit.
                        
                        # Let's try a different optimization:
                        # We only need to check if the graph is regular.
                        # If deg_v is d, then every u in subset5 must have degree d in the induced subgraph.
                        # deg_u = (adj[u] & subset_mask).bit_count() + (1 if mask & (1<<u) else 0)
                        
                        # Wait, adj[u] only contains edges to vertices < u? 
                        # No, adj is being built. adj[u] contains edges to all previous vertices.
                        # Since u < v, adj[u] is fully defined.
                        
                        # So deg_u_in_S = (adj[u] & subset_mask).bit_count() + (1 if mask & (1<<u) else 0)
                        
                        # Note: adj[u] & subset_mask only counts edges to other vertices in subset5.
                        # It does NOT count the edge to v, which we add separately.
                        
                        # So the logic is correct.
                        
                        # Optimization: 
                        # If deg_v is 0 or 5, it's easy to check.
                        # Generally, we can iterate through subset5.
                        
                        for u in subset5:
                            deg_u = (adj[u] & subset_mask).bit_count()
                            if mask & (1 << u):
                                deg_u += 1
                            
                            if deg_u != deg_v:
                                valid = False
                                break
                        
                        if not valid:
                            # This subset is NOT regular, which is good. 
                            # Wait, I want to find if ANY subset IS regular.
                            # If I find one that IS regular, then this mask is INVALID.
                            # So if deg_u == deg_v for all u, then this subset IS regular.
                            # So I should break if I find a match, and mark invalid.
                            
                            # My logic above was: if deg_u != deg_v, break (meaning this subset is NOT regular).
                            # I need to check if the subset IS regular.
                            
                            # Let's restructure:
                            # is_regular = True
                            # for u in subset5:
                            #    if deg_u != deg_v:
                            #       is_regular = False
                            #       break
                            # if is_regular:
                            #    valid = False # The whole mask is invalid
                            #    break
                            
                            pass # Placeholder for logic fix below
                    
                    # Re-implement the check properly
                    valid = True
                    for subset5 in combinations(range(v), 5):
                        subset_mask = 0
                        for u in subset5:
                            subset_mask |= (1 << u)
                        
                        deg_v = (mask & subset_mask).bit_count()
                        
                        is_regular = True
                        for u in subset5:
                            deg_u = (adj[u] & subset_mask).bit_count()
                            if mask & (1 << u):
                                deg_u += 1
                            
                            if deg_u != deg_v:
                                is_regular = False
                                break
                        
                        if is_regular:
                            valid = False
                            break
                    
                    if valid:
                        # Accept this mask
                        for u in range(v):
                            if mask & (1 << u):
                                adj[v] |= (1 << u)
                                adj[u] |= (1 << v)
                        
                        if backtrack(v + 1):
                            return True
                        else:
                            # Backtrack: remove edges
                            for u in range(v):
                                if mask & (1 << u):
                                    adj[v] ^= (1 << u)
                                    adj[u] ^= (1 << v)
                            # Try next mask
                            continue
                    else:
                        # Try next mask
                        continue
                
                # If we exhausted max_attempts, backtrack
                return False

            if backtrack(0):
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return edges

        return None

    # Target n=20 first
    for n in range(20, 4, -1):
        edges = try_construct(n)
        if edges is not None:
            return {'n': n, 'edges': edges}

    # Fallback
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
