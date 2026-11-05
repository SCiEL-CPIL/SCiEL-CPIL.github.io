import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # --- Helper: Verify a graph (list of adj masks) ---
    def is_valid_graph(adj_masks, n):
        """
        Checks if the graph has no regular induced subgraph of size >= 6.
        Returns True if valid.
        """
        # Optimization: If a graph has a regular induced subgraph of size k > 6,
        # it doesn't necessarily imply one of size 6. 
        # However, for random graphs, size 6 is the main bottleneck.
        # We must check ALL sizes >= 6 for correctness.
        
        # Precompute masks for all subsets of size 6..n
        # This is expensive if done naively, but n<=21.
        
        # Fast check for size 6
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask).bit_count())
            
            if len(set(degs)) == 1:
                return False
        
        # If size 6 is clean, check larger sizes
        # This is much rarer and more expensive, but we must do it.
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask).bit_count())
                
                if len(set(degs)) == 1:
                    return False
        return True

    # --- Helper: Count violations of size 6 (for optimization) ---
    def count_violations_size6(adj_masks, n):
        count = 0
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask).bit_count())
            
            if len(set(degs)) == 1:
                count += 1
        return count

    # --- Strategy 1: Algebraic Construction (Paley-like) ---
    # Paley graph exists for p = 4k+3.
    # p=19 is a candidate.
    # Let's check if Paley(19) is valid.
    
    def is_quadratic_residue(a, p):
        if a == 0: return False
        return pow(a, (p-1)//2, p) == 1

    def build_paley(n):
        if n != 19: return None
        p = 19
        adj = [0]*p
        for i in range(p):
            for j in range(i+1, p):
                if is_quadratic_residue((i-j) % p, p) or is_quadratic_residue((j-i) % p, p):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Try Paley(19)
    adj19 = build_paley(19)
    if adj19 is not None:
        if is_valid_graph(adj19, 19):
            edges = []
            for i in range(19):
                m = adj19[i]
                for j in range(i+1, 19):
                    if m & (1 << j):
                        edges.append([i, j])
            return {'n': 19, 'edges': edges}

    # --- Strategy 2: Local Search for n=20, 21 ---
    # We use a "Memetic" approach: Random Restart + Local Search (Flip)
    # Objective: Minimize number of size-6 regular induced subgraphs.
    # If we reach 0, we check larger sizes.
    
    best_n = 5
    best_edges = []
    
    # We will try to find a valid graph for n=21, then 20.
    # If we find one, we return it.
    
    # Precompute subset masks for size 6 to speed up violation counting?
    # For n=21, C(21,6) = 54264.
    # Storing 54k masks is fine.
    
    def get_subset_masks(n, k):
        masks = []
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks.append((mask, subset))
        return masks

    # We will iterate through target_n from 21 down to 6
    for target_n in range(21, 5, -1):
        if time.time() > deadline - 5.0: # Reserve 5s for final check/conversion
            break
            
        # Precompute masks for this n
        masks_6 = get_subset_masks(target_n, 6)
        
        # Local Search Parameters
        max_restarts = 50
        max_steps_per_restart = 2000
        
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Evaluate initial state
            current_violations = count_violations_size6(adj, target_n)
            
            # If 0 violations for size 6, check if fully valid
            if current_violations == 0:
                if is_valid_graph(adj, target_n):
                    edges = []
                    for i in range(target_n):
                        m = adj[i]
                        for j in range(i+1, target_n):
                            if m & (1 << j):
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}
            
            # Local Search Loop
            # We want to minimize violations.
            # A "move" is flipping an edge (u, v).
            # Flipping an edge affects the degrees of u and v in any subset containing u or v.
            # Specifically, for a subset S of size 6:
            # If u, v not in S: No change.
            # If u in S, v not in S: deg(u) changes by +/- 1.
            # If v in S, u not in S: deg(v) changes by +/- 1.
            # If u, v in S: deg(u) and deg(v) change by +/- 1.
            
            # Efficient evaluation:
            # We can't easily update the violation count for ALL subsets in O(1).
            # We can only re-evaluate the subsets affected by the flip.
            # Subsets affected:
            # 1. Subsets containing u but not v.
            # 2. Subsets containing v but not u.
            # 3. Subsets containing both u and v.
            
            # For n=21, C(21,6) = 54k.
            # Subsets containing u: C(20, 5) = 15504.
            # This is small enough to re-evaluate directly if we do it smartly.
            
            # However, implementing the exact delta update for "is this subset regular?" is complex.
            # Simpler: Just re-count violations from scratch? No, too slow (54k * 6 ops * 2000 steps * 50 restarts is huge).
            
            # Alternative: Simulated Annealing / Hill Climbing with random restarts
            # Since we have a time budget, let's just try many random restarts with a simpler "repair" phase.
            # Or, just pure random search is very weak.
            
            # Let's try a "Greedy Repair" approach:
            # 1. Find all violating subsets.
            # 2. For each violating subset, pick a random edge within it (or incident to it) and flip it.
            # 3. Re-evaluate.
            
            # But re-evaluation is the bottleneck.
            # Let's use the fact that n is small.
            # We can store the degrees for each vertex in each subset?
            # Memory: 54k subsets * 6 vertices * 1 byte = ~300KB. This is fine.
            # Let's precompute the "state" of each subset: the degrees of the 6 vertices.
            
            # This allows us to update quickly.
            # State: list of (mask, subset, [deg0, deg1, ..., deg5])
            
            def build_subset_states(adj, n, masks):
                states = []
                for mask, subset in masks:
                    degs = []
                    for v in subset:
                        degs.append((adj[v] & mask).bit_count())
                    states.append([mask, subset, degs])
                return states

            def count_violations_from_states(states):
                count = 0
                for state in states:
                    degs = state[2]
                    if len(set(degs)) == 1:
                        count += 1
                return count

            # Update states after flipping edge (u, v)
            def flip_edge(adj, u, v):
                # Check if edge exists
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return 1 # Flipped off
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    return 0 # Flipped on

            # This approach is getting complex. Let's stick to the previous program's strategy but optimized.
            # The previous program scored 2.8 (n=20).
            # It used random restarts and a fast verifier.
            # The verifier was the bottleneck if it didn't exit early.
            
            # Let's try a different structural change:
            # **Cayley Graph Construction**
            # Let G be a group of order n (or close).
            # For n=20, use Z_20 or D_10.
            # Define a connection set S.
            # A Cayley graph is vertex-transitive.
            # Induced subgraphs of Cayley graphs are often not regular.
            
            # Let's test Z_20 with a random connection set S (symmetric, 0 not in S).
            # |S| = d. Degree d.
            
            def build_cayley_z20():
                n = 20
                adj = [0]*n
                # Random symmetric set S
                # Possible edges: 1..9 (since Z_20 is abelian, 1 and 19 are same, etc.)
                # Distances: 1, 2, ..., 9
                available = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                random.shuffle(available)
                # Choose degree ~ n/2 = 10?
                # Since we pick pairs (d, 20-d), we have 9 pairs.
                # Max degree 18.
                # Let's pick 5 random pairs -> degree 10.
                chosen_pairs = available[:5]
                S = set()
                for d in chosen_pairs:
                    S.add(d)
                    S.add(20-d)
                
                for i in range(n):
                    for d in S:
                        j = (i + d) % n
                        if j != i:
                            adj[i] |= (1 << j)
                return adj

            # Try a few Cayley graphs
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                adj = build_cayley_z20()
                if is_valid_graph(adj, 20):
                    edges = []
                    for i in range(20):
                        m = adj[i]
                        for j in range(i+1, 20):
                            if m & (1 << j):
                                edges.append([i, j])
                    return {'n': 20, 'edges': edges}
            
            # If Cayley fails, fall back to optimized random search for this n
            # We will use the "states" approach for faster local search if we implement it.
            # But for now, let's just do more random restarts with the fast verifier.
            # The previous program did this.
            
            # To improve: Use a better initial density.
            # Regular subgraphs are more likely in balanced graphs.
            # Try p=0.4, 0.5, 0.6.
            
            for p in [0.4, 0.5, 0.6]:
                for _ in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    adj = [0] * target_n
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if random.random() < p:
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                    
                    if is_valid_graph(adj, target_n):
                        edges = []
                        for i in range(target_n):
                            m = adj[i]
                            for j in range(i+1, target_n):
                                if m & (1 << j):
                                    edges.append([i, j])
                        return {'n': target_n, 'edges': edges}

    # If we get here, we haven't found a valid graph for n >= 6.
    # Return a trivial valid graph (n=5)
    # But we should try to return the best we found?
    # The loop breaks on deadline.
    # If we exited the loop without returning, we didn't find one for any n.
    # Return n=5.
    
    return {'n': 5, 'edges': []}
