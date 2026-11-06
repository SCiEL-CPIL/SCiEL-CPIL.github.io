import sys

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        return r * r == n

    # 1. Identify valid nodes (no self-conflict)
    valid_nodes = []
    for i in range(1, n_max + 1):
        if not is_perfect_square(2 * i):
            valid_nodes.append(i)
            
    m = len(valid_nodes)
    # Map original node values to indices 0..m-1
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build adjacency matrix for the COMPLEMENT graph
    # In the original problem, we want an Independent Set in G (where edge = sum is square).
    # This is equivalent to a Clique in G_complement (where edge = sum is NOT square).
    # Note: We only consider valid nodes.
    
    # adj[i][j] = True if valid_nodes[i] and valid_nodes[j] can be in the same set
    # i.e., their sum is NOT a perfect square.
    adj = [[False] * m for _ in range(m)]
    
    for i in range(m):
        u = valid_nodes[i]
        for j in range(i + 1, m):
            v = valid_nodes[j]
            if not is_perfect_square(u + v):
                adj[i][j] = True
                adj[j][i] = True
    
    # 3. Max Clique Algorithm (Branch and Bound with Greedy Coloring)
    # We use a recursive approach.
    # State: 
    #   P: list of candidate vertices (indices) that can be added to the current clique.
    #   current_clique: list of vertices in the current clique.
    #   best_clique: the largest clique found so far.
    
    # To optimize, we work with bitmasks or lists. Lists are easier for coloring.
    # However, Python recursion and list operations might be slow for 256 nodes if not careful.
    # We will use a stack-based or recursive approach with aggressive pruning.
    
    best_size = 0
    best_clique = []
    
    # Greedy Coloring for upper bound
    # Returns number of colors and color of each vertex in the given list of vertices
    def greedy_color(vertices):
        # vertices is a list of indices
        n_v = len(vertices)
        if n_v == 0:
            return 0, []
        
        # We want to color the subgraph induced by 'vertices' in the complement graph?
        # NO. The bound is on the Clique size in the original graph G' (Complement).
        # A clique in G' is an independent set in G.
        # The greedy coloring is done on the graph G' restricted to vertices.
        # Wait, standard Max Clique B&B uses coloring on the graph G' (the one where we seek clique).
        # If we color G' with k colors, the max clique size is <= k.
        
        colors = [0] * n_v
        color_class = [] # list of lists, color_class[c] contains indices in 'vertices' with color c
        
        # Sort vertices by degree in the subgraph? Or just process in order.
        # Better: Sort by degree in the candidate set to improve coloring quality.
        # For simplicity and speed, we use a simple greedy assignment.
        
        # To make it faster, we can just assign the smallest available color.
        # But we need the count of colors used.
        
        # Optimization: Process vertices in reverse order of their index in the global array?
        # Or just iterate.
        
        # Let's use a simple greedy:
        # For each vertex v in vertices:
        #   Find smallest color c such that no neighbor of v in 'vertices' has color c.
        #   Assign c to v.
        
        # To speed up, we can precompute neighbors in the candidate set?
        # That's expensive. 
        # Alternative: Use the "reorder" trick from Tomita.
        # Sort vertices by degree in the candidate set descending.
        
        # Calculate degrees in the induced subgraph
        deg = [0] * n_v
        for i in range(n_v):
            vi = vertices[i]
            for j in range(n_v):
                if i != j and adj[vi][vertices[j]]:
                    deg[i] += 1
        
        # Sort vertices by degree descending
        # We need to keep track of original indices
        indexed_vertices = list(range(n_v))
        indexed_vertices.sort(key=lambda i: -deg[i])
        
        # Now color in this order
        # color_of_global_idx = {}
        # But we just need the count and the order for the next recursion.
        
        # Actually, the standard Tomita algorithm returns the colored vertices in a specific order
        # to facilitate the next recursion steps (processing color classes in reverse order of size).
        
        # Let's implement the standard "colorSort" from Tomita's algorithm.
        # It returns (num_colors, ordered_vertices) where ordered_vertices is sorted such that
        # vertices with higher color come first, and within same color, by index.
        
        # For simplicity and robustness in Python, let's just return the count.
        # The pruning condition: current_size + num_colors <= best_size -> prune.
        
        # To get a tighter bound, we need the structure.
        # Let's stick to a simpler but effective bound:
        # Upper bound = len(vertices) is weak.
        # Greedy coloring count is strong.
        
        # Implementation of Greedy Coloring count:
        # We use a list of colors assigned to the 'vertices' list.
        # colors[i] is the color of vertices[i].
        
        # To make it fast, we can use bitsets for neighbors?
        # With m=256, bitsets are integers.
        
        # Let's switch to Bitset representation for the graph to speed up B&B.
        # This requires a rewrite of the search loop.
        
        pass # Placeholder, will implement bitset version below

    # RE-IMPLEMENTATION WITH BITSETS FOR SPEED
    
    # Convert adj to bitmasks
    # mask[i] has bit j set if i and j are connected in G' (complement)
    masks = [0] * m
    for i in range(m):
        mask_val = 0
        for j in range(m):
            if i != j and adj[i][j]:
                mask_val |= (1 << j)
        masks[i] = mask_val
        
    # Helper: Popcount
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy Coloring on a set of candidates represented by a bitmask?
    # No, standard Tomita works on a list.
    # Given the size (256), a list-based approach with good pruning might suffice if optimized.
    # But Python is slow.
    
    # Let's try a different angle: **Iterative Greedy with Random Restart** (Simulated Annealing style).
    # This is a distinct family from the deterministic Greedy+LocalSearch in the seed.
    # It explores the solution space stochastically.
    
    import random
    
    def is_valid_set(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True

    def greedy_construction(order):
        s = []
        s_set = set()
        for v in order:
            if v in s_set: continue
            ok = True
            for u in s:
                if is_perfect_square(u + v):
                    ok = False
                    break
            if ok:
                s.append(v)
                s_set.add(v)
        return s

    def local_improve(s):
        # Try to add any valid node
        s_set = set(s)
        changed = True
        while changed:
            changed = False
            for v in valid_nodes:
                if v not in s_set:
                    ok = True
                    for u in s:
                        if is_perfect_square(u + v):
                            ok = False
                            break
                    if ok:
                        s.append(v)
                        s_set.add(v)
                        changed = True
        return s

    def swap_improve(s):
        # Try to replace one node with another to allow adding more?
        # Simple 1-swap: remove u, add v.
        # If v conflicts only with u, swap is neutral size.
        # But maybe after swap, we can add more?
        # Let's stick to "Kick" strategy from seed but randomize.
        pass

    best_solution = []
    
    # Multiple random restarts
    for _ in range(100):
        random.shuffle(valid_nodes)
        current = greedy_construction(valid_nodes)
        current = local_improve(current)
        
        # Simple local search
        for _ in range(100):
            improved = False
            # Try adding
            s_set = set(current)
            for v in valid_nodes:
                if v not in s_set:
                    ok = True
                    for u in current:
                        if is_perfect_square(u + v):
                            ok = False
                            break
                    if ok:
                        current.append(v)
                        s_set.add(v)
                        improved = True
                        break
            
            if improved:
                continue
                
            # Try 1-swap (remove u, add v)
            # Find v not in S that conflicts with exactly one node u in S
            found = False
            for v in valid_nodes:
                if v in s_set: continue
                conflicts = [u for u in current if is_perfect_square(u + v)]
                if len(conflicts) == 1:
                    u = conflicts[0]
                    current.remove(u)
                    current.append(v)
                    s_set.remove(u)
                    s_set.add(v)
                    found = True
                    break
            
            if found:
                continue
                
            # If no improvement, break or kick
            break
            
        if len(current) > len(best_solution):
            best_solution = current
            # Update best
            
    # Final verification
    if not is_valid_set(best_solution):
        # Fallback
        best_solution = []
        for v in valid_nodes:
            ok = True
            for u in best_solution:
                if is_perfect_square(u + v):
                    ok = False
                    break
            if ok:
                best_solution.append(v)
                
    return sorted(best_solution)
