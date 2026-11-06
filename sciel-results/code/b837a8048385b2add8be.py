import time
import random

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    # We aim for n=50.
    # State: adj[c][i] is a bitmask of vertices j such that edge (i,j) has color c.
    # Vertices are 0..n-1.
    
    def is_valid_extension(adj, n, new_masks):
        """
        Check if adding vertex n with edges defined by new_masks (list of 4 ints) is valid.
        new_masks[c] has bit i set if edge (i, n) has color c.
        """
        for c in range(4):
            mask = new_masks[c]
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                # If i is in new_masks[c], we check if i has any neighbor in color c
                # that is also in new_masks[c].
                # adj[c][i] contains all neighbors of i with color c in the existing graph.
                # If (adj[c][i] & mask) != 0, there is a triangle (i, j, n) with j in mask.
                if adj[c][i] & mask:
                    return False
                m ^= lsb
        return True

    def add_vertex(adj, n, new_masks):
        """
        Add vertex n to the graph.
        adj is a list of 4 lists of ints.
        """
        new_adj = [adj[c][:] for c in range(4)]
        # Update existing vertices' bitmasks to include n
        for c in range(4):
            mask = new_masks[c]
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                new_adj[c][i] |= (1 << n)
                m ^= lsb
        
        # Add the new vertex's bitmasks
        for c in range(4):
            new_adj[c].append(new_masks[c])
            
        return new_adj

    def greedy_construct(target_n, rng, deadline):
        """
        Try to construct a coloring of K_target_n.
        Returns the final adj list or None.
        """
        # Start with n=1
        # adj[c][0] = 0
        adj = [[0] for _ in range(4)]
        
        for current_n in range(1, target_n):
            if time.time() > deadline:
                return None
            
            found = False
            # Try random assignments
            # Optimization: We can shuffle the order of vertices or colors?
            # Just random restarts.
            attempts = 0
            while not found:
                attempts += 1
                if attempts > 10000:
                    return None # Restart outer loop or fail
                
                # Generate random masks
                new_masks = [0, 0, 0, 0]
                for i in range(current_n):
                    c = rng.randint(0, 3)
                    new_masks[c] |= (1 << i)
                    
                if is_valid_extension(adj, current_n, new_masks):
                    adj = add_vertex(adj, current_n, new_masks)
                    found = True
        
        return adj

    # Try to reach n=50
    # We might need multiple restarts if we get stuck, but greedy usually works for small n.
    # For n=50, greedy might fail. We need backtracking or local search.
    # Let's try a simple backtracking search with randomization.
    
    def backtracking_search(target_n, rng, deadline):
        adj = [[0] for _ in range(4)]
        
        def backtrack(n):
            if time.time() > deadline:
                return False
            if n == target_n:
                return True
            
            # Try random assignments for vertex n
            # We can order the assignments by some heuristic?
            # Just random.
            for _ in range(1000): # Limit attempts per node to avoid infinite loops
                new_masks = [0, 0, 0, 0]
                for i in range(n):
                    c = rng.randint(0, 3)
                    new_masks[c] |= (1 << i)
                    
                if is_valid_extension(adj, n, new_masks):
                    new_adj = add_vertex(adj, n, new_masks)
                    if backtrack(n + 1):
                        # Copy back? Or just return the adj?
                        # Since we modify adj in place in the recursive call?
                        # No, add_vertex returns a new list.
                        # We need to update the outer scope or return it.
                        # Let's use a nonlocal or return the adj.
                        # This is inefficient for deep recursion.
                        # Let's use a global best_adj.
                        return True
            return False
        
        # This simple backtracking is likely to fail for n=50 due to the branching factor.
        # We need a smarter approach.
        pass

    # Better approach: Iterative Deepening A* or just aggressive random restarts with local search?
    # Given the time limit, and that n=50 is the target, let's try to find a known solution.
    # Actually, the greedy method often works for Ramsey numbers if we use a good heuristic.
    # Heuristic: When adding vertex n, prefer colors that are "less connected" in the existing graph?
    
    # Let's try a different strategy:
    # Use the known construction for n=50.
    # The construction is based on Z_50.
    # I will hardcode the edge colors for n=50 if I can generate them correctly.
    # Since I cannot run code here, I will use the fast search to *find* it during the first few seconds.
    # If it finds it, great. If not, it returns the best found.
    
    # To ensure we get n=50, we can use a specific seed or a more robust search.
    # Let's use the bitset greedy search with multiple restarts.
    
    best_adj = None
    best_n = 0
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        rng2 = random.Random(123 + restart)
        adj = [[0] for _ in range(4)]
        current_n = 1
        stuck = False
        
        while current_n < 50:
            if time.time() > deadline - 2:
                stuck = True
                break
            
            # Try to add vertex current_n
            found = False
            attempts = 0
            while not found and attempts < 5000:
                attempts += 1
                new_masks = [0, 0, 0, 0]
                for i in range(current_n):
                    c = rng2.randint(0, 3)
                    new_masks[c] |= (1 << i)
                
                if is_valid_extension(adj, current_n, new_masks):
                    adj = add_vertex(adj, current_n, new_masks)
                    current_n += 1
                    found = True
            
            if not found:
                stuck = True
                break
        
        if current_n > best_n:
            best_n = current_n
            best_adj = adj
    
    # Convert best_adj to the required format
    # colors: list of length n*(n-1)/2
    # Order: lex (i,j)
    
    if best_adj is None:
        # Fallback to n=1
        return {'n': 1, 'colors': []}
    
    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Find color of edge (i,j)
            # Check adj[c][i] for bit j
            c_found = 0
            for c in range(4):
                if best_adj[c][i] & (1 << j):
                    c_found = c
                    break
            colors.append(c_found)
            
    return {'n': n, 'colors': colors}
