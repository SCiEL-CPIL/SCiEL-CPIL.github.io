import time
import random
import itertools

def count_c5(adj, n):
    """
    Counts the number of 5-cycles in the graph.
    adj is a list of bitmasks or a 2D list.
    For speed, we use a 2D list of booleans or ints.
    """
    # Using adjacency matrix for fast lookup
    # n is small, so O(n^5) is ~ 33^5 = 39M operations. 
    # In Python, this is too slow for a single check if done many times.
    # We need a faster check or a different objective.
    
    # Alternative: Just check if ANY C5 exists.
    # We can iterate over all 5-combinations.
    # 33 choose 5 is ~ 237k.
    # For each combo, check 5 edges.
    # 237k * 5 = 1.2M operations per check.
    # This is feasible if we do ~1000 checks.
    
    count = 0
    for combo in itertools.combinations(range(n), 5):
        u, v, w, x, y = combo
        # Check if u-v-w-x-y-u is a cycle
        # We need to check all 12 permutations? No, just one canonical form.
        # A set of 5 vertices forms a C5 if the induced subgraph is a C5.
        # This means each vertex has degree 2 in the induced subgraph.
        deg = [0]*5
        verts = [u,v,w,x,y]
        for i in range(5):
            for j in range(i+1, 5):
                if adj[verts[i]][verts[j]]:
                    deg[i] += 1
                    deg[j] += 1
        if all(d == 2 for d in deg):
            count += 1
    return count

def max_independent_set_size(adj, n):
    """
    Finds the size of the maximum independent set.
    Uses a simple branch and bound or greedy + local search.
    For n=33, we can try all subsets? No.
    We can use the complement graph clique size.
    """
    # Greedy heuristic for clique in complement
    # Complement adjacency: not adj[u][v] and u != v
    # We want max clique in complement.
    
    # Simple recursive search with pruning
    # Given the small n, we can try a randomized greedy approach multiple times.
    
    best = 0
    for _ in range(20): # 20 random restarts
        # Random order
        order = list(range(n))
        random.shuffle(order)
        current = []
        for v in order:
            # Check if v can be added to current
            # v must not be adjacent to any in current in ORIGINAL graph
            # i.e., v is NOT adjacent to any in current
            can_add = True
            for u in current:
                if adj[u][v]:
                    can_add = False
                    break
            if can_add:
                current.append(v)
        if len(current) > best:
            best = len(current)
            if best >= 9:
                return best
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Fallback: The known valid graph with n=32
    def get_fallback():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}

    best_result = get_fallback()
    
    # Try to find a graph with n > 32
    for n_target in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        # Initialize graph
        # Start with 8 K4s and n_target - 32 extra vertices
        # This keeps the "clique" structure as a starting point
        
        adj = [[0]*n_target for _ in range(n_target)]
        edges = []
        
        # Add 8 K4s
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[base+i][base+j] = 1
                    adj[base+j][base+i] = 1
                    edges.append([base+i, base+j])
        
        # Add extra vertices connected randomly
        for v in range(32, n_target):
            # Connect to random vertices
            # To keep alpha low, connect to many vertices
            # To keep C5 low, avoid creating C5s
            # Random connection
            for u in range(v):
                if random.random() < 0.5:
                    adj[v][u] = 1
                    adj[u][v] = 1
                    edges.append([u, v])
        
        # Local Search
        # Objective: Minimize C5_count + 10 * (alpha - 8)_+
        # We want C5_count == 0 and alpha < 9
        
        current_energy = float('inf')
        best_local_energy = float('inf')
        best_local_adj = [row[:] for row in adj]
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Calculate current energy
            # This is the bottleneck.
            # We need a fast energy function.
            
            # Fast C5 check:
            # If we just need to know if it's 0, we can stop early.
            # But for SA, we need a value.
            # Let's use the count of C5s.
            
            # Optimization: Only recalculate if graph changed.
            
            # To make it faster, we can approximate.
            # But let's try to make the exact check fast enough.
            # 237k combos * 5 edges * 2000 steps = 2.3 billion ops. Too slow.
            
            # We need a smarter approach.
            # Instead of full count, we can just check if ANY C5 exists.
            # And check if ANY I9 exists.
            # Energy = (1 if C5 else 0) * 100 + (1 if I9 else 0) * 10
            # This is a step function. SA might struggle.
            
            # Let's try a different strategy:
            # Random restarts with a "repair" phase.
            
            pass # Skip SA for now, use random restarts with validation
            
        # Since SA is too slow for exact checks, we use Random Restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Randomize graph
            adj = [[0]*n_target for _ in range(n_target)]
            edges = []
            
            # Start with sparse graph? Or dense?
            # Dense helps alpha, but hurts C5.
            # Let's try p=0.5
            
            for u in range(n_target):
                for v in range(u+1, n_target):
                    if random.random() < 0.5:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        edges.append([u, v])
                        
            # Check validity
            is_valid = True
            
            # Check C5
            # We can use a faster C5 detection algorithm?
            # Or just the combo check, but break early.
            for combo in itertools.combinations(range(n_target), 5):
                u, v, w, x, y = combo
                deg = [0]*5
                verts = [u,v,w,x,y]
                for i in range(5):
                    for j in range(i+1, 5):
                        if adj[verts[i]][verts[j]]:
                            deg[i] += 1
                            deg[j] += 1
                if all(d == 2 for d in deg):
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Check I9
            # Use the greedy heuristic from before, but make it more robust?
            # Or just check all 9-combos?
            # 33 choose 9 is 10M. Too slow for 50 restarts.
            
            # Use the greedy max independent set size check.
            # If greedy finds 9, it's invalid.
            # If greedy finds < 9, it might still be invalid, but we accept it as a candidate?
            # No, we must be exact.
            
            # We can use a simple backtracking for I9.
            # Since n is 33, and we know alpha is likely small if C5-free and dense,
            # backtracking might be fast if we prune well.
            
            def has_independent_set_9(adj, n):
                # Backtracking
                # Sort vertices by degree (ascending) to prune?
                # Actually, we want to find a set of 9 non-adjacent vertices.
                
                # Use a stack
                # If we can't find 9, return False
                
                # Heuristic: If the graph is dense, alpha is small.
                # We can try to find a clique of size 9 in the complement.
                
                # Let's use a simple recursive function with memoization? No, state is large.
                
                # We'll use a randomized greedy check with multiple restarts.
                # If ANY greedy run finds 9, it's invalid.
                # If NONE find 9, we assume valid?
                # Risky.
                
                # Better: Use the exact check but optimized.
                # Iterate 9-combos.
                # 10M combos.
                # In Python, 10M iterations takes ~0.5s.
                # 50 restarts * 0.5s = 25s.
                # This is feasible within 140s!
                
                for combo in itertools.combinations(range(n), 9):
                    # Check if all pairs are non-adjacent
                    valid = True
                    for i in range(9):
                        for j in range(i+1, 9):
                            if adj[combo[i]][combo[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return True
                return False

            if has_independent_set_9(adj, n_target):
                continue
                
            # If we reach here, it's valid!
            best_result = {'n': n_target, 'edges': edges}
            return best_result

    return best_result
