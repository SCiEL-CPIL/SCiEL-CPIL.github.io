import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def find_violation(adj, n):
        """
        Finds a violating induced subgraph (size >= 6) where all degrees are equal.
        Returns the subset as a list of indices, or None if no violation found.
        Optimized for n <= 22.
        """
        # Check size 6 first, as it's the most common constraint and has the most subsets
        # C(22, 6) = 74613. C(21, 6) = 54264.
        # We can vectorize the check for a specific subset size using bit operations or numpy.
        
        # For speed, we'll iterate combinations but use a fast inner check.
        # A faster approach: 
        # For each subset S, compute degrees d_v = sum(u in S \ {v} : u~v).
        # If all d_v are equal, it's regular.
        
        # Let's use a hybrid: 
        # 1. Precompute adjacency matrix as uint8 or bool for numpy.
        # 2. For each combination, extract submatrix and check regularity.
        
        # However, iterating 75k combinations in Python with numpy overhead might be slow if done naively.
        # Let's try to optimize the inner loop using bit masks first, as Python integers are fast for bit_count.
        
        # We will check k=6, then k=7, etc.
        
        for k in range(6, n + 1):
            # Optimization: if k > n, break
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                # Check regularity
                # We need degree of each v in S within S.
                # deg(v) = (adj[v] & mask).bit_count()
                
                first_deg = -1
                regular = True
                for v in S:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        regular = False
                        break
                
                if regular:
                    return list(S)
        return None

    def try_find_graph(n, time_limit):
        """
        Local search / Simulated Annealing to find a valid graph of size n.
        """
        start = time.time()
        
        # Initialize with a random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Check if initial is valid
        viol = find_violation(adj, n)
        if viol is None:
            return adj
        
        # Simulated Annealing
        # Temperature starts high, decreases.
        # Move: Flip a random edge.
        # Cost: Number of violating subsets? Or just binary (valid/invalid)?
        # Binary cost is hard for SA. 
        # Better: Count the number of violating subsets of size 6.
        # If we minimize the count of size-6 regular subgraphs, we might reach 0.
        
        def count_violations_size6(adj, n):
            count = 0
            for S in itertools.combinations(range(n), 6):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                first_deg = -1
                regular = True
                for v in S:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        regular = False
                        break
                if regular:
                    count += 1
            return count

        # This counting is expensive. C(22,6) ~ 75k checks.
        # Each check is ~6 bit_counts. 75k * 6 = 450k ops.
        # In Python, this might take ~0.1-0.2s per evaluation.
        # So we can only do ~100-500 steps in 30s.
        
        # Alternative: Use the violation found to guide the move.
        # If we have a violating set S, flipping an edge inside S or incident to S is more likely to break it.
        
        current_adj = adj[:]
        current_viol = viol
        
        T = 1.0
        cooling_rate = 0.99
        
        while time.time() - start < time_limit:
            # Pick a move
            # Strategy: Flip an edge involved in the current violation, or a random edge.
            if current_viol and random.random() < 0.5:
                # Pick two random vertices from the violation
                u, v = random.sample(current_viol, 2)
            else:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
            
            # Flip edge (u, v)
            mask_uv = (1 << u) | (1 << v)
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            # Check new state
            new_viol = find_violation(current_adj, n)
            
            if new_viol is None:
                return current_adj
            
            # Decide whether to accept
            # Heuristic: Accept if new violation is "smaller" or "different"?
            # Since we don't have a continuous cost, we can just accept if we haven't improved for a while,
            # or use a random walk.
            # Given the time limit, a simple random walk with occasional resets might be better.
            # Let's revert if it doesn't look promising? 
            # Actually, let's just keep the move. It's a random walk.
            # But we need to escape local minima.
            
            # If we are stuck in the same violation, maybe revert?
            # For simplicity, we just keep the move.
            
            # Cool down
            T *= cooling_rate
            
            # Occasionally restart if no progress? 
            # We don't track "progress" easily.
            # Let's add a counter for steps since last change in violation set?
            # Too complex for this time.
            
            # To escape local minima, we can do a "kick": flip a few random edges.
            if random.random() < 0.01:
                for _ in range(5):
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 != v2:
                        current_adj[u2] ^= (1 << v2)
                        current_adj[v2] ^= (1 << u2)
                current_viol = find_violation(current_adj, n)
                if current_viol is None:
                    return current_adj

        return None

    # Try n=22
    # We have ~135s. Let's give n=22 60s.
    adj_22 = try_find_graph(22, 60)
    if adj_22 is not None:
        # Verify fully (just in case find_violation missed something, though it checks all k)
        if find_violation(adj_22, 22) is None:
            return {'n': 22, 'edges': get_edges(adj_22, 22)}
    
    # Try n=21
    # Remaining time ~75s. Give 40s.
    adj_21 = try_find_graph(21, 40)
    if adj_21 is not None:
        if find_violation(adj_21, 21) is None:
            return {'n': 21, 'edges': get_edges(adj_21, 21)}
            
    # Try n=20
    # Remaining time ~35s. Give 30s.
    adj_20 = try_find_graph(20, 30)
    if adj_20 is not None:
        if find_violation(adj_20, 20) is None:
            return {'n': 20, 'edges': get_edges(adj_20, 20)}
            
    # Fallback: n=5
    return {'n': 5, 'edges': []}
