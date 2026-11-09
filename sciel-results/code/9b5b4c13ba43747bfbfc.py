import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - time.time()

    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    def popcount(x):
        return x.bit_count()

    def build_c4free(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for u, v in edges:
            common = adj[u] & adj[v]
            if popcount(common) <= 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def find_independent_set(adj, n, target_size):
        """
        Greedy search for an independent set of size target_size.
        Returns the set (as a list of vertices) or None if not found.
        """
        rng = random.Random(12345) # Fixed seed for determinism in heuristic? Or random?
        # Let's use random order for robustness
        order = list(range(n))
        rng.shuffle(order)
        
        indep = []
        in_set = set()
        
        for v in order:
            if v in in_set:
                continue
            # Check if v is non-adjacent to all in indep
            if all((adj[v] & (1 << u)) == 0 for u in indep):
                indep.append(v)
                in_set.add(v)
                if len(indep) >= target_size:
                    return indep
        return None

    def exact_alpha(adj, n):
        best_alpha = 0
        
        def branch(remaining, current_size):
            nonlocal best_alpha
            if remaining == 0:
                if current_size > best_alpha:
                    best_alpha = current_size
                return
            
            if current_size + popcount(remaining) <= best_alpha:
                return
                
            rem = remaining
            best_v = -1
            max_deg = -1
            while rem:
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                rem ^= lsb
                
            if best_v == -1:
                return
                
            new_remaining = remaining & ~(adj[best_v] | (1 << best_v))
            branch(new_remaining, current_size + 1)
            
            branch(remaining & ~(1 << best_v), current_size)

        full_mask = (1 << n) - 1
        branch(full_mask, 0)
        return best_alpha

    def local_search(adj, n, seed):
        """
        Try to reduce alpha by adding edges.
        """
        rng = random.Random(seed)
        current_alpha = exact_alpha(adj, n)
        
        if current_alpha < 11:
            return True
            
        for _ in range(50): # Max iterations
            if time_left() < 5.0:
                return False
                
            # Find an independent set of size 11
            # We use a random greedy approach to find *an* independent set of size 11
            # If we can't find one, alpha might be < 11 (but exact_alpha said no)
            # So we should find one.
            indep_set = find_independent_set(adj, n, 11)
            
            if indep_set is None:
                # Should not happen if exact_alpha >= 11
                # But maybe greedy failed. Try again with different seed?
                continue
            
            # Try to add an edge between two vertices in indep_set
            added = False
            # Shuffle pairs to randomize choice
            pairs = list(zip(indep_set, indep_set))
            # Filter u < v
            valid_pairs = [(u, v) for u, v in pairs if u < v]
            rng.shuffle(valid_pairs)
            
            for u, v in valid_pairs:
                common = adj[u] & adj[v]
                if popcount(common) <= 1:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if not added:
                # Cannot add any edge within this independent set.
                # Try to remove an edge? Or just break and restart.
                # For now, break and restart with new random graph.
                break
            
            # Re-check alpha
            current_alpha = exact_alpha(adj, n)
            if current_alpha < 11:
                return True

        return False

    best_result = None
    target_ns = [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    for n in target_ns:
        if time_left() < 5.0:
            break
            
        max_trials = 200 if n <= 38 else 100
        
        for trial in range(max_trials):
            if time_left() < 2.0:
                break
                
            seed = trial * 7919 + n
            adj = build_c4free(n, seed)
            
            # Check if already good
            alpha = exact_alpha(adj, n)
            if alpha < 11:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Try local search
            if local_search(adj, n, seed):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    return fallback()
