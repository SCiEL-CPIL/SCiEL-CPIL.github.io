import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set, S_list):
        """Check if x can be added to S without violating sum-free property."""
        for a in S_list:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def build_conflict_graph(V):
        """
        Build the conflict graph for the sum-free problem.
        Two elements conflict if a + b = c for some a, b, c in the set.
        Returns adjacency as a list of sets.
        """
        V_set = set(V)
        n = len(V)
        idx = {v: i for i, v in enumerate(V)}
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                s = V[i] + V[j]
                if s in V_set:
                    k = idx[s]
                    adj[i].add(k)
                    adj[k].add(i)
                    if j != i:
                        adj[j].add(k)
                        adj[k].add(j)
        return adj

    def greedy_mis(adj, time_limit, rng_seed=None):
        """
        Greedy maximum independent set using conflict degrees.
        """
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
            
        n = len(adj)
        in_set = [False] * n
        removed = set()
        
        # Start with all nodes
        remaining = set(range(n))
        
        # Greedy: repeatedly pick node with min degree in remaining
        while remaining and time.time() < start_time + time_limit:
            # Find node with minimum degree in remaining
            min_deg = float('inf')
            best_node = None
            for node in remaining:
                deg = len(adj[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
            
            if best_node is None:
                break
                
            in_set[best_node] = True
            remaining.discard(best_node)
            # Remove neighbors from remaining
            for neighbor in adj[best_node]:
                remaining.discard(neighbor)
        
        return [i for i in range(n) if in_set[i]]

    def local_search_improve(V, adj, initial_set, time_limit, rng_seed=None):
        """
        Improve an initial independent set via local search.
        """
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
            
        n = len(V)
        idx = {v: i for i, v in enumerate(V)}
        
        S = set(initial_set)
        S_indices = set(initial_set)
        
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            
            # Try to add each non-in-set node
            candidates = [i for i in range(n) if i not in S_indices]
            rng.shuffle(candidates)
            
            for i in candidates:
                if time.time() > start_time + time_limit:
                    break
                    
                # Check if i can be added
                can_add_i = True
                for j in adj[i]:
                    if j in S_indices:
                        can_add_i = False
                        break
                        
                if can_add_i:
                    S_indices.add(i)
                    improved = True
            
            # Try 1-swap: remove one, add another
            if not improved:
                for i in list(S_indices):
                    if time.time() > start_time + time_limit:
                        break
                        
                    for j in [k for k in range(n) if k not in S_indices]:
                        if time.time() > start_time + time_limit:
                            break
                            
                        # Check if removing i and adding j is valid
                        # j must not conflict with any remaining node in S_indices - {i}
                        valid = True
                        for k in adj[j]:
                            if k != i and k in S_indices:
                                valid = False
                                break
                                
                        if valid:
                            S_indices.discard(i)
                            S_indices.add(j)
                            improved = True
                            break
        
        return [V[i] for i in S_indices]

    def diverse_init_local_search(V, time_limit, num_starts=8):
        """
        Try multiple diverse initializations and local search from each.
        """
        best_S = []
        best_size = 0
        n = len(V)
        
        # Build conflict graph once
        adj = build_conflict_graph(V)
        
        # Time per start
        time_per_start = max(1.0, time_limit / num_starts - 1)
        
        # Define diverse initializations
        inits = []
        
        # 1. Odds only
        odds = [x for x in V if x % 2 != 0]
        if odds:
            inits.append(odds)
        
        # 2. Positives only
        pos = [x for x in V if x > 0]
        if pos:
            inits.append(pos)
        
        # 3. Negatives only
        neg = [x for x in V if x < 0]
        if neg:
            inits.append(neg)
        
        # 4. High absolute value elements
        sorted_by_abs = sorted(V, key=lambda x: abs(x), reverse=True)
        inits.append(sorted_by_abs[:n//2])
        
        # 5. Low absolute value elements
        sorted_by_abs_asc = sorted(V, key=lambda x: abs(x))
        inits.append(sorted_by_abs_asc[:n//2])
        
        # 6. Random sample
        rng = random.Random()
        inits.append(rng.sample(V, n//2))
        
        # 7. Another random sample
        inits.append(rng.sample(V, n//3))
        
        # 8. Greedy MIS result
        greedy_result_indices = greedy_mis(adj, time_per_start // 2, rng_seed=42)
        greedy_result = [V[i] for i in greedy_result_indices]
        inits.append(greedy_result)
        
        for init in inits:
            if time.time() > start_time + time_limit - 2:
                break
            
            # Convert to indices
            idx = {v: i for i, v in enumerate(V)}
            init_indices = [idx[x] for x in init if x in idx]
            
            # Ensure it's an independent set by removing conflicts
            S_indices = set(init_indices)
            # Remove conflicts iteratively
            changed = True
            while changed and time.time() < start_time + time_limit - 2:
                changed = False
                for i in list(S_indices):
                    for j in adj[i]:
                        if j in S_indices and j > i:
                            S_indices.discard(j)
                            changed = True
                            break
            
            # Now do local search
            remaining_time = time_limit - (time.time() - start_time) - 1
            if remaining_time < 0.5:
                result = [V[i] for i in S_indices]
            else:
                result = local_search_improve(V, adj, list(S_indices), remaining_time, rng_seed=hash(frozenset(init)) % 10000)
            
            if len(result) > best_size:
                best_size = len(result)
                best_S = result
        
        if not best_S:
            # Fallback: odds
            best_S = [x for x in V if x % 2 != 0]
        
        return best_S

    results = []

    # Time allocation: give more time to larger instances
    # Instance 1 (60): fast
    # Instance 2 (120): medium
    # Instance 3 (240): slow
    time_alloc = {60: 10.0, 120: 25.0, 240: 70.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)

        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        if remaining_time < 3:
            # Emergency: just return odds
            odds = [x for x in V if x % 2 != 0]
            results.append(odds)
            continue

        t_limit = min(time_alloc.get(size, 15.0), remaining_time - 3)

        # Run diverse init local search
        best_S = diverse_init_local_search(V, t_limit, num_starts=8)

        # Final verification
        if not is_sum_free(best_S):
            # Fallback to odds
            best_S = [x for x in V if x % 2 != 0]

        results.append(sorted(best_S))

    return results
