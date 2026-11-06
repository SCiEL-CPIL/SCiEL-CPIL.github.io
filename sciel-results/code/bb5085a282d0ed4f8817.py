import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0

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

    def get_conflicts(S):
        """Return dict: element -> number of conflicts it's involved in."""
        S_set = set(S)
        S_list = list(S)
        n = len(S_list)
        conflicts = {x: 0 for x in S}
        
        # Use numpy for vectorized conflict detection if n is large
        if n > 100:
            arr = np.array(S_list)
            # Compute all pair sums
            sums = arr[:, None] + arr[None, :]
            # Check which sums are in S_set
            for i in range(n):
                for j in range(i, n):
                    s = S_list[i] + S_list[j]
                    if s in S_set:
                        conflicts[S_list[i]] += 1
                        if j != i:
                            conflicts[S_list[j]] += 1
                        conflicts[s] += 1
        else:
            for i in range(n):
                for j in range(i, n):
                    s = S_list[i] + S_list[j]
                    if s in S_set:
                        conflicts[S_list[i]] += 1
                        if j != i:
                            conflicts[S_list[j]] += 1
                        conflicts[s] += 1
        
        return conflicts

    def can_add(x, S_set, S_list):
        """Check if x can be added to S without creating a sum-free violation."""
        # Check: x + a = b for some a, b in S
        for a in S_list:
            if x + a in S_set:
                return False
        # Check: a + b = x for some a, b in S
        for a in S_list:
            if x - a in S_set:
                return False
        return True

    def greedy_optimize(V, time_limit, rng):
        """Greedy removal + hill climbing optimization."""
        # Strategy: Start with a promising initial set
        # Sort by absolute value descending - larger elements are "safer"
        # Actually, let's try: start with all positive, or use a heuristic
        
        # Heuristic: elements with |x| > 5000 are less likely to be involved in conflicts
        # because a+b=c requires c to be in range
        # Let's try starting with the largest magnitude elements
        
        V_sorted = sorted(V, key=lambda x: abs(x), reverse=True)
        
        # Try multiple initializations and keep the best
        best_set = set()
        best_size = 0
        
        # Initialization 1: Greedy from largest magnitude
        S = set()
        for x in V_sorted:
            if can_add(x, S, list(S)):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Initialization 2: All positives (if any)
        positives = [x for x in V if x > 0]
        if positives:
            S_pos = set(positives)
            # Check if sum-free
            if is_sum_free(S_pos):
                if len(S_pos) > best_size:
                    best_size = len(S_pos)
                    best_set = set(S_pos)
        
        # Initialization 3: All negatives (if any)
        negatives = [x for x in V if x < 0]
        if negatives:
            S_neg = set(negatives)
            if is_sum_free(S_neg):
                if len(S_neg) > best_size:
                    best_size = len(S_neg)
                    best_set = set(S_neg)
        
        # Now do greedy removal from the best initialization
        S = set(best_set)
        removed = [x for x in V if x not in S]
        
        # Greedy removal phase
        while not is_sum_free(S) and time.time() < start_time + time_limit:
            conflicts = get_conflicts(S)
            if not conflicts:
                break
            
            max_deg = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_deg]
            x_to_remove = rng.choice(candidates)
            
            S.remove(x_to_remove)
            removed.append(x_to_remove)
        
        # Hill climbing: add back elements
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            rng.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return list(S)

    def optimize_with_restarts(V, time_limit, rng):
        """Multiple restarts with different random seeds for tie-breaking."""
        best_set = []
        best_size = 0
        n = len(V)
        
        # Allocate time per restart
        # For small n, more restarts; for large n, fewer
        if n <= 60:
            max_restarts = 8
        elif n <= 120:
            max_restarts = 5
        else:
            max_restarts = 3
        
        per_restart = time_limit / max_restarts
        
        for r in range(max_restarts):
            if time.time() > start_time + time_limit:
                break
            
            # Create a new RNG for this restart
            restart_rng = random.Random(rng.randint(0, 2**31))
            
            res = greedy_optimize(V, per_restart, restart_rng)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
            
            # Early stop if we've found a very good solution
            # Bourgain bound: (n+2)/3
            bourgain = (n + 2) / 3
            if best_size >= bourgain * 1.15:  # 15% above Bourgain
                break
        
        return best_set

    results = []
    
    # Time allocation: give more time to larger instances
    # Total available: ~115s
    # Instance 1 (60): ~15s
    # Instance 2 (120): ~30s  
    # Instance 3 (240): ~65s
    
    time_alloc = {
        (60, 801): 15.0,
        (120, 802): 30.0,
        (240, 803): 65.0,
    }
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        t_limit = time_alloc[(size, seed)]
        
        # Check remaining time
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < t_limit:
            t_limit = remaining * 0.9  # Use 90% of remaining
        
        rng = random.Random(42)  # Fixed seed for reproducibility
        result = optimize_with_restarts(V, t_limit, rng)
        results.append(result)
    
    return results
