import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S_set):
        """Check if S_set is sum-free. O(n^2) but with early exit."""
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def compute_conflicts(S_set):
        """Return dict: element -> number of conflicts involving it.
        A conflict is a triple (a, b, c) with a+b=c, all in S_set.
        Each such triple contributes 1 to conflicts[a], conflicts[b], conflicts[c].
        """
        conflicts = {}
        for a in S_set:
            conflicts[a] = 0
        for a in S_set:
            for b in S_set:
                s = a + b
                if s in S_set:
                    conflicts[a] += 1
                    conflicts[b] += 1
                    conflicts[s] += 1
        return conflicts

    def greedy_remove(S_set, removed_list, time_limit, rng):
        """Iteratively remove the element with most conflicts until sum-free."""
        S = set(S_set)
        removed = list(removed_list)
        
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break
            
            conflicts = compute_conflicts(S)
            max_deg = max(conflicts.values()) if conflicts else 0
            if max_deg == 0:
                break
            
            candidates = [x for x in S if conflicts[x] == max_deg]
            x_to_remove = rng.choice(candidates)
            S.discard(x_to_remove)
            if x_to_remove in removed:
                removed.remove(x_to_remove)
        
        return S, removed

    def can_add(x, S_set):
        """Check if x can be added to S_set without violating sum-free."""
        for a in S_set:
            if (x + a) in S_set or (x - a) in S_set:
                return False
        return True

    def hill_climb_add(S_set, removed_list, time_limit, rng):
        """Try to add elements back from removed_list. Multiple passes."""
        S = set(S_set)
        removed = list(removed_list)
        
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            rng.shuffle(removed)
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                if can_add(x, S):
                    S.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return S, removed

    def random_seeded_optimize(V, time_limit, rng):
        """Start with random subset, then remove conflicts and add back."""
        # Start with a random subset (about 60% of elements)
        keep_prob = 0.6
        S = set(x for x in V if rng.random() < keep_prob)
        removed = [x for x in V if x not in S]
        
        # Greedy removal to make sum-free
        S, removed = greedy_remove(S, removed, time_limit, rng)
        
        # Hill climbing to add back
        S, removed = hill_climb_add(S, removed, time_limit, rng)
        
        return list(S)

    def parity_seeded_optimize(V, time_limit, rng):
        """Start with odd numbers, then optimize."""
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        
        S = set(odds)
        removed = list(evens)
        
        # Greedily add evens that don't conflict
        for x in evens:
            if can_add(x, S):
                S.add(x)
                removed.remove(x)
        
        # Hill climbing
        S, removed = hill_climb_add(S, removed, time_limit, rng)
        
        return list(S)

    def topk_seeded_optimize(V, time_limit, rng):
        """Start with largest |x| values (more likely to be sum-free as they're hard to hit), then optimize."""
        sorted_V = sorted(V, key=lambda x: abs(x), reverse=True)
        # Take top 50%
        k = max(1, len(sorted_V) // 2)
        S = set(sorted_V[:k])
        removed = sorted_V[k:]
        
        # Greedy removal
        S, removed = greedy_remove(S, removed, time_limit, rng)
        
        # Hill climbing
        S, removed = hill_climb_add(S, removed, time_limit, rng)
        
        return list(S)

    def sign_seeded_optimize(V, time_limit, rng):
        """Start with all positive or all negative numbers."""
        positives = [x for x in V if x > 0]
        negatives = [x for x in V if x < 0]
        
        # Try both signs and pick the larger
        best_S = set()
        best_removed = []
        
        for initial_set, initial_removed in [(set(positives), negatives), (set(negatives), positives)]:
            S = set(initial_set)
            rem = list(initial_removed)
            S, rem = greedy_remove(S, rem, min(time_limit, 3.0), rng)
            S, rem = hill_climb_add(S, rem, time_limit, rng)
            if len(S) > len(best_S):
                best_S = S
                best_removed = rem
        
        return list(best_S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Allocate time based on instance size
        if size <= 60:
            base_time = 12.0
        elif size <= 120:
            base_time = 18.0
        else:
            base_time = 28.0
        
        # Try multiple strategies with different seeds
        strategies = [
            ("random", random_seeded_optimize),
            ("parity", parity_seeded_optimize),
            ("topk", topk_seeded_optimize),
            ("sign", sign_seeded_optimize),
        ]
        
        # Run each strategy with time budget
        time_per_strategy = base_time / len(strategies)
        
        for strat_name, strat_func in strategies:
            if time.time() > start_time + DEADLINE - 5:
                break
            
            remaining = min(time_per_strategy, DEADLINE - (time.time() - start_time))
            if remaining < 1.0:
                break
            
            rng = random.Random(seed * 31 + len(strat_name))
            res = strat_func(V, remaining, rng)
            
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
        
        # If time remains, do more restarts with random strategy (most flexible)
        elapsed = time.time() - start_time
        remaining_total = DEADLINE - elapsed - 5
        
        if remaining_total > 3.0 and best_size > 0:
            extra_restarts = min(5, int(remaining_total / 4.0))
            for r in range(extra_restarts):
                if time.time() > start_time + DEADLINE - 3:
                    break
                
                t_left = min(6.0, (DEADLINE - 3) - (time.time() - start_time))
                if t_left < 1.0:
                    break
                
                # Alternate between random and topk
                if r % 2 == 0:
                    strat_func = random_seeded_optimize
                else:
                    strat_func = topk_seeded_optimize
                
                rng = random.Random(seed * 1000 + r * 7919 + 42)
                res = strat_func(V, t_left, rng)
                
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = res
        
        results.append(best_set)

    return results
