import time
import random

def construct():
    start = time.time()
    DEADLINE = 135.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(x, Sset, S_list):
        """O(N) check: can x be added to S without violating sum-free?"""
        for a in S_list:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def greedy_construct(V, rng, time_limit, start_time):
        """
        Randomized greedy: build a sum-free set by adding elements in
        random order, skipping those that would create a conflict.
        """
        S = set()
        S_list = []
        Sset = S
        items = list(V)
        rng.shuffle(items)
        
        for x in items:
            if time.time() - start_time > time_limit:
                break
            if can_add(x, Sset, S_list):
                S.add(x)
                S_list.append(x)
        
        return S, S_list

    def local_search(S, S_list, V, rng, time_limit, start_time):
        """
        Perturbation-based local search:
        1. Try to add removed elements (hill climb)
        2. Periodically perturb: remove a few random elements, then re-add greedily
        """
        Sset = set(S)
        removed = [x for x in V if x not in Sset]
        best_S = set(S)
        best_S_list = list(S_list)
        best_size = len(S)
        
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() - start_time < time_limit:
            # Phase 1: Hill climb - try to add removed elements
            improved = False
            still_removed = []
            rng.shuffle(removed)
            for x in removed:
                if time.time() - start_time > time_limit:
                    still_removed.append(x)
                    continue
                if can_add(x, Sset, S_list):
                    S.add(x)
                    Sset.add(x)
                    S_list.append(x)
                    improved = True
                else:
                    still_removed.append(x)
            removed = still_removed
            
            if improved:
                no_improve_count = 0
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    best_S_list = list(S_list)
            else:
                no_improve_count += 1
            
            # Phase 2: Perturbation if stuck
            if no_improve_count >= max_no_improve or (not improved and len(removed) > 0 and rng.random() < 0.3):
                # Remove 2-5 random elements from S to create space
                n_remove = min(len(S_list), rng.randint(2, 5))
                to_remove = rng.sample(S_list, n_remove)
                for x in to_remove:
                    S.remove(x)
                    Sset.remove(x)
                    S_list.remove(x)
                    removed.append(x)
                
                # Re-greedy add from removed
                rng.shuffle(removed)
                for x in removed:
                    if time.time() - start_time > time_limit:
                        break
                    if can_add(x, Sset, S_list):
                        S.add(x)
                        Sset.add(x)
                        S_list.append(x)
                
                # Keep only what's not in S
                removed = [x for x in removed if x not in Sset]
                no_improve_count = 0
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    best_S_list = list(S_list)
        
        return best_S, best_S_list

    def sign_seed(V):
        """Create initial seed from sign-based partition."""
        pos = [x for x in V if x > 0]
        neg = [x for x in V if x < 0]
        # Pick the larger sign group, ensure sum-free
        if len(pos) >= len(neg):
            S = set(pos)
        else:
            S = set(neg)
        
        # Ensure sum-free by removing conflicts
        S_list = list(S)
        changed = True
        while changed and S_list:
            changed = False
            Sset = set(S_list)
            for i in range(len(S_list)):
                if time.time() - start > DEADLINE - 5:
                    break
                a = S_list[i]
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    if a + b in Sset:
                        # Remove the one that creates the conflict
                        S_list.remove(a)
                        changed = True
                        break
                if changed:
                    break
        
        return set(S_list), list(S_list)

    def random_seed(V, rng, size_hint):
        """Create a random initial seed of given size, then make sum-free."""
        n = min(size_hint, len(V))
        sample = set(rng.sample(V, n))
        S_list = list(sample)
        
        # Make sum-free by removing conflicts
        changed = True
        while changed and S_list:
            changed = False
            Sset = set(S_list)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    if a + b in Sset:
                        S_list.remove(a)
                        changed = True
                        break
                if changed:
                    break
        
        return set(S_list), list(S_list)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Emergency fallback: just return a valid sum-free subset
            S, _ = sign_seed(V)
            results.append(sorted(S))
            continue
        
        # Allocate time per instance
        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 3
        
        best_set = set()
        best_size = 0
        
        # Strategy 1: Sign-based seed + local search
        S_sign, S_sign_list = sign_seed(V)
        if len(S_sign) > best_size:
            best_size = len(S_sign)
            best_set = set(S_sign)
        
        if time.time() - start < DEADLINE - 5:
            t1 = min(time_per_instance * 0.3, DEADLINE - (time.time() - start) - 5)
            if t1 > 1:
                rng1 = random.Random(seed * 7 + 1)
                S1, _ = local_search(S_sign, S_sign_list, V, rng1, t1, start)
                if len(S1) > best_size:
                    best_size = len(S1)
                    best_set = set(S1)
        
        # Strategy 2: Randomized greedy construction + local search
        for trial in range(3):
            if time.time() - start > DEADLINE - 5:
                break
            t2 = min(time_per_instance * 0.2, DEADLINE - (time.time() - start) - 5)
            if t2 < 1:
                break
            
            rng2 = random.Random(seed * 13 + trial * 100 + 42)
            S_greedy, S_greedy_list = greedy_construct(V, rng2, t2 * 0.3, start)
            
            if len(S_greedy) > best_size:
                best_size = len(S_greedy)
                best_set = set(S_greedy)
            
            # Local search on greedy result
            if time.time() - start < DEADLINE - 5:
                t3 = min(t2 * 0.7, DEADLINE - (time.time() - start) - 5)
                if t3 > 0.5:
                    S2, _ = local_search(S_greedy, S_greedy_list, V, rng2, t3, start)
                    if len(S2) > best_size:
                        best_size = len(S2)
                        best_set = set(S2)
        
        # Strategy 3: Multiple random seeds + local search
        for trial in range(2):
            if time.time() - start > DEADLINE - 5:
                break
            t4 = min(time_per_instance * 0.15, DEADLINE - (time.time() - start) - 5)
            if t4 < 1:
                break
            
            rng3 = random.Random(seed * 31 + trial * 500 + 7)
            seed_size = max(10, size // 3)
            S_rand, S_rand_list = random_seed(V, rng3, seed_size)
            
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_set = set(S_rand)
            
            if time.time() - start < DEADLINE - 5:
                t5 = min(t4 * 0.8, DEADLINE - (time.time() - start) - 5)
                if t5 > 0.5:
                    S3, _ = local_search(S_rand, S_rand_list, V, rng3, t5, start)
                    if len(S3) > best_size:
                        best_size = len(S3)
                        best_set = set(S3)
        
        # Final verification: ensure best_set is sum-free
        if not is_sum_free(best_set):
            # Fix by removing conflicts
            best_list = list(best_set)
            best_set = set()
            for x in best_list:
                if can_add(x, best_set, list(best_set)):
                    best_set.add(x)
        
        results.append(sorted(best_set))
    
    return results
