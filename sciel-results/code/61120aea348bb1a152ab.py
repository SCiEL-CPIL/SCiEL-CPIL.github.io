"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    best = []
    best_size = 0
    N = 1000
    limit = 2 # Max representations
    
    def try_add(x, chosen, counts):
        """
        Attempts to add x to chosen.
        Returns True if successful, False otherwise.
        Modifies counts and chosen in place if successful.
        """
        # Check if x can be added
        # Sums are x+y for y in chosen, and 2x
        # We need to ensure no sum count exceeds 'limit'
        
        # Optimization: Pre-check if any existing sum with x is already at limit
        # Actually, we just need to simulate the addition.
        
        # To be fast, we iterate and check.
        # We don't want to modify counts if it fails.
        # So we check first.
        
        # Sums to check:
        # 2*x
        # x + y for y in chosen
        
        # If counts.get(2*x, 0) >= limit, fail
        if counts.get(2 * x, 0) >= limit:
            return False
            
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= limit:
                return False
        
        # If we get here, it's safe to add
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for y in chosen:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        chosen.append(x)
        return True

    def greedy(order):
        chosen = []
        counts = {}
        for x in order:
            try_add(x, chosen, counts)
        return chosen

    def local_search(initial_set):
        """
        Perform local search by swapping elements.
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        # Rebuild counts for current_set
        counts = {}
        for i, a in enumerate(current_set):
            for b in current_set:
                if a <= b:
                    s = a + b
                    counts[s] = counts.get(s, 0) + 1
        
        # List of elements in set and not in set
        in_set = list(current_set)
        out_set = [x for x in range(1, N + 1) if x not in current_set]
        
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid getting stuck in the same local minimum
            random.shuffle(in_set)
            random.shuffle(out_set)
            
            for a in in_set:
                for b in out_set:
                    if a == b:
                        continue
                    # Try swap a -> b
                    # Check if removing a and adding b is valid
                    
                    # 1. Remove a from counts
                    # Sums involving a: a+a, a+y for y in current_set if y != a
                    # We need to be careful with the set logic.
                    
                    # Let's just construct the new set and check validity from scratch?
                    # That is O(N^2) which is too slow for many swaps.
                    
                    # Efficient swap check:
                    # New Set S' = S - {a} + {b}
                    # We need to check if all sums in S' have count <= 2.
                    
                    # The sums that change are those involving a or b.
                    # Sums NOT involving a or b remain unchanged and were valid.
                    # So we only need to check sums involving a or b.
                    
                    # Sums involving a in S:
                    # a+a, a+y (y in S, y != a)
                    # Sums involving b in S' (b is new):
                    # b+b, b+y (y in S, y != a, y != b)
                    
                    # However, the counts of these sums change.
                    # This is complex to track incrementally without a full rebuild or careful bookkeeping.
                    
                    # Given N=1000, a full rebuild for a candidate swap is O(|S|^2) ~ 1600 ops.
                    # Number of swaps to test: |S| * (N-|S|) ~ 40 * 960 = 38400.
                    # Total ops: 38400 * 1600 ~ 60 million.
                    # In Python, this might take 10-20 seconds per iteration.
                    # We have 120s. So maybe 5-10 iterations of this local search.
                    
                    # Let's try a simpler check:
                    # Create S_prime
                    # Check validity of S_prime
                    # If valid, check if we can extend it?
                    # For now, just accept if valid and size is same.
                    # To improve size, we need to extend.
                    
                    # Let's just do a greedy extension after a valid swap.
                    
                    S_prime = (current_set - {a}) | {b}
                    
                    # Check validity of S_prime
                    counts_prime = {}
                    valid = True
                    # Sort for efficiency? No, just iterate
                    sp_list = list(S_prime)
                    for i in range(len(sp_list)):
                        ai = sp_list[i]
                        for j in range(i, len(sp_list)):
                            aj = sp_list[j]
                            s = ai + aj
                            counts_prime[s] = counts_prime.get(s, 0) + 1
                            if counts_prime[s] > limit:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Try to extend S_prime greedily
                        # Use a random order for extension
                        remaining = [x for x in range(1, N + 1) if x not in S_prime]
                        random.shuffle(remaining)
                        extended_set = list(S_prime)
                        extended_counts = counts_prime
                        for x in remaining:
                             # We need a helper to add to existing counts
                             # Re-use try_add logic but on lists/dicts
                             # To avoid code duplication, let's just use the greedy function on the remaining?
                             # No, greedy starts from empty.
                             
                             # Let's implement a quick add check
                             ok = True
                             if extended_counts.get(2*x, 0) >= limit:
                                 ok = False
                             else:
                                 for y in extended_set:
                                     s = x + y
                                     if extended_counts.get(s, 0) >= limit:
                                         ok = False
                                         break
                             
                             if ok:
                                 extended_counts[2*x] = extended_counts.get(2*x, 0) + 1
                                 for y in extended_set:
                                     s = x + y
                                     extended_counts[s] = extended_counts.get(s, 0) + 1
                                 extended_set.append(x)
                             
                        if len(extended_set) > current_size:
                            current_set = set(extended_set)
                            current_size = len(current_set)
                            improved = True
                            # Update in_set/out_set
                            in_set = list(current_set)
                            out_set = [x for x in range(1, N + 1) if x not in current_set]
                            # Break inner loops to restart with new set
                            break
                if improved:
                    break
        return list(current_set)

    # 1. Initial Greedy Phases
    # Ascending
    res = greedy(range(1, N + 1))
    if len(res) > best_size:
        best_size = len(res)
        best = res

    # Descending
    res = greedy(range(N, 0, -1))
    if len(res) > best_size:
        best_size = len(res)
        best = res

    # Random Greedy
    start_time = time.time()
    time_limit = 110 # Leave 10s for local search
    
    seed = 0
    while time.time() - start_time < time_limit:
        random.seed(seed)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        res = greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best = res
        seed += 1
        
    # 2. Local Search on the best found set
    # Run a few iterations of local search
    # We'll do a simplified version: Try random swaps and greedy extension
    
    start_ls_time = time.time()
    ls_time_limit = 10
    
    current_best_set = set(best)
    current_best_size = best_size
    
    # We'll try a few random perturbations and greedy repairs
    for _ in range(5): # 5 rounds of local search
        if time.time() - start_ls_time > ls_time_limit:
            break
            
        # Pick a random subset of elements to remove (perturbation)
        # And then greedily add back
        # This is a form of "Iterated Greedy"
        
        # Remove 2-5 random elements
        remove_count = random.randint(2, 5)
        if len(current_best_set) > remove_count:
            to_remove = random.sample(list(current_best_set), remove_count)
            perturbed_set = current_best_set - set(to_remove)
            
            # Greedily add remaining numbers to perturbed_set
            # We need to rebuild counts for perturbed_set
            counts = {}
            p_list = list(perturbed_set)
            for i in range(len(p_list)):
                a = p_list[i]
                for j in range(i, len(p_list)):
                    b = p_list[j]
                    s = a + b
                    counts[s] = counts.get(s, 0) + 1
            
            # Add remaining
            remaining = [x for x in range(1, N + 1) if x not in perturbed_set]
            random.shuffle(remaining)
            
            for x in remaining:
                if counts.get(2 * x, 0) >= limit:
                    continue
                ok = True
                for y in p_list:
                    s = x + y
                    if counts.get(s, 0) >= limit:
                        ok = False
                        break
                if ok:
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    for y in p_list:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    p_list.append(x)
            
            if len(p_list) > current_best_size:
                current_best_size = len(p_list)
                current_best_set = set(p_list)
                best = p_list
                best_size = current_best_size

    return best
# EVOLVE-BLOCK-END
