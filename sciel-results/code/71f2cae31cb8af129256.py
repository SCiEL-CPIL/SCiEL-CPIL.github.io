import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 256
    
    # Find all primes up to N
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, N + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, N + 1) if sieve[i]]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # For each number 2..N, compute its prime factor set
    def prime_factor_set(n):
        s = set()
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                s.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            s.add(n)
        return s
    
    # Enumerate all valid prime subsets (product <= N)
    # A subset is valid if product of its primes <= N
    valid_subsets = []  # list of (frozenset_of_prime_indices, product_value)
    
    # Use recursion to enumerate subsets with product <= N
    def enumerate_subsets(idx, current_set, current_product):
        if time.time() > deadline - 5:
            return
        if current_set:
            valid_subsets.append((frozenset(current_set), current_product))
        for i in range(idx, num_primes):
            p = primes[i]
            new_product = current_product * p
            if new_product > N:
                break
            current_set.append(i)
            enumerate_subsets(i + 1, current_set, new_product)
            current_set.pop()
    
    enumerate_subsets(0, [], 1)
    
    # Build mapping from frozenset to index for quick lookup
    # For local search, we work with a selection of subsets
    
    # Each subset has: prime indices (frozenset), value (product)
    # We want to select a collection of non-overlapping subsets maximizing sum of values
    
    num_subsets = len(valid_subsets)
    subset_primes = [s[0] for s in valid_subsets]
    subset_values = [s[1] for s in valid_subsets]
    
    # Precompute overlap matrix? Too large. Instead, check on the fly.
    
    def is_valid_selection(selected_indices):
        used_primes = set()
        for idx in selected_indices:
            for p in subset_primes[idx]:
                if p in used_primes:
                    return False
                used_primes.add(p)
        return True
    
    def selection_sum(selected_indices):
        return sum(subset_values[i] for i in selected_indices)
    
    def selection_prime_set(selected_indices):
        s = set()
        for idx in selected_indices:
            s.update(subset_primes[idx])
        return s
    
    # Greedy initialization: sort subsets by value/size ratio or just value, pick greedily
    def greedy_init():
        selected = []
        used_primes = set()
        # Sort by value descending, then by size ascending
        order = sorted(range(num_subsets), key=lambda i: (-subset_values[i], len(subset_primes[i])))
        for idx in order:
            if time.time() > deadline - 5:
                break
            can_add = True
            for p in subset_primes[idx]:
                if p in used_primes:
                    can_add = False
                    break
            if can_add:
                selected.append(idx)
                used_primes.update(subset_primes[idx])
        return selected
    
    # Local search
    def local_search(selected, max_iters=5000):
        selected = list(selected)
        current_sum = selection_sum(selected)
        best_selected = list(selected)
        best_sum = current_sum
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding a subset
            # For efficiency, only try subsets that don't conflict
            used_primes = selection_prime_set(selected)
            
            # Try adding
            for idx in range(num_subsets):
                if time.time() > deadline - 5:
                    break
                if idx in selected:
                    continue
                can_add = True
                for p in subset_primes[idx]:
                    if p in used_primes:
                        can_add = False
                        break
                if can_add:
                    new_sum = current_sum + subset_values[idx]
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_selected = selected + [idx]
                        selected = best_selected
                        current_sum = new_sum
                        used_primes.update(subset_primes[idx])
                        improved = True
                    break  # Only try first valid add per iteration for speed
            
            if not improved:
                # Try swap: remove one, add one
                if not selected:
                    break
                remove_idx = random.choice(selected)
                new_selected = [i for i in selected if i != remove_idx]
                new_used = selection_prime_set(new_selected)
                new_sum = current_sum - subset_values[remove_idx]
                
                for idx in range(num_subsets):
                    if time.time() > deadline - 5:
                        break
                    if idx in new_selected:
                        continue
                    can_add = True
                    for p in subset_primes[idx]:
                        if p in new_used:
                            can_add = False
                            break
                    if can_add:
                        swap_sum = new_sum + subset_values[idx]
                        if swap_sum > best_sum:
                            best_sum = swap_sum
                            best_selected = new_selected + [idx]
                            selected = best_selected
                            current_sum = swap_sum
                            improved = True
                        break
            
            if not improved:
                # Try 2-opt: remove two, add two
                if len(selected) < 2:
                    break
                if random.random() < 0.5:
                    # Try removing one and adding two
                    remove_idx = random.choice(selected)
                    new_selected = [i for i in selected if i != remove_idx]
                    new_used = selection_prime_set(new_selected)
                    new_sum = current_sum - subset_values[remove_idx]
                    
                    # Try adding two
                    candidates = []
                    for idx in range(num_subsets):
                        if idx in new_selected:
                            continue
                        can_add = True
                        for p in subset_primes[idx]:
                            if p in new_used:
                                can_add = False
                                break
                        if can_add:
                            candidates.append(idx)
                    
                    if len(candidates) >= 2:
                        # Try pairs
                        for i1 in range(len(candidates)):
                            if time.time() > deadline - 5:
                                break
                            for i2 in range(i1 + 1, len(candidates)):
                                c1 = candidates[i1]
                                c2 = candidates[i2]
                                # Check if c1 and c2 are compatible
                                conflict = False
                                for p in subset_primes[c1]:
                                    if p in subset_primes[c2]:
                                        conflict = True
                                        break
                                if not conflict:
                                    two_sum = new_sum + subset_values[c1] + subset_values[c2]
                                    if two_sum > best_sum:
                                        best_sum = two_sum
                                        best_selected = new_selected + [c1, c2]
                                        selected = best_selected
                                        current_sum = two_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            continue
                
                # If no improvement, break or continue with random perturbation
                if random.random() < 0.1:
                    # Perturb: remove a random element
                    if selected:
                        remove_idx = random.choice(selected)
                        selected = [i for i in selected if i != remove_idx]
                        current_sum = selection_sum(selected)
                else:
                    break
        
        return best_selected, best_sum
    
    # Run multiple restarts
    best_overall = []
    best_overall_sum = 0
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if r == 0:
            selected = greedy_init()
        elif r == 1:
            # Random subset of single primes
            selected = [random.randint(0, num_subsets - 1) for _ in range(10)]
            # Filter to valid
            valid_sel = []
            used = set()
            for idx in selected:
                can_add = True
                for p in subset_primes[idx]:
                    if p in used:
                        can_add = False
                        break
                if can_add:
                    valid_sel.append(idx)
                    used.update(subset_primes[idx])
            selected = valid_sel
        else:
            # Perturbed greedy
            selected = greedy_init()
            if selected and random.random() < 0.5:
                # Remove a random element
                remove_idx = random.choice(selected)
                selected = [i for i in selected if i != remove_idx]
        
        result, result_sum = local_search(selected)
        if result_sum > best_overall_sum:
            best_overall_sum = result_sum
            best_overall = result
    
    # Convert selected subsets back to numbers
    # Each subset corresponds to a number (the product of its primes)
    # But we need to return the actual integers
    
    result_numbers = [1]  # Always include 1
    for idx in best_overall:
        prod = subset_values[idx]
        result_numbers.append(prod)
    
    # Verify: all distinct, all in 1..256, pairwise coprime
    # Check distinctness
    assert len(result_numbers) == len(set(result_numbers))
    # Check range
    for x in result_numbers:
        assert 1 <= x <= 256
    # Check pairwise coprime
    for i in range(len(result_numbers)):
        for j in range(i + 1, len(result_numbers)):
            assert math.gcd(result_numbers[i], result_numbers[j]) == 1
    
    return result_numbers
