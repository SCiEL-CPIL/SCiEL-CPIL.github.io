import time
import random

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    Uses exponent-vector collision checking for speed, combined with
    randomized greedy restarts and swap-based local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Prime factorization for each number 2-40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(n):
        """Return exponent vector as tuple for primes list."""
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                exponents[i] += 1
                n //= p
        return tuple(exponents)
    
    # Precompute exponent vectors for all numbers 2-40
    exp_vectors = {}
    for n in range(2, 41):
        exp_vectors[n] = factorize(n)
    
    zero = (0,) * num_primes
    
    def _is_valid_set(nums):
        """Check if all subset products of nums are distinct using exponent vectors."""
        if not nums:
            return True
        sums = {zero}
        for n in nums:
            v = exp_vectors[n]
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums |= new_sums
        return True
    
    def _can_add(nums, candidate, sums):
        """Check if candidate can be added to set with given sum set."""
        v = exp_vectors[candidate]
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False
        return True
    
    def _greedy_build(order, deadline):
        """Greedy build in given order, tracking sums incrementally."""
        selected = []
        sums = {zero}
        for n in order:
            if time.time() > deadline:
                break
            v = exp_vectors[n]
            # Check collision
            collision = False
            new_sums_list = []
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(num_primes))
                if ns in sums:
                    collision = True
                    break
                new_sums_list.append(ns)
            if not collision:
                sums.update(new_sums_list)
                selected.append(n)
        return selected, sums
    
    def _local_search(current, sums, deadline):
        """Try swap-based local search to improve the set."""
        best = list(current)
        best_size = len(best)
        best_sums = set(sums)
        candidates = [x for x in range(2, 41) if x not in best]
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for c in candidates:
                    if time.time() > deadline:
                        break
                    # Try swapping best[i] with c
                    trial = list(best)
                    trial[i] = c
                    
                    # Check validity of trial set
                    trial_sums = {zero}
                    valid = True
                    for n in trial:
                        v = exp_vectors[n]
                        new_prods = set()
                        for s in trial_sums:
                            ns = tuple(s[i2] + v[i2] for i2 in range(num_primes))
                            if ns in trial_sums:
                                valid = False
                                break
                            new_prods.add(ns)
                        if not valid:
                            break
                        trial_sums |= new_prods
                    
                    if valid:
                        # Try to add more elements
                        trial2 = list(trial)
                        tp = set(trial_sums)
                        for c2 in range(2, 41):
                            if time.time() > deadline:
                                break
                            if c2 in trial2:
                                continue
                            if _can_add(trial2, c2, tp):
                                v = exp_vectors[c2]
                                for s in list(tp):
                                    tp.add(tuple(s[i2] + v[i2] for i2 in range(num_primes)))
                                trial2.append(c2)
                        if len(trial2) > best_size:
                            best = trial2
                            best_size = len(best)
                            best_sums = tp
                            improved = True
                            break
                if improved:
                    break
        return best, best_sums
    
    # Start with a good deterministic set as baseline
    base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    if _is_valid_set(base_set):
        best = list(base_set)
        # Compute sums for best
        best_sums = {zero}
        for n in best:
            v = exp_vectors[n]
            for s in list(best_sums):
                best_sums.add(tuple(s[i] + v[i] for i in range(num_primes)))
    else:
        best = []
        best_sums = {zero}
    
    best_size = len(best)
    all_nums = list(range(2, 41))
    
    # Multiple deterministic strategies
    strategies = [
        lambda x: x,  # ascending
        lambda x: -x,  # descending
        lambda x: (x in [2,4,8,16,32], x),  # powers of 2 first
        lambda x: (x in [3,9,27], x),  # powers of 3 first
        lambda x: sum(exp_vectors[x]),  # total exponent count
        lambda x: max(exp_vectors[x]),  # max exponent
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        ordered = sorted(range(2, 41), key=strategy)
        current_set, current_sums = _greedy_build(ordered, deadline)
        if len(current_set) > best_size:
            best = current_set
            best_size = len(best)
            best_sums = current_sums
    
    # Randomized restarts with local search
    while time.time() < deadline:
        random_order = list(all_nums)
        random.shuffle(random_order)
        current_set, current_sums = _greedy_build(random_order, deadline)
        
        # Local search
        if len(current_set) >= best_size:
            improved_set, improved_sums = _local_search(current_set, current_sums, deadline)
            if len(improved_set) > best_size:
                best = improved_set
                best_size = len(best)
                best_sums = improved_sums
    
    # Final verification
    if not _is_valid_set(best):
        # Fallback to known good solution
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 16, 9, 25]
    
    return [int(x) for x in best]
