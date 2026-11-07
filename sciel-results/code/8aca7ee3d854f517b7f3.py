import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 4:
            return False
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
        return False

    # Generate pool of candidates: Primes and Prime Powers > 20
    # Smaller numbers are more likely to divide sums of larger numbers?
    # Actually, smaller numbers divide MORE things. So large numbers are safer.
    pool = []
    for n in range(1, 257):
        if is_prime(n) or is_prime_power(n):
            if n > 20: # Avoid very small numbers
                pool.append(n)
                
    # Sort pool by value descending? 
    # Or keep random.
    
    def check_valid_addition(current_sums, x):
        """
        Returns new_sums set if x can be added, else None.
        """
        new_sums = {x}
        for s in current_sums:
            ns = x + s
            if ns in current_sums:
                return None
            if ns in new_sums:
                return None
            new_sums.add(ns)
        
        all_sums = current_sums.union(new_sums)
        
        for ns in new_sums:
            for t in all_sums:
                if ns == t:
                    continue
                if t % ns == 0:
                    return None
                if ns % t == 0:
                    return None
        return new_sums

    best_set = []
    
    # Strategy 1: Greedy on full pool
    greedy_set = set()
    greedy_sums = set()
    for x in sorted(pool, reverse=True):
        if time.time() > deadline - 10:
            break
        new_sums = check_valid_addition(greedy_sums, x)
        if new_sums is not None:
            greedy_set.add(x)
            greedy_sums = new_sums
    if len(greedy_set) > len(best_set):
        best_set = list(greedy_set)

    # Strategy 2: Randomized DFS on subsets
    # Use a fixed seed for reproducibility in testing, but random for eval?
    # The prompt implies deterministic behavior is not required, just valid output.
    # But using random might be risky if it finds nothing.
    # We already have a greedy baseline.
    
    random.seed(123)
    
    # We will try to improve the greedy set or find a better one.
    # Let's run DFS on a subset of the pool of size 15.
    # If we find a set of size > len(best_set), update.
    
    num_trials = 0
    while time.time() < deadline - 5:
        num_trials += 1
        k = 15
        if len(pool) < k:
            k = len(pool)
        sub_pool = random.sample(pool, k)
        
        # Sort sub_pool to help DFS? 
        # Maybe sort by value descending to find large sums early?
        sub_pool.sort(reverse=True)
        
        best_in_trial = []
        
        def dfs(idx, current_set, current_sums):
            if time.time() > deadline - 5:
                return
            
            if len(current_set) > len(best_in_trial):
                best_in_trial.clear()
                best_in_trial.extend(current_set)
                
            # Pruning: if len(current_set) + (len(sub_pool) - idx) <= len(best_in_trial):
            #    return

            for i in range(idx, len(sub_pool)):
                if time.time() > deadline - 5:
                    return
                x = sub_pool[i]
                new_sums = check_valid_addition(current_sums, x)
                if new_sums is not None:
                    dfs(i + 1, current_set | {x}, new_sums)

        dfs(0, set(), set())
        
        if len(best_in_trial) > len(best_set):
            best_set = best_in_trial
            # If we found a very large set, we might stop early?
            # Or continue to see if we can get even larger.
            
    # Final verification
    if not best_set:
        return [3]
        
    elems = sorted(best_set)
    sums = set()
    valid = True
    for mask in range(1, 1 << len(elems)):
        s = 0
        for i in range(len(elems)):
            if mask & (1 << i):
                s += elems[i]
        if s in sums:
            valid = False
            break
        sums.add(s)
        
    if valid:
        sums_list = sorted(sums)
        for i in range(len(sums_list)):
            for j in range(i+1, len(sums_list)):
                if sums_list[j] % sums_list[i] == 0:
                    valid = False
                    break
            if not valid:
                break
            
    if valid:
        return [int(x) for x in best_set]
    else:
        return [3]
