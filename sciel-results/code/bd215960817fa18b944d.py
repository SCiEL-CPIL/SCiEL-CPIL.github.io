import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    # Precompute: for each possible sum value v (1..255*30=7650 max),
    # which larger values are multiples of v (i.e., v divides them)
    # We'll use this to quickly check conflicts.
    # Actually, let's think about the max possible sum: 30 elements * 256 = 7680
    MAX_SUM = 30 * 256  # 7680

    # For each value v in 1..MAX_SUM, precompute the set of multiples
    # that are also in range. But that's too big. Instead, we'll check
    # divisibility on the fly but use a smarter method.

    def get_subset_sums(S):
        """Get all non-empty subset sums of S as a sorted list."""
        sums = set()
        for i in range(len(S)):
            x = S[i]
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums |= new_sums
        return sorted(sums)

    def check_conflict(sums_list):
        """Check if any two distinct sums divide each other.
        sums_list must be sorted. Returns True if conflict exists."""
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            # Check if a divides any b > a in the list
            # b must be a multiple of a: b = 2a, 3a, 4a, ...
            for k in range(2, MAX_SUM // a + 1):
                b = a * k
                if b > sums_list[-1]:
                    break
                # Binary search for b in sums_list
                # Since sums_list is sorted, use bisect
                lo, hi = i + 1, n - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if sums_list[mid] == b:
                        return True
                    elif sums_list[mid] < b:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        return False

    def is_valid(S):
        """Check if all non-empty subset sums are divisibility-incomparable."""
        if not S:
            return True
        sums = get_subset_sums(S)
        return not check_conflict(sums)

    def can_add_incremental(S, x, sums_set, sums_sorted):
        """Try to add x to S. sums_set is the set of current subset sums,
        sums_sorted is the sorted list. Returns (ok, new_sums_set, new_sums_sorted)."""
        # New sums: x and s+x for each s in sums_set
        new_sums_set = set(sums_set)
        new_sums_set.add(x)
        for s in sums_set:
            new_sums_set.add(s + x)
        
        new_sums_sorted = sorted(new_sums_set)
        
        if check_conflict(new_sums_sorted):
            return False, None, None
        
        return True, new_sums_set, new_sums_sorted

    # Candidate pool: focus on numbers that are "hard to divide"
    # Use numbers in a good range. Larger numbers have fewer multiples in range.
    # But we also need enough candidates.
    
    # Strategy: Use a wide range but prioritize numbers with few small divisors
    # Numbers that are prime or product of large primes are good.
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Build candidate pool: all numbers from 1 to 256
    # Weight by "divisibility resistance"
    all_candidates = list(range(1, 257))
    
    # Score each candidate: fewer divisors = better
    def divisor_count(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # For the search, we'll use a backtracking approach with a good ordering
    # Key insight: start with large numbers (fewer multiples in range)
    # and prefer numbers with few divisors
    
    # Pre-sort candidates by a heuristic: prefer large, prefer few divisors
    candidate_order = sorted(all_candidates, key=lambda x: (-x, divisor_count(x)))
    
    # Also prepare a prime-only pool for some strategies
    primes_pool = [x for x in all_candidates if is_prime(x)]
    primes_order = sorted(primes_pool, reverse=True)
    
    # Backtracking search with time limit
    def backtrack_search(candidates, max_size, time_limit):
        """Backtracking search for the largest valid set.
        Returns the best set found."""
        best_set = []
        best_size = 0
        deadline = time.time() + time_limit
        
        # We'll use a recursive backtracking with pruning
        # State: current set, current sums
        def backtrack(current_set, sums_set, sums_sorted, start_idx):
            nonlocal best_set, best_size, deadline
            
            if time.time() > deadline:
                return
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            
            if len(current_set) >= max_size:
                return
            
            # Try adding each candidate from start_idx onwards
            for i in range(start_idx, len(candidates)):
                if time.time() > deadline:
                    return
                
                x = candidates[i]
                if x in current_set:
                    continue
                
                ok, new_sums_set, new_sums_sorted = can_add_incremental(
                    current_set, x, sums_set, sums_sorted
                )
                
                if ok:
                    current_set.append(x)
                    backtrack(current_set, new_sums_set, new_sums_sorted, i + 1)
                    current_set.pop()
                
                # Pruning: if we've already found a set of size max_size, stop
                if best_size >= max_size:
                    return
        
        backtrack([], set(), [], 0)
        return best_set

    # Simulated annealing for local improvement
    def simulated_annealing(initial_set, time_limit):
        """Improve a set using simulated annealing.
        Moves: add an element, remove an element, swap an element."""
        best_set = initial_set[:]
        best_sums = get_subset_sums(best_set)
        best_sums_set = set(best_sums)
        
        current_set = initial_set[:]
        current_sums_set = set(current_sums)
        current_sums_sorted = sorted(current_sums_set)
        
        deadline = time.time() + time_limit
        temperature = 100.0
        cooling_rate = 0.95
        
        # Candidate pool for additions/swaps
        add_pool = [x for x in all_candidates if x not in current_set]
        
        while time.time() < deadline and temperature > 0.1:
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and add_pool:
                x = random.choice(add_pool)
                ok, new_sums_set, new_sums_sorted = can_add_incremental(
                    current_set, x, current_sums_set, current_sums_sorted
                )
                if ok:
                    new_set = current_set + [x]
                    # Accept if valid (always accept valid improvements)
                    current_set = new_set
                    current_sums_set = new_sums_set
                    current_sums_sorted = new_sums_sorted
                    add_pool = [y for y in add_pool if y != x]
                    if len(current_set) > len(best_set):
                        best_set = current_set[:]
                        best_sums_set = set(current_sums_set)
            
            elif move_type == 'remove' and len(current_set) > 1:
                idx = random.randint(0, len(current_set) - 1)
                removed = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_sums = get_subset_sums(new_set) if new_set else []
                new_sums_set = set(new_sums)
                new_sums_sorted = sorted(new_sums_set)
                
                # Accept with probability based on temperature
                # We prefer to keep larger sets, so only accept removal if it helps
                # For simplicity, accept removals that allow future additions
                if random.random() < temperature / 100.0:
                    current_set = new_set
                    current_sums_set = new_sums_set
                    current_sums_sorted = new_sums_sorted
                    if removed not in add_pool:
                        add_pool.append(removed)
            
            elif move_type == 'swap' and current_set and add_pool:
                idx = random.randint(0, len(current_set) - 1)
                old = current_set[idx]
                new = random.choice(add_pool)
                
                new_set = current_set[:idx] + [new] + current_set[idx+1:]
                new_sums = get_subset_sums(new_set) if new_set else []
                new_sums_set = set(new_sums)
                new_sums_sorted = sorted(new_sums_set)
                
                if not check_conflict(new_sums_sorted):
                    current_set = new_set
                    current_sums_set = new_sums_set
                    current_sums_sorted = new_sums_sorted
                    add_pool.remove(new)
                    add_pool.append(old)
                    if len(current_set) > len(best_set):
                        best_set = current_set[:]
                        best_sums_set = set(current_sums_set)
            
            temperature *= cooling_rate
        
        return best_set

    # Multi-strategy search
    best_overall = []
    
    # Strategy 1: Backtracking with prime-heavy pool
    if time.time() < deadline:
        t = min(20.0, deadline - time.time() - 5.0)
        if t > 2:
            result = backtrack_search(primes_order, 30, t)
            if len(result) > len(best_overall):
                best_overall = result
    
    # Strategy 2: Backtracking with full candidate pool (descending)
    if time.time() < deadline:
        t = min(20.0, deadline - time.time() - 5.0)
        if t > 2:
            result = backtrack_search(candidate_order, 30, t)
            if len(result) > len(best_overall):
                best_overall = result
    
    # Strategy 3: Backtracking with descending order
    if time.time() < deadline:
        t = min(15.0, deadline - time.time() - 5.0)
        if t > 2:
            desc_order = sorted(all_candidates, reverse=True)
            result = backtrack_search(desc_order, 30, t)
            if len(result) > len(best_overall):
                best_overall = result
    
    # Strategy 4: Simulated annealing starting from best found
    if best_overall and time.time() < deadline:
        t = min(30.0, deadline - time.time() - 5.0)
        if t > 5:
            result = simulated_annealing(best_overall, t)
            if len(result) > len(best_overall):
                best_overall = result
    
    # Strategy 5: Multiple random restarts with backtracking
    if time.time() < deadline:
        t = min(15.0, deadline - time.time() - 5.0)
        if t > 3:
            for _ in range(3):
                if time.time() > deadline:
                    break
                shuffled = all_candidates[:]
                random.shuffle(shuffled)
                # Sort by a random-ish heuristic
                shuffled.sort(key=lambda x: (-x * random.random(), divisor_count(x)))
                t_single = t / 3.0
                result = backtrack_search(shuffled, 30, t_single)
                if len(result) > len(best_overall):
                    best_overall = result
    
    # Final validation and cleanup
    if best_overall:
        # Verify validity
        if not is_valid(best_overall):
            # Fallback to a known valid small set
            best_overall = [256, 255, 254, 253, 251]
            if not is_valid(best_overall):
                best_overall = [256, 255, 251]
                if not is_valid(best_overall):
                    best_overall = [256, 255]
        
        # Ensure distinct and in range
        best_overall = list(set(best_overall))
        best_overall = [x for x in best_overall if 1 <= x <= 256]
        best_overall.sort()
        
        # Ensure Python int type
        result = [int(x) for x in best_overall]
        return result
    else:
        # Minimal valid fallback
        fallback = [256, 255, 251]
        if is_valid(fallback):
            return [int(x) for x in fallback]
        return [int(x) for x in [256, 255]]

# EVOLVE-BLOCK-END
