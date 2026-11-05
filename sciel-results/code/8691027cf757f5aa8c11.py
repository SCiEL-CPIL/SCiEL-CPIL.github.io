"""Improved construction for the increasing-unique-consecutive-block-sums problem."""
import time
import random
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 58.0  # safety margin before 60s

    def compute_sums(values):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(values)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += values[j]
                if total in sums:
                    return None  # invalid
                sums.add(total)
        return sums

    def greedy_sequence(forward=True, time_deadline=None):
        """
        Greedy construction. If forward=True, build an increasing sequence from low to high.
        If forward=False, build a decreasing sequence from high to low, then reverse.
        """
        values = []
        sums = set()
        if forward:
            last_val = 0
            while last_val < 256:
                if time_deadline and time.time() > time_deadline:
                    break
                candidates = []
                for x in range(last_val + 1, 257):
                    # Check if x can be appended
                    total = x
                    ok = True
                    if total in sums:
                        ok = False
                    else:
                        for a in reversed(values):
                            total += a
                            if total in sums:
                                ok = False
                                break
                    if ok:
                        candidates.append(x)
                if not candidates:
                    break
                x = candidates[0]  # pick smallest
                values.append(x)
                last_val = x
                total = x
                sums.add(total)
                for a in reversed(values[:-1]):
                    total += a
                    sums.add(total)
        else:
            next_val = 257
            while next_val > 1:
                if time_deadline and time.time() > time_deadline:
                    break
                candidates = []
                for x in range(next_val - 1, 0, -1):
                    total = x
                    ok = True
                    if total in sums:
                        ok = False
                    else:
                        for a in reversed(values):
                            total += a
                            if total in sums:
                                ok = False
                                break
                    if ok:
                        candidates.append(x)
                if not candidates:
                    break
                x = candidates[0]  # pick largest (since we iterate descending)
                values.append(x)
                next_val = x
                total = x
                sums.add(total)
                for a in reversed(values[:-1]):
                    total += a
                    sums.add(total)
            values = values[::-1]
        return values

    def randomized_trial(strategy, rand, time_deadline):
        """
        Randomized greedy with different strategies.
        strategy: 'random', 'large', 'small', 'biased', 'spread', 'gap', 'prime', 'fib'
        """
        values = []
        sums = set()
        last_val = 0
        
        # Precompute some helper data
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        
        while last_val < 256:
            if time.time() > time_deadline:
                break
            
            candidates = []
            for x in range(last_val + 1, 257):
                total = x
                ok = True
                if total in sums:
                    ok = False
                else:
                    for a in reversed(values):
                        total += a
                        if total in sums:
                            ok = False
                            break
                if ok:
                    candidates.append(x)
            
            if not candidates:
                break
            
            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                k = min(15, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                k = min(15, len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'biased':
                # Bias towards values that maximize the gap to existing sums
                if sums:
                    avg_sum = sum(sums) / len(sums)
                    scored = [(abs(x - avg_sum) + rand.random() * 0.1, x) for x in candidates]
                    scored.sort(key=lambda t: -t[0])
                    x = scored[0][1]
                else:
                    x = rand.choice(candidates)
            elif strategy == 'spread':
                # Pick from the middle 50% of candidates
                n = len(candidates)
                lo = n // 4
                hi = 3 * n // 4
                if hi > lo:
                    x = rand.choice(candidates[lo:hi])
                else:
                    x = rand.choice(candidates)
            elif strategy == 'gap':
                # Try to maximize the minimum gap to existing sums
                if sums:
                    def min_gap(val):
                        return min(abs(val - s) for s in sums) if sums else 0
                    scored = [(min_gap(x) + rand.random() * 0.01, x) for x in candidates]
                    scored.sort(key=lambda t: -t[0])
                    x = scored[0][1]
                else:
                    x = rand.choice(candidates)
            elif strategy == 'prime':
                # Prefer primes
                prime_candidates = [x for x in candidates if x in primes]
                if prime_candidates:
                    x = rand.choice(prime_candidates)
                else:
                    x = rand.choice(candidates)
            elif strategy == 'fib':
                # Prefer Fibonacci-like numbers
                fib_candidates = [x for x in candidates if x in fibs]
                if fib_candidates:
                    x = rand.choice(fib_candidates)
                else:
                    # Pick from candidates closest to next Fibonacci
                    next_fib = min(fibs, key=lambda f: abs(f - last_val - 1))
                    x = min(candidates, key=lambda c: abs(c - next_fib))
            else:
                x = rand.choice(candidates)
            
            values.append(x)
            last_val = x
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        
        return values

    def local_search(initial_seq, time_deadline):
        """
        Try to improve a sequence by local modifications:
        - Try replacing an element with a nearby valid one
        - Try inserting a new element
        - Try removing an element to make room, then re-extend
        """
        values = list(initial_seq)
        
        best_len = len(values)
        best_seq = values[:]
        
        def is_valid_seq(seq):
            s = set()
            for i, v in enumerate(seq):
                total = v
                if total in s:
                    return False
                s.add(total)
                for j in range(i - 1, -1, -1):
                    total += seq[j]
                    if total in s:
                        return False
                    s.add(total)
            return True
        
        def extend_from_end(seq, time_deadline):
            """Greedily extend sequence from the end."""
            result = list(seq)
            sums = set()
            for i, v in enumerate(result):
                total = v
                sums.add(total)
                for j in range(i - 1, -1, -1):
                    total += result[j]
                    sums.add(total)
            
            last_val = result[-1] if result else 0
            while last_val < 256:
                if time.time() > time_deadline:
                    break
                for x in range(last_val + 1, 257):
                    total = x
                    ok = True
                    if total in sums:
                        ok = False
                    else:
                        for a in reversed(result):
                            total += a
                            if total in sums:
                                ok = False
                                break
                    if ok:
                        result.append(x)
                        last_val = x
                        sums.add(total)
                        for a in reversed(result[:-1]):
                            total += a
                            sums.add(total)
                        break
                else:
                    break
            return result
        
        # Strategy 1: Try inserting elements at various positions
        for _ in range(100):
            if time.time() > time_deadline:
                break
            
            if len(values) < 256:
                pos = random.randint(0, len(values))
                if pos == 0:
                    lo = 1
                    hi = values[0] - 1 if values else 256
                elif pos == len(values):
                    lo = values[-1] + 1
                    hi = 256
                else:
                    lo = values[pos - 1] + 1
                    hi = values[pos] - 1
                
                if lo <= hi:
                    for _ in range(5):
                        x = random.randint(lo, hi)
                        new_seq = values[:pos] + [x] + values[pos:]
                        if is_valid_seq(new_seq):
                            values = new_seq
                            if len(values) > best_len:
                                best_len = len(values)
                                best_seq = values[:]
                            break
        
        # Strategy 2: Remove an element and re-extend (breaks local optima)
        for _ in range(50):
            if time.time() > time_deadline:
                break
            
            if len(values) > 2:
                # Remove a random element (not first or last to keep it interesting)
                remove_idx = random.randint(1, len(values) - 2)
                shortened = values[:remove_idx] + values[remove_idx+1:]
                # Re-extend greedily from the end
                extended = extend_from_end(shortened, time_deadline)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_seq = extended[:]
                    values = extended
        
        return best_seq

    def hill_climb(initial_seq, time_deadline):
        """
        Hill climbing with backtracking: try to extend the sequence, and if stuck,
        backtrack to a good prefix and try different continuations.
        """
        values = list(initial_seq)
        best_seq = values[:]
        best_len = len(values)
        
        def is_valid_seq(seq):
            s = set()
            for i, v in enumerate(seq):
                total = v
                if total in s:
                    return False
                s.add(total)
                for j in range(i - 1, -1, -1):
                    total += seq[j]
                    if total in s:
                        return False
                    s.add(total)
            return True
        
        def get_sums(seq):
            s = set()
            for i, v in enumerate(seq):
                total = v
                s.add(total)
                for j in range(i - 1, -1, -1):
                    total += seq[j]
                    s.add(total)
            return s
        
        # Main loop: try to extend, backtrack if stuck
        while time.time() < time_deadline:
            improved = False
            last_val = values[-1] if values else 0
            
            # Try to extend from the end
            for x in range(last_val + 1, 257):
                new_seq = values + [x]
                if is_valid_seq(new_seq):
                    values = new_seq
                    if len(values) > best_len:
                        best_len = len(values)
                        best_seq = values[:]
                    improved = True
                    break
            
            if not improved and len(values) > 3:
                # Backtrack: try removing the last few elements and trying different values
                for backtrack_depth in [1, 2, 3]:
                    if len(values) <= backtrack_depth:
                        continue
                    prefix = values[:-backtrack_depth]
                    prefix_sums = get_sums(prefix)
                    prefix_last = prefix[-1] if prefix else 0
                    
                    # Try different continuations from this prefix
                    tried = set()
                    for x in range(prefix_last + 1, 257):
                        if x in tried:
                            continue
                        tried.add(x)
                        
                        # Build a candidate continuation
                        candidate = prefix + [x]
                        candidate_sums = set(prefix_sums)
                        total = x
                        candidate_sums.add(total)
                        for a in reversed(prefix):
                            total += a
                            if total in candidate_sums:
                                break
                            candidate_sums.add(total)
                        else:
                            # Valid so far, try to extend further
                            extended = list(candidate)
                            ext_sums = set(candidate_sums)
                            ext_last = x
                            while ext_last < 256:
                                found_next = False
                                for y in range(ext_last + 1, 257):
                                    total2 = y
                                    ok = True
                                    if total2 in ext_sums:
                                        ok = False
                                    else:
                                        for a in reversed(extended):
                                            total2 += a
                                            if total2 in ext_sums:
                                                ok = False
                                                break
                                    if ok:
                                        extended.append(y)
                                        ext_last = y
                                        ext_sums.add(total2)
                                        for a in reversed(extended[:-1]):
                                            total2 += a
                                            ext_sums.add(total2)
                                        found_next = True
                                        break
                                if not found_next:
                                    break
                            
                            if len(extended) > best_len:
                                best_len = len(extended)
                                best_seq = extended[:]
                                values = extended
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return best_seq

    # Phase 1: Baselines
    best_seq = greedy_sequence(forward=True)
    best_len = len(best_seq)
    
    dec_seq = greedy_sequence(forward=False)
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)
    
    # Phase 2: Randomized trials
    strategies = ['random', 'large', 'small', 'biased', 'spread', 'gap', 'prime', 'fib', 'random', 'large', 'biased', 'gap']
    trial_count = 0
    phase2_deadline = start_time + time_limit * 0.5
    
    while time.time() < phase2_deadline:
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count * 1000 + 42)
        seq = randomized_trial(strategy, rand, time.time() + 1.0)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        
        trial_count += 1
    
    # Phase 3: Local search and hill climbing on the best sequence
    phase3_deadline = start_time + time_limit * 0.8
    
    # Try local search from the best sequence
    ls_seq = local_search(best_seq, phase3_deadline)
    if len(ls_seq) > best_len:
        best_seq = ls_seq
        best_len = len(best_seq)
    
    # Try hill climbing
    hc_seq = hill_climb(best_seq, phase3_deadline)
    if len(hc_seq) > best_len:
        best_seq = hc_seq
        best_len = len(hc_seq)
    
    # Phase 4: More randomized trials if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        trial_count = 1000
        while time.time() < start_time + time_limit - 0.5:
            strategy = strategies[trial_count % len(strategies)]
            rand = random.Random(trial_count * 777 + 13)
            seq = randomized_trial(strategy, rand, time.time() + 0.5)
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
            
            # Occasionally do a quick hill climb on the new sequence
            if len(seq) >= best_len - 1 and time.time() < start_time + time_limit - 1.0:
                hc = hill_climb(seq, time.time() + 0.3)
                if len(hc) > best_len:
                    best_seq = hc
                    best_len = len(hc)
            
            trial_count += 1
    
    # Final polish: one more local search + hill climb if time allows
    if time.time() < start_time + time_limit - 0.5:
        final_ls = local_search(best_seq, start_time + time_limit - 0.3)
        if len(final_ls) > best_len:
            best_seq = final_ls
            best_len = len(best_seq)
        
        if time.time() < start_time + time_limit - 0.3:
            final_hc = hill_climb(best_seq, start_time + time_limit - 0.1)
            if len(final_hc) > best_len:
                best_seq = final_hc
                best_len = len(best_seq)
    
    return best_seq

# EVOLVE-BLOCK-END
