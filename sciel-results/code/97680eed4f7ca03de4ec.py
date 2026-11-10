"""Structural change: simulated annealing / random restarts with local search.
Instead of pure DFS backtracking, we use a randomized local-search approach:
1. Build a greedy baseline.
2. Use simulated annealing to try to extend the sequence by swapping/removing
   elements and inserting new ones, with a cooling schedule.
3. Multiple restarts with different random seeds to explore different regions.
"""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def all_block_sums(vals):
        """Compute all consecutive block sums for a strictly increasing sequence."""
        s = set()
        n = len(vals)
        for i in range(n):
            t = 0
            for j in range(i, n):
                t += vals[j]
                s.add(t)
        return s

    def valid(seq):
        """Check if sequence is valid: strictly increasing, in [1,256], distinct block sums."""
        if not seq:
            return True
        if any(v < 1 or v > 256 for v in seq):
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        bs = all_block_sums(seq)
        expected = len(seq) * (len(seq) + 1) // 2
        return len(bs) == expected

    def greedy():
        """Greedy construction: pick smallest valid element at each step."""
        values, sums = [], set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def greedy_with_start(start_val=1):
        """Greedy construction starting from a given value."""
        values, sums = [], set()
        for x in range(start_val, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def try_insert(values, sums, x):
        """Try to insert x into the sequence at position pos.
        Returns (new_values, new_sums) or None."""
        # Find insertion position (must maintain strictly increasing)
        pos = None
        for i, v in enumerate(values):
            if x < v:
                pos = i
                break
        if pos is None:
            pos = len(values)
        if pos > 0 and values[pos - 1] >= x:
            return None
        if pos < len(values) and values[pos] <= x:
            return None

        # Compute new block sums that would be created
        # New sums involving x:
        # - x itself
        # - x + values[pos+1], x + values[pos+1] + values[pos+2], ...
        # - values[pos-1] + x, values[pos-2] + values[pos-1] + x, ...
        # - values[pos-k] + ... + values[pos-1] + x + values[pos] + ... + values[pos+m]
        # This is complex. Let's just compute all block sums for the new sequence.
        new_values = values[:pos] + [x] + values[pos:]
        new_sums = all_block_sums(new_values)
        expected = len(new_values) * (len(new_values) + 1) // 2
        if len(new_sums) != expected:
            return None
        return new_values, new_sums

    def local_search(initial_seq, time_limit, rng):
        """Simulated annealing style local search to extend the sequence."""
        if not valid(initial_seq):
            # Fix: build greedy from scratch
            initial_seq = greedy()
        
        best_seq = list(initial_seq)
        best_len = len(best_seq)
        best_sums = all_block_sums(best_seq)
        
        current_seq = list(initial_seq)
        current_sums = all_block_sums(current_seq)
        
        # Temperature schedule
        T = 100.0
        T_min = 1.0
        cooling = 0.995
        
        while time.time() < time_limit:
            T = max(T_min, T * cooling)
            
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to insert a random unused value
                used = set(current_seq)
                available = [x for x in range(1, 257) if x not in used]
                if not available:
                    continue
                x = rng.choice(available)
                new_seq, new_sums = try_insert(current_seq, current_sums, x)
                if new_seq is not None:
                    # Always accept if it extends
                    if len(new_seq) > len(current_seq):
                        current_seq = new_seq
                        current_sums = new_sums
                        if len(current_seq) > best_len:
                            best_seq = list(current_seq)
                            best_len = len(best_seq)
                            best_sums = all_block_sums(best_seq)
                    # Otherwise accept with probability based on temperature
                    elif rng.random() < 0.1:
                        current_seq = new_seq
                        current_sums = new_sums
            elif move_type < 0.7:
                # Try to swap two elements (if it maintains order)
                if len(current_seq) < 2:
                    continue
                i, j = rng.sample(range(len(current_seq)), 2)
                if i > j:
                    i, j = j, i
                # Swap values[i] and values[j]
                new_seq = list(current_seq)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                # Check if still strictly increasing
                if all(new_seq[k] < new_seq[k+1] for k in range(len(new_seq)-1)):
                    new_sums = all_block_sums(new_seq)
                    expected = len(new_seq) * (len(new_seq) + 1) // 2
                    if len(new_sums) == expected:
                        # Accept swap (neutral move)
                        current_seq = new_seq
                        current_sums = new_sums
            elif move_type < 0.85:
                # Try to remove a random element and reinsert it elsewhere
                if len(current_seq) < 2:
                    continue
                idx = rng.randrange(len(current_seq))
                x = current_seq[idx]
                new_seq = current_seq[:idx] + current_seq[idx+1:]
                # Try to reinsert x
                result = try_insert(new_seq, all_block_sums(new_seq), x)
                if result is not None:
                    new_seq, new_sums = result
                    current_seq = new_seq
                    current_sums = new_sums
            else:
                # Try to replace the last element with a larger one
                if not current_seq:
                    continue
                last = current_seq[-1]
                if last < 256:
                    new_last = rng.randint(last + 1, 256)
                    new_seq = current_seq[:-1] + [new_last]
                    new_sums = all_block_sums(new_seq)
                    expected = len(new_seq) * (len(new_seq) + 1) // 2
                    if len(new_sums) == expected:
                        current_seq = new_seq
                        current_sums = new_sums
        
        return best_seq

    def targeted_backtrack(prefix, target_len, time_limit, rng):
        """Backtrack from prefix to reach target_len, using randomization."""
        vals = list(prefix)
        sums = all_block_sums(vals) if vals else set()
        lo = vals[-1] + 1 if vals else 1
        
        def bt(vals, sums, lo, tlimit, depth=0):
            if time.time() > tlimit:
                return None
            if len(vals) == target_len:
                return list(vals)
            
            # Pruning: not enough values left
            if 256 - lo + 1 < target_len - len(vals):
                return None
            
            # Try candidates in a slightly randomized order for diversity
            candidates = list(range(lo, 257))
            if depth > 2 and rng.random() < 0.3:
                # Shuffle a portion
                rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > tlimit:
                    return None
                # Check if x can be added
                total = x
                ok = True
                for a in reversed(vals):
                    total += a
                    if total in sums:
                        ok = False
                        break
                if not ok:
                    continue
                # Also check x itself
                if x in sums:
                    continue
                
                vals.append(x)
                # Compute new sums
                new_sums = set()
                t = x
                new_sums.add(t)
                for a in reversed(vals[:-1]):
                    t += a
                    new_sums.add(t)
                sums2 = sums | new_sums
                
                res = bt(vals, sums2, x + 1, tlimit, depth + 1)
                if res is not None:
                    return res
                vals.pop()
            return None
        
        return bt(vals, sums, lo, time_limit)

    # Main algorithm
    best = greedy()
    best_len = len(best)
    
    # Try different random seeds for local search
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        time_per_run = min(3.0, (deadline - 5 - time.time()) / max(1, 50 - seed))
        if time_per_run < 0.1:
            break
        
        # Use a slightly perturbed starting point
        start_seq = list(best)
        # Randomly remove a few elements to create diversity
        if len(start_seq) > 3 and seed > 0:
            n_remove = min(rng.randint(0, 3), len(start_seq) - 2)
            for _ in range(n_remove):
                if start_seq:
                    idx = rng.randrange(len(start_seq))
                    start_seq.pop(idx)
        
        result = local_search(start_seq, time.time() + time_per_run, rng)
        if valid(result) and len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Try targeted backtracking to extend beyond current best
    for target_len in range(best_len + 1, best_len + 15):
        if time.time() > deadline - 3:
            break
        
        remaining = deadline - 3 - time.time()
        if remaining < 0.5:
            break
        
        # Try with different prefix lengths
        for plen in range(min(best_len, target_len - 1), max(0, target_len - 10) - 1, -1):
            if time.time() > deadline - 3:
                break
            prefix = best[:plen]
            tlimit = min(deadline - 3, time.time() + remaining * 0.3)
            if time.time() >= tlimit:
                continue
            
            rng = random.Random(target_len * 1000 + plen)
            result = targeted_backtrack(prefix, target_len, tlimit, rng)
            if result is not None and valid(result) and len(result) == target_len:
                best = result
                best_len = target_len
                break
        
        if best_len == target_len:
            continue
        else:
            # If we couldn't reach this target, try next
            pass
    
    # Final validation
    if not valid(best):
        best = greedy()
    
    return best
# EVOLVE-BLOCK-END
