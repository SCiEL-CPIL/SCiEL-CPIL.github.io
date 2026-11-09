import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    random.seed(42)

    # Start with a known good greedy solution
    def greedy_construct():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    best_seq, best_sums = greedy_construct()
    best_len = len(best_seq)

    if time.time() > deadline:
        return best_seq

    # Backtracking search to improve
    # We try to find longer sequences by backtracking
    # State: (prefix, used_sums, next_min)
    # We use iterative deepening: try to find a sequence of length L
    
    def get_consecutive_sums(prefix, new_val):
        """Get all consecutive block sums for prefix + [new_val]"""
        tails = [new_val]
        total = new_val
        for a in reversed(prefix):
            total += a
            tails.append(total)
        return tails

    def backtrack(prefix, used_sums, target_len, max_val):
        """Try to extend prefix to target_len using values up to max_val"""
        if len(prefix) == target_len:
            return prefix[:]
        
        if time.time() > deadline:
            return None
        
        last = prefix[-1] if prefix else 0
        # Try candidate values from last+1 to max_val
        # Heuristic: try values that are "spread out" to avoid sum collisions
        # Try a few candidates per recursion level
        
        candidates = []
        for v in range(last + 1, max_val + 1):
            candidates.append(v)
        
        # Shuffle a subset of candidates to diversify
        if len(candidates) > 50:
            sample_size = min(50, len(candidates))
            candidates = random.sample(candidates, sample_size)
            # Always include the smallest few
            candidates = list(set(candidates + list(range(last+1, min(last+10, max_val+1)))))
            candidates.sort()
        
        for v in candidates:
            tails = get_consecutive_sums(prefix, v)
            if not used_sums.intersection(tails):
                new_prefix = prefix + [v]
                new_used = used_sums.union(tails)
                result = backtrack(new_prefix, new_used, target_len, max_val)
                if result is not None:
                    return result
                if time.time() > deadline:
                    return None
        return None

    # Try to improve with backtracking
    # Start from current best length and try to go higher
    current_max_val = best_seq[-1] if best_seq else 256
    
    for target_len in range(best_len + 1, best_len + 15):
        if time.time() > deadline:
            break
        
        # Try multiple times with different random seeds
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            random.seed(hash((target_len, attempt)) % 100000)
            result = backtrack([], set(), target_len, 256)
            if result is not None and len(result) > best_len:
                best_seq = result
                best_len = len(result)
                break

    if time.time() > deadline:
        return best_seq

    # Try to further optimize: replace some elements to allow longer sequences
    # Local search: try replacing a random element with a larger one to free up sum space
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        trial = best_seq[:]
        # Pick a random position to try replacing
        if len(trial) < 2:
            break
        pos = random.randint(0, len(trial) - 1)
        old_val = trial[pos]
        
        # Try replacing with a value that doesn't cause sum collisions
        # Remove old sums, add new sums
        # This is complex, so let's just try a simple perturbation
        
        # Try swapping with a nearby value
        if pos > 0:
            neighbor = trial[pos-1]
            if old_val - neighbor > 3:
                # Try a smaller value
                for new_val in range(neighbor + 1, old_val):
                    trial[pos] = new_val
                    # Check validity
                    valid = True
                    all_sums = set()
                    for i in range(len(trial)):
                        s = 0
                        for j in range(i, len(trial)):
                            s += trial[j]
                            if s in all_sums:
                                valid = False
                                break
                            all_sums.add(s)
                        if not valid:
                            break
                    if valid:
                        # Try to extend
                        last = trial[-1]
                        for v in range(last + 1, 257):
                            tails = get_consecutive_sums(trial, v)
                            if not all_sums.intersection(tails):
                                trial.append(v)
                                break
                        if len(trial) > best_len:
                            best_seq = trial
                            best_len = len(trial)
                        break
                    trial[pos] = old_val

    return best_seq
# EVOLVE-BLOCK-END
