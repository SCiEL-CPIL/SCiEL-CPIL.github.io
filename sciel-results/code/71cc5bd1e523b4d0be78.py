import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin before the 90s kill
    
    def greedy_construct():
        """Standard ascending greedy: pick smallest available x that doesn't create duplicate block sums."""
        values = []
        sums = set()
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

    def greedy_with_random_skip():
        """Greedy that occasionally skips a value to explore different paths."""
        values = []
        sums = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        for x in candidates:
            if x <= values[-1] if values else False:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        # Ensure strictly increasing
        values.sort()
        return values

    def greedy_reverse_order():
        """Try values in a different order - start from larger values first, then fill in."""
        # This is tricky because we need strictly increasing.
        # Instead, let's try a greedy that considers all candidates at each step.
        values = []
        sums = set()
        
        for pos in range(30):  # Try to build up to 30 elements
            best_x = None
            for x in range((values[-1] + 1) if values else 1, 257):
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    best_x = x
                    break
            if best_x is None:
                break
            values.append(best_x)
            tails = [best_x]
            total = best_x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
        return values

    # Run baseline
    baseline_seq = greedy_construct()
    best_seq = baseline_seq
    best_len = len(best_seq)
    
    # Try multiple random restarts with different strategies
    random.seed(42)
    
    # Strategy 1: Multiple greedy with slight randomization
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        values = []
        sums = set()
        
        for pos in range(35):
            if time.time() - start_time > time_limit:
                break
            
            # Get all valid candidates
            valid_candidates = []
            start = (values[-1] + 1) if values else 1
            for x in range(start, 257):
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Choose a candidate - sometimes smallest, sometimes random
            if attempt % 3 == 0:
                # Smallest
                x = valid_candidates[0]
            elif attempt % 3 == 1:
                # Random from first half of valid candidates
                x = random.choice(valid_candidates[:max(1, len(valid_candidates)//2)])
            else:
                # Random from all valid candidates
                x = random.choice(valid_candidates)
            
            values.append(x)
            tails = [x]
            total = x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
        
        if len(values) > best_len:
            best_len = len(values)
            best_seq = values[:]
    
    # Strategy 2: Local search / improvement
    # Try to extend the best sequence found so far
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        seq = best_seq[:]
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        
        # Try to extend
        extended = False
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            last_val = seq[-1] if seq else 0
            valid_candidates = []
            for x in range(last_val + 1, 257):
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            x = random.choice(valid_candidates)
            seq.append(x)
            tails = [x]
            total = x
            for a in reversed(seq[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
            extended = True
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    # Strategy 3: Backtracking with pruning for short sequences
    # Try to find a sequence of length best_len + 1, best_len + 2, etc.
    def check_valid(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def dfs_extend(seq, used_sums, suffix_sums, target_len):
        """Try to extend seq to reach target_len."""
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if len(seq) == target_len:
            best_seq = seq[:]
            best_len = target_len
            return True
            
        last_val = seq[-1] if seq else 0
        
        # Try candidates in order of increasing value
        for x in range(last_val + 1, 257):
            # Calculate new suffix sums
            new_suffixes = [x]
            current_sum = x
            
            valid = True
            if x in used_sums:
                valid = False
            
            if valid:
                for s in suffix_sums:
                    current_sum += s
                    new_suffixes.append(current_sum)
                    if current_sum in used_sums:
                        valid = False
                        break
            
            if not valid:
                continue
            
            new_used_sums = used_sums.union(new_suffixes)
            
            if dfs_extend(seq + [x], new_used_sums, new_suffixes, target_len):
                return True
                
        return False

    # Try to find longer sequences via DFS
    for target_len in range(best_len + 1, best_len + 5):
        if time.time() - start_time > time_limit:
            break
        
        # Start DFS from empty
        found = dfs_extend([], set(), [], target_len)
        
        if found:
            continue
        else:
            break

    return best_seq
