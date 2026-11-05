import time
import random

def _is_valid(seq):
    """
    Check if all non-empty consecutive block sums are distinct.
    seq must be sorted.
    """
    if not seq:
        return True
    
    L = len(seq)
    # Prefix sums P[0]=0, P[i] = sum(seq[0..i-1])
    P = [0] * (L + 1)
    for i in range(L):
        P[i + 1] = P[i] + seq[i]
        
    sums = set()
    # Consecutive block from i to j-1 has sum P[j] - P[i]
    # i ranges from 0 to L-1
    # j ranges from i+1 to L
    for i in range(L):
        for j in range(i + 1, L + 1):
            s = P[j] - P[i]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_init(seed=0):
    """
    Generate an initial sequence using a randomized greedy approach.
    """
    random.seed(seed)
    seq = []
    sums = set()
    
    # Standard greedy: try to add smallest available
    # We iterate through 1..256. 
    # To add diversity, we can shuffle the order of checking or use the buffer method.
    # Let's use a simple greedy: for x in 1..256, if valid, add.
    # But to get diverse starts, we can randomize the "greediness".
    
    # Actually, the current program's `_greedy_randomized` is decent.
    # Let's implement a fast version of "try to add smallest valid".
    
    for x in range(1, 257):
        # Check if x can be appended
        # New sums: x, x+seq[-1], x+seq[-1]+seq[-2], ...
        # We need to check against existing sums.
        
        # Efficient check:
        # Existing sums are in `sums`.
        # New sums to check:
        # x
        # x + seq[-1]
        # x + seq[-1] + seq[-2]
        # ...
        
        new_sums = [x]
        total = x
        for v in reversed(seq):
            total += v
            new_sums.append(total)
            
        valid = True
        for s in new_sums:
            if s in sums:
                valid = False
                break
        
        if valid:
            seq.append(x)
            sums.update(new_sums)
            
    return seq

def _local_search(initial_seq, time_limit):
    """
    Improve the sequence using local search.
    """
    current_seq = list(initial_seq)
    best_seq = list(initial_seq)
    best_len = len(current_seq)
    
    all_candidates = list(range(1, 257))
    
    # Precompute sets for fast lookup
    current_set = set(current_seq)
    
    start_time = time.time()
    
    # Track stagnation
    no_improve_count = 0
    no_change_count = 0
    
    while time.time() - start_time < time_limit:
        # 1. Try to Add
        # Pick a random candidate not in current_seq
        available = [x for x in all_candidates if x not in current_set]
        
        if available:
            # Try a few random adds
            found_add = False
            for _ in range(10): # Try 10 random candidates
                x = random.choice(available)
                new_seq = current_seq + [x]
                new_seq.sort()
                if _is_valid(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    found_add = True
                    break
            if found_add:
                no_change_count = 0
                continue
        
        # 2. Try to Swap
        # Pick a random element to remove and a random element to add
        if current_seq:
            found_swap = False
            for _ in range(10): # Try 10 random swaps
                x_rem = random.choice(current_seq)
                # Pick x_add from available
                if not available:
                    break
                x_add = random.choice(available)
                
                if x_add == x_rem:
                    continue
                    
                # Construct new sequence
                # Remove x_rem
                # Add x_add
                # Since we need sorted, we can just create a new list
                
                # Optimization: only create list if likely valid? No, validity check is cheap.
                new_seq = [x for x in current_seq if x != x_rem]
                new_seq.append(x_add)
                new_seq.sort()
                
                if _is_valid(new_seq):
                    current_seq = new_seq
                    current_set.remove(x_rem)
                    current_set.add(x_add)
                    # Update available list
                    available = [x for x in all_candidates if x not in current_set]
                    found_swap = True
                    no_change_count = 0
                    break
            
            if found_swap:
                continue
                
        # 3. Try to Remove (to escape local minima)
        # Only if we've been stuck for a while
        if no_change_count > 50 and current_seq:
            x_rem = random.choice(current_seq)
            current_seq.remove(x_rem)
            current_set.remove(x_rem)
            available = [x for x in all_candidates if x not in current_set]
            no_change_count = 0
            no_improve_count += 1
            
            # If we removed an element that was in best_seq, we might want to revert?
            # No, we keep current_seq separate. best_seq is only updated on improvement.
            
        else:
            no_change_count += 1
            no_improve_count += 1

        # Periodic reset if stuck too long?
        if no_improve_count > 1000:
            # Restart from best_seq
            current_seq = list(best_seq)
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            no_improve_count = 0
            no_change_count = 0

    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # 1. Generate multiple diverse initial sequences
    initial_seqs = []
    
    # Standard greedy
    seq_greedy = _greedy_init(seed=0)
    initial_seqs.append(seq_greedy)
    
    # Randomized greedy with different seeds
    for seed in range(1, 20):
        # We can modify the greedy to be more random
        # For now, _greedy_init is deterministic given seed? 
        # My _greedy_init above is deterministic (always picks smallest valid).
        # Let's make a randomized version for initialization.
        
        def _greedy_randomized_init(seed_val):
            random.seed(seed_val)
            seq = []
            sums = set()
            candidates_buffer = []
            
            for x in range(1, 257):
                # Check if x can be appended
                new_sums = [x]
                total = x
                for v in reversed(seq):
                    total += v
                    new_sums.append(total)
                
                valid = True
                for s in new_sums:
                    if s in sums:
                        valid = False
                        break
                
                if valid:
                    candidates_buffer.append((x, new_sums))
                    
                    # Randomly decide when to commit
                    # If buffer is large, pick one randomly
                    if len(candidates_buffer) >= 5:
                        idx = random.randint(0, len(candidates_buffer) - 1)
                        x_val, tails = candidates_buffer[idx]
                        seq.append(x_val)
                        sums.update(tails)
                        candidates_buffer = []
                        
            if candidates_buffer:
                idx = random.randint(0, len(candidates_buffer) - 1)
                x_val, tails = candidates_buffer[idx]
                seq.append(x_val)
                sums.update(tails)
                
            return seq

        seq_rand = _greedy_randomized_init(seed)
        initial_seqs.append(seq_rand)
        
    # Update best from inits
    for s in initial_seqs:
        if len(s) > best_len:
            best_len = len(s)
            best_seq = s[:]
            
    # 2. Apply Local Search to the best initial sequences
    # We have ~55s. Let's allocate time to local search.
    # We'll run local search on the top 3 distinct initial sequences.
    
    # Deduplicate initial sequences
    unique_inits = []
    seen = set()
    for s in initial_seqs:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_inits.append(s)
            
    # Sort by length descending
    unique_inits.sort(key=lambda x: len(x), reverse=True)
    
    # Run local search on the best ones
    # We give each search a time slice.
    # If we have 50s, and 5 candidates, 10s each?
    # Or just run on the single best one for the full time?
    # Running on multiple diverse starts is better to avoid global local optima.
    
    time_per_search = 10
    for i, init_seq in enumerate(unique_inits[:5]): # Top 5
        if time.time() > deadline - 5:
            break
            
        time_left_for_this = min(time_per_search, deadline - time.time())
        if time_left_for_this <= 0:
            break
            
        improved_seq = _local_search(init_seq, time_left_for_this)
        
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq[:]
            
    return best_seq
