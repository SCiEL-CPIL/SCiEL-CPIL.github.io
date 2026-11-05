import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe (90s kill, 60s eval limit)
    
    # Helper: Check if sequence has distinct consecutive block sums
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper: Greedy polish with multiple orderings
    def greedy_polish(seq, rng, time_limit):
        """Try greedy addition in multiple orderings, return best result."""
        best = seq[:]
        best_len = len(best)
        all_cands = set(range(1, 257))
        
        # Try ascending order
        test_seq = seq[:]
        test_set = set(test_seq)
        for x in sorted(all_cands - test_set):
            if time.time() > time_limit:
                break
            candidate = test_seq + [x]
            candidate.sort()
            if is_valid(candidate):
                test_seq = candidate
                test_set.add(x)
        if len(test_seq) > best_len:
            best = test_seq
            best_len = len(test_seq)
        
        # Try descending order
        test_seq = seq[:]
        test_set = set(test_seq)
        for x in sorted(all_cands - test_set, reverse=True):
            if time.time() > time_limit:
                break
            candidate = test_seq + [x]
            candidate.sort()
            if is_valid(candidate):
                test_seq = candidate
                test_set.add(x)
        if len(test_seq) > best_len:
            best = test_seq
            best_len = len(test_seq)
        
        # Try random order
        test_seq = seq[:]
        test_set = set(test_seq)
        remaining = list(all_cands - test_set)
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() > time_limit:
                break
            candidate = test_seq + [x]
            candidate.sort()
            if is_valid(candidate):
                test_seq = candidate
                test_set.add(x)
        if len(test_seq) > best_len:
            best = test_seq
            best_len = len(test_seq)
        
        return best

    # Helper: Greedy build from a specific order
    def greedy_build(order):
        selected = []
        for x in order:
            # Optimization: Check if x can be added
            # Since we maintain sorted order, x is appended at the end? 
            # No, 'order' is arbitrary. We must insert and sort.
            # For efficiency in this specific helper, we assume we just try to add.
            # If 'order' is sorted, it's faster.
            
            # Quick check: if x is already in selected, skip
            if x in selected:
                continue
                
            candidate = selected + [x]
            candidate.sort()
            if is_valid(candidate):
                selected = candidate
        return selected

    # Initial Baselines
    best_seq = []
    
    # 1. Standard Ascending Greedy
    seq_asc = greedy_build(list(range(1, 257)))
    if len(seq_asc) > len(best_seq):
        best_seq = seq_asc
        
    # 2. Descending Greedy (Picks largest first, inserts at front)
    # This is equivalent to greedy_build with descending order, 
    # but the insertion logic in greedy_build handles sorting.
    seq_desc = greedy_build(list(range(256, 0, -1)))
    if len(seq_desc) > len(best_seq):
        best_seq = seq_desc
        
    best_len = len(best_seq)
    all_candidates = set(range(1, 257))
    
    # Local Search / Randomized Improvement
    # We will use the best_seq as a starting point for local search.
    # We also do random restarts with shuffled greedy builds.
    
    rng = random.Random(42)
    
    # Set of currently used numbers in best_seq for quick lookup
    current_seq = best_seq[:]
    current_set = set(current_seq)
    
    # Track stall
    stall_counter = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Determine move type
        # Bias towards adding if we are not at max possible length
        # Max possible length is bounded by 256, but practically much smaller.
        
        r = rng.random()
        
        if r < 0.7:
            # Try to ADD a random unused element
            available = list(all_candidates - current_set)
            if not available:
                break # No more elements to add
            
            # Shuffle to pick random, or pick from end? 
            # Random is better for exploration.
            x = rng.choice(available)
            
            new_seq = current_seq + [x]
            new_seq.sort()
            
            if is_valid(new_seq):
                current_seq = new_seq
                current_set.add(x)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1
                
        elif r < 0.85:
            # Try to REMOVE a random element (to escape local maxima)
            if current_seq:
                x = rng.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
                # We don't update best_seq on removal
                stall_counter += 1
                # After removal, immediately try greedy polish to fill back in
                if stall_counter > 20:
                    polish_time_limit = min(deadline, time.time() + 1.0)
                    test_seq = greedy_polish(current_seq, rng, polish_time_limit)
                    if len(test_seq) > len(current_seq):
                        current_seq = test_seq
                        current_set = set(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = current_seq[:]
                            last_improvement_time = time.time()
                        stall_counter = 0
                
        else:
            # Try to SWAP: Remove one, try to add a different one
            if current_seq:
                x_to_remove = rng.choice(current_seq)
                available = list(all_candidates - current_set)
                if available:
                    x_to_add = rng.choice(available)
                    if x_to_add != x_to_remove:
                        new_seq = [v for v in current_seq if v != x_to_remove]
                        new_seq.append(x_to_add)
                        new_seq.sort()
                        if is_valid(new_seq):
                            current_seq = new_seq
                            current_set.remove(x_to_remove)
                            current_set.add(x_to_add)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = current_seq[:]
                                last_improvement_time = time.time()
                                stall_counter = 0
                            else:
                                stall_counter += 1
                        else:
                            stall_counter += 1
                # After swap attempt (whether successful or not), if stalled, try polish
                if stall_counter > 30:
                    polish_time_limit = min(deadline, time.time() + 1.0)
                    test_seq = greedy_polish(current_seq, rng, polish_time_limit)
                    if len(test_seq) > len(current_seq):
                        current_seq = test_seq
                        current_set = set(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = current_seq[:]
                            last_improvement_time = time.time()
                        stall_counter = 0
        # End Move Selection
        
        # Periodic Greedy Polish
        # Every 50 moves or if stalled, try a greedy sweep with multiple orderings
        if stall_counter > 50 or (time.time() - last_improvement_time > 2.0 and stall_counter > 10):
            # Use multi-ordering greedy polish
            polish_time_limit = min(deadline, time.time() + 3.0)
            test_seq = greedy_polish(current_seq, rng, polish_time_limit)
            
            if len(test_seq) > len(current_seq):
                current_seq = test_seq
                current_set = set(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                stall_counter = 0
            else:
                # If greedy polish didn't help, maybe restart from a random baseline
                # to diversify
                if stall_counter > 200:
                    # Random Restart
                    order = list(range(1, 257))
                    rng.shuffle(order)
                    restart_seq = greedy_build(order)
                    
                    # Also try a "smart" restart: take best_seq, remove a few, shuffle rest
                    # For now, simple shuffle restart
                    if len(restart_seq) > len(current_seq):
                        current_seq = restart_seq
                        current_set = set(current_seq)
                    else:
                        # If restart is worse, keep current but force a big perturbation
                        # Remove 3 random elements
                        if len(current_seq) > 3:
                            to_remove = rng.sample(current_seq, 3)
                            for x in to_remove:
                                current_seq.remove(x)
                                current_set.remove(x)
                    stall_counter = 0
                    
    return best_seq

if __name__ == "__main__":
    # Simple test
    res = construct()
    # print(res)
    # print(len(res))
    pass
