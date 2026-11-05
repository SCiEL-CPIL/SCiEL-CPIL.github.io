import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe (90s kill, 60s eval limit)
    
    # Helper: Check if sequence has distinct consecutive block sums
    # Optimized for speed: use local variables, break early
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        sums = set()
        for i in range(n):
            s = 0
            # Unroll inner loop slightly? No, Python is slow.
            # Just keep it tight.
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper: Greedy fill a sequence with candidates in a specific order
    # Returns the extended sequence if valid additions were made
    def greedy_fill(seq, order, time_limit):
        current = seq[:]
        current_set = set(current)
        
        # We iterate through the order. If an element is not in current, try to add it.
        # To be efficient, we check if adding it keeps the sequence valid.
        # Since the sequence must be strictly increasing, we just append and sort.
        # However, sorting every time is O(N log N). 
        # We can maintain a sorted list and use bisect for insertion, 
        # but checking validity requires the whole list anyway.
        
        for x in order:
            if time.time() > time_limit:
                break
            if x in current_set:
                continue
            
            # Try adding x
            new_seq = current + [x]
            new_seq.sort()
            
            if is_valid(new_seq):
                current = new_seq
                current_set.add(x)
                
        return current

    # Initial Baselines
    best_seq = []
    
    # 1. Standard Ascending Greedy
    seq_asc = greedy_fill([], list(range(1, 257)), deadline)
    if len(seq_asc) > len(best_seq):
        best_seq = seq_asc
        
    # 2. Descending Greedy (Picks largest first)
    # This often works better for distinct sums because large numbers
    # create large, distinct sums quickly.
    seq_desc = greedy_fill([], list(range(256, 0, -1)), deadline)
    if len(seq_desc) > len(best_seq):
        best_seq = seq_desc
        
    # 3. "Spaced" Greedy: Start with powers of 2 or similar?
    # Let's try a greedy that picks the largest available valid number at each step.
    # This is similar to descending but more dynamic.
    def greedy_largest_first():
        selected = []
        available = set(range(1, 257))
        while available:
            # Find the largest x in available such that selected + [x] is valid
            # We can iterate from largest to smallest
            found = False
            for x in sorted(available, reverse=True):
                candidate = selected + [x]
                candidate.sort()
                if is_valid(candidate):
                    selected = candidate
                    available.remove(x)
                    found = True
                    break
            if not found:
                break
        return selected

    seq_largest = greedy_largest_first()
    if len(seq_largest) > len(best_seq):
        best_seq = seq_largest

    # 4. Random Restarts with Greedy Fill
    rng = random.Random(42)
    
    # Try a few random orders
    for _ in range(5):
        order = list(range(1, 257))
        rng.shuffle(order)
        seq_rand = greedy_fill([], order, deadline)
        if len(seq_rand) > len(best_seq):
            best_seq = seq_rand

    best_len = len(best_seq)
    all_candidates = set(range(1, 257))
    
    # Local Search / Randomized Improvement
    # We will use the best_seq as a starting point for local search.
    
    current_seq = best_seq[:]
    current_set = set(current_seq)
    
    stall_counter = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Determine move type
        r = rng.random()
        
        if r < 0.6:
            # Try to ADD a random unused element
            available = list(all_candidates - current_set)
            if not available:
                break # No more elements to add
            
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
                
        elif r < 0.8:
            # Try to REMOVE a random element (to escape local maxima)
            if current_seq:
                x = rng.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
                stall_counter += 1
                
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
        
        # Periodic Greedy Polish
        # If stalled, try to fill gaps with a greedy sweep
        if stall_counter > 20 or (time.time() - last_improvement_time > 1.0 and stall_counter > 5):
            # Try multiple orders for polishing
            polish_limit = min(deadline, time.time() + 2.0)
            
            # Order 1: Ascending
            test_seq = greedy_fill(current_seq, sorted(all_candidates - current_set), polish_limit)
            if len(test_seq) > len(current_seq):
                current_seq = test_seq
                current_set = set(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                stall_counter = 0
            
            # Order 2: Descending
            if time.time() < deadline:
                test_seq = greedy_fill(current_seq, sorted(all_candidates - current_set, reverse=True), polish_limit)
                if len(test_seq) > len(current_seq):
                    current_seq = test_seq
                    current_set = set(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                        last_improvement_time = time.time()
                    stall_counter = 0
            
            # Order 3: Random
            if time.time() < deadline:
                rem = list(all_candidates - current_set)
                rng.shuffle(rem)
                test_seq = greedy_fill(current_seq, rem, polish_limit)
                if len(test_seq) > len(current_seq):
                    current_seq = test_seq
                    current_set = set(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = current_seq[:]
                        last_improvement_time = time.time()
                    stall_counter = 0
            
            # If still stalled, force a restart or perturbation
            if stall_counter > 100:
                # Perturbation: Remove 3 random elements
                if len(current_seq) > 3:
                    to_remove = rng.sample(current_seq, 3)
                    for x in to_remove:
                        current_seq.remove(x)
                        current_set.remove(x)
                stall_counter = 0

    return best_seq

if __name__ == "__main__":
    res = construct()
    # print(len(res))
    pass
