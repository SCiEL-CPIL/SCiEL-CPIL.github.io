import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
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

    def greedy_asc():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals

    def greedy_desc():
        vals = []
        sums_set = set()
        for x in range(256, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals

    # Initialize with best of greedy
    seq_asc = greedy_asc()
    seq_desc = greedy_desc()
    
    best_seq = seq_asc if len(seq_asc) >= len(seq_desc) else seq_desc
    best_len = len(best_seq)
    
    # Local Search Function
    def local_search(initial_seq, seed):
        nonlocal best_seq, best_len, start_time, time_limit
        random.seed(seed)
        seq = initial_seq.copy()
        current_len = len(seq)
        
        # Track used values
        used_vals = set(seq)
        available = [x for x in range(1, 257) if x not in used_vals]
        
        # To speed up, we can pre-compute sums, but re-computing is O(N^2) which is fine for N~20
        # We will try to insert elements.
        
        stuck_count = 0
        max_stuck = 100
        
        while time.time() - start_time < time_limit:
            if stuck_count > max_stuck:
                # Kick: remove a random element to create space
                if len(seq) > 1:
                    idx = random.randint(0, len(seq) - 1)
                    removed = seq.pop(idx)
                    used_vals.remove(removed)
                    available.append(removed)
                    available.sort()
                    stuck_count = 0
                else:
                    # If we are down to 1 element, restart this local search instance
                    return seq
            
            # Try to insert a random available element
            # Bias towards larger elements? Or random?
            # Larger elements tend to have larger sums, reducing collision with existing small sums.
            # But we need to maintain strictly increasing order.
            
            # Strategy: Pick a random element from available, and a random position to insert.
            # To be efficient, we can pick a random position and then find an available element
            # that fits in value order? No, we just pick an element and find its insertion point.
            
            if not available:
                return seq
            
            # Pick a random element to try
            x = random.choice(available)
            
            # Find insertion position to maintain sorted order
            # Since seq is sorted, we can use bisect
            import bisect
            pos = bisect.bisect_left(seq, x)
            
            # Create new sequence
            new_seq = seq[:pos] + [x] + seq[pos:]
            
            if is_valid(new_seq):
                # Accept move
                seq = new_seq
                used_vals.add(x)
                available.remove(x)
                current_len = len(seq)
                stuck_count = 0
                
                # Update global best
                if current_len > best_len:
                    best_len = current_len
                    best_seq = seq.copy()
            else:
                stuck_count += 1
                
            # Periodic check for time
            if time.time() - start_time > time_limit:
                break
                
        return seq

    # Multi-start Local Search
    # Start with the greedy results and some random starts
    starts = [seq_asc, seq_desc]
    
    # Add a few random starts
    for _ in range(3):
        # Randomized greedy: at each step, pick a random valid candidate
        seq = []
        sums_set = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # This naive random greedy might be very short. 
        # Better: Start with a small random seed and try to extend.
        # Let's just use the greedy results as strong bases.
        pass

    # Run local search on the best greedy sequences with different seeds
    for i in range(20): # Max 20 restarts, but time limited
        if time.time() - start_time > time_limit:
            break
        
        # Alternate between ascending and descending greedy bases, or just use the best one
        base = seq_asc if i % 2 == 0 else seq_desc
        # Add some noise to the base? 
        # Maybe just run local search on the base.
        # To diversify, we can randomly remove a few elements from the base first
        perturbed_base = base.copy()
        if len(perturbed_base) > 5:
            # Remove 1-2 random elements
            for _ in range(random.randint(1, 2)):
                if perturbed_base:
                    idx = random.randint(0, len(perturbed_base) - 1)
                    perturbed_base.pop(idx)
        
        local_search(perturbed_base, seed=i)
        
        # Also try a pure random start occasionally
        if i % 5 == 0:
            random_start = []
            # Just pick 5 random numbers and sort them, then try to extend
            random_start = sorted(random.sample(range(1, 257), 5))
            if is_valid(random_start):
                local_search(random_start, seed=100+i)
            else:
                # Try to fix it by removing one
                for i_rm in range(5):
                    trial = random_start[:i_rm] + random_start[i_rm+1:]
                    if is_valid(trial):
                        local_search(trial, seed=200+i)
                        break

    return best_seq
