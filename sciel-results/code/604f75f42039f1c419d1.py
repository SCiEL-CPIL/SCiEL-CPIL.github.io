import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def compute_sums(seq):
        """Compute all consecutive block sums for a sorted sequence."""
        L = len(seq)
        if L == 0:
            return set()
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums
    
    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        L = len(seq)
        if L == 0:
            return True
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def incremental_check(seq, x, existing_sums):
        """
        Check if x can be added to sorted seq without creating duplicate sums.
        Returns True if valid, False otherwise.
        Also returns the set of new sums created by adding x.
        """
        k = bisect.bisect_left(seq, x)
        L = len(seq)
        
        new_sums = set()
        
        # x alone
        if x in existing_sums:
            return False, None
        new_sums.add(x)
        
        # Left extensions: blocks ending at x
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            s = left_sum + x
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Right extensions: blocks starting at x
        right_sum = 0
        for j in range(k, L):
            right_sum += seq[j]
            s = x + right_sum
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Both sides: blocks containing x with elements on both sides
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            right_sum = 0
            for j in range(k, L):
                right_sum += seq[j]
                s = left_sum + x + right_sum
                if s in existing_sums or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums
    
    def greedy_build(strategy, existing_sums=None):
        """
        Build a sequence greedily using the given strategy.
        strategy: 'increasing', 'decreasing', 'random', or a list of values to try in order.
        """
        seq = []
        sums = set() if existing_sums is None else existing_sums.copy()
        
        if strategy == 'increasing':
            candidates = list(range(1, 257))
        elif strategy == 'decreasing':
            candidates = list(range(256, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            candidates = strategy
        
        for x in candidates:
            if x in seq:
                continue
            valid, new_sums = incremental_check(seq, x, sums)
            if valid:
                k = bisect.bisect_left(seq, x)
                seq = seq[:k] + [x] + seq[k:]
                sums.update(new_sums)
        
        return seq, sums
    
    # Try multiple greedy strategies
    best_seq = []
    best_sums = set()
    
    strategies = [
        'increasing',
        'decreasing',
    ]
    
    # Add some randomized strategies
    for _ in range(5):
        rand_strat = list(range(1, 257))
        random.shuffle(rand_strat)
        strategies.append(rand_strat)
    
    for strat in strategies:
        seq, sums = greedy_build(strat)
        if len(seq) > len(best_seq):
            best_seq = seq
            best_sums = sums
    
    # Local search with informed moves
    current_seq = list(best_seq)
    current_sums = set(best_sums)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)
    
    all_values = set(range(1, 257))
    
    def try_add(x):
        """Try to add x to current_seq. Returns (success, new_seq, new_sums)."""
        valid, new_sums = incremental_check(current_seq, x, current_sums)
        if valid:
            k = bisect.bisect_left(current_seq, x)
            new_seq = current_seq[:k] + [x] + current_seq[k:]
            new_sums_set = current_sums | new_sums
            return True, new_seq, new_sums_set
        return False, None, None
    
    def try_swap(x_remove, x_add):
        """Try to swap x_remove with x_add. Returns (success, new_seq, new_sums)."""
        # Remove x_remove, then try to add x_add
        if x_remove not in current_seq:
            return False, None, None
        
        # Build sequence without x_remove
        seq_without = [x for x in current_seq if x != x_remove]
        sums_without = compute_sums(seq_without)
        
        # Now try to add x_add to seq_without
        valid, new_sums = incremental_check(seq_without, x_add, sums_without)
        if valid:
            k = bisect.bisect_left(seq_without, x_add)
            new_seq = seq_without[:k] + [x_add] + seq_without[k:]
            new_sums_set = sums_without | new_sums
            return True, new_seq, new_sums_set
        return False, None, None
    
    # Informed local search
    stuck_count = 0
    max_stuck = 100
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.6:
            # Try to add a random available value
            current_set = set(current_seq)
            available = [x for x in all_values if x not in current_set]
            if available:
                # Sort by some heuristic: prefer values that create fewer conflicts
                # For now, just pick randomly
                x = random.choice(available)
                success, new_seq, new_sums = try_add(x)
                if success:
                    current_seq = new_seq
                    current_sums = new_sums
                    stuck_count = 0
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq_final = list(current_seq)
                else:
                    stuck_count += 1
                    
        elif r < 0.85:
            # Try to swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_values if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    success, new_seq, new_sums = try_swap(x_rem, x_add)
                    if success:
                        current_seq = new_seq
                        current_sums = new_sums
                        stuck_count = 0
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq_final = list(current_seq)
                    else:
                        stuck_count += 1
        else:
            # Try to remove a random element to escape local minima
            if current_seq and len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                current_sums = compute_sums(current_seq)
                stuck_count = 0
        
        # If stuck for too long, restart with a new random greedy
        if stuck_count > max_stuck:
            rand_strat = list(range(1, 257))
            random.shuffle(rand_strat)
            seq, sums = greedy_build(rand_strat)
            if len(seq) > len(current_seq):
                current_seq = seq
                current_sums = sums
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_final = list(current_seq)
            stuck_count = 0
    
    return sorted(best_seq_final)
