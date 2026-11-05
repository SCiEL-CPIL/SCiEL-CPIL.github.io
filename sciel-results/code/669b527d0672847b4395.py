import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        L = len(seq)
        # Prefix sums
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

    def compute_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums

    def greedy_increasing():
        """Standard greedy: add smallest possible next."""
        values = []
        sums = set()
        for x in range(1, 257):
            potential_sums = set()
            potential_sums.add(x)
            
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            for s in suffix_sums:
                potential_sums.add(x + s)
            
            if not sums.intersection(potential_sums):
                values.append(x)
                sums.update(potential_sums)
        
        return values

    def greedy_decreasing():
        """Greedy: pick largest available, then next largest."""
        values = []
        sums = set()
        for x in range(256, 0, -1):
            potential_sums = set()
            potential_sums.add(x)
            
            prefix_sums = []
            current_sum = 0
            for v in values:
                current_sum += v
                prefix_sums.append(current_sum)
            
            for s in prefix_sums:
                potential_sums.add(x + s)
            
            if not sums.intersection(potential_sums):
                values.append(x)
                values.sort()
                sums.update(potential_sums)
        
        return values

    def greedy_random(seed_val):
        """Randomized greedy with given seed."""
        rng = random.Random(seed_val)
        values = []
        sums = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        for x in candidates:
            potential_sums = set()
            potential_sums.add(x)
            
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            for s in suffix_sums:
                potential_sums.add(x + s)
            
            if not sums.intersection(potential_sums):
                values.append(x)
                sums.update(potential_sums)
        
        return sorted(values)

    # 1. Get baseline from multiple greedy strategies
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)
    
    # Try several randomized greedy variants
    for seed in range(10):
        rand_seq = greedy_random(seed)
        if len(rand_seq) > best_len:
            best_seq = rand_seq
            best_len = len(rand_seq)

    # 2. Local Search with improved strategies
    current_seq = list(best_seq)
    current_sums = compute_sums(current_seq)
    
    all_candidates = list(range(1, 257))
    
    def try_add(x, seq, sums):
        """Try adding x to seq, return new sums if valid, else None."""
        L = len(seq)
        # New sums are: x, and x + suffix_sum for each suffix of seq
        # Suffix sums: sum(seq[-1:]), sum(seq[-2:]), ..., sum(seq)
        new_sums = {x}
        suffix_sum = 0
        for i in range(L - 1, -1, -1):
            suffix_sum += seq[i]
            new_sums.add(x + suffix_sum)
        
        if sums & new_sums:
            return None
        return sums | new_sums

    def try_remove(x, seq, sums):
        """Remove x from seq, return new sums."""
        new_seq = [v for v in seq if v != x]
        return compute_sums(new_seq)

    def try_swap(x_rem, x_add, seq, sums):
        """Swap x_rem with x_add, return new sums if valid, else None."""
        new_seq = [v for v in seq if v != x_rem]
        new_seq.append(x_add)
        new_seq.sort()
        
        # Check validity
        L = len(new_seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + new_seq[i]
        
        new_sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in new_sums:
                    return None
                new_sums.add(s)
        return new_sums

    while time.time() < deadline:
        r = random.random()
        
        if r < 0.5:  # Add - highest priority for length increase
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if available:
                # Try multiple candidates
                random.shuffle(available)
                for x in available[:20]:
                    new_sums = try_add(x, current_seq, current_sums)
                    if new_sums is not None:
                        current_seq.append(x)
                        current_seq.sort()
                        current_sums = new_sums
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
                        break
        
        elif r < 0.75:  # Remove
            if current_seq:
                x = random.choice(current_seq)
                new_sums = try_remove(x, current_seq, current_sums)
                current_seq.remove(x)
                current_sums = new_sums
        
        else:  # Swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_sums = try_swap(x_rem, x_add, current_seq, current_sums)
                    if new_sums is not None:
                        current_seq.remove(x_rem)
                        current_seq.append(x_add)
                        current_seq.sort()
                        current_sums = new_sums
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)

    # Final polish: try to add more elements
    current_set = set(best_seq)
    available = [x for x in all_candidates if x not in current_set]
    random.shuffle(available)
    for x in available:
        if time.time() > deadline - 1:
            break
        new_sums = try_add(x, best_seq, compute_sums(best_seq))
        if new_sums is not None:
            best_seq.append(x)
            best_seq.sort()
            best_len = len(best_seq)

    return sorted(best_seq)
