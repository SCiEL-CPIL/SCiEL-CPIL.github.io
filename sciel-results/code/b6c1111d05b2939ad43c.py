import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    def get_block_sums(seq):
        """Get all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        sums = []
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return len(sums) == len(set(sums))

    def can_extend(prefix, new_val, existing_sums=None):
        """Check if adding new_val to prefix keeps all block sums distinct."""
        if existing_sums is None:
            existing_sums = get_block_sums(prefix)
        
        n = len(prefix)
        
        # New sums involving new_val
        new_sums = [new_val]
        s = new_val
        for i in range(n-1, -1, -1):
            s += prefix[i]
            new_sums.append(s)
        
        # Check no overlap with existing sums
        for ns in new_sums:
            if ns in existing_sums:
                return False
        # Check new_sums are distinct among themselves
        if len(new_sums) != len(set(new_sums)):
            return False
        return True

    # Strategy 1: Try different initial constructions with various growth patterns
    best = []
    best_sums = set()
    
    # Try various growth patterns
    growth_patterns = [
        lambda x: x * 2,
        lambda x: x * 2 + 1,
        lambda x: x * 3 // 2,
        lambda x: x + x // 2 + 1,
        lambda x: x + 1,
        lambda x: x + 2,
        lambda x: x + 3,
    ]
    
    for base in range(1, 15):
        for growth in growth_patterns:
            seq = []
            sums_set = set()
            current = base
            while current <= MAX_VAL:
                if can_extend(seq, current, sums_set):
                    n = len(seq)
                    s = current
                    sums_set.add(s)
                    for i in range(n - 1, -1, -1):
                        s += seq[i]
                        sums_set.add(s)
                    seq.append(current)
                current = growth(current)
                if current <= seq[-1] if seq else False:
                    current = (seq[-1] if seq else 0) + 1
            if len(seq) > len(best):
                best = seq
                best_sums = sums_set
    
    # Strategy 2: Greedy from 1
    seq = []
    sums_set = set()
    for x in range(1, MAX_VAL + 1):
        if can_extend(seq, x, sums_set):
            n = len(seq)
            s = x
            sums_set.add(s)
            for i in range(n - 1, -1, -1):
                s += seq[i]
                sums_set.add(s)
            seq.append(x)
    if len(seq) > len(best):
        best = seq
        best_sums = sums_set

    # Strategy 3: Local search with replacement
    def local_search(seq, sums_set, time_budget):
        """Try to improve the sequence by local modifications."""
        seq = list(seq)
        sums_set = set(sums_set)
        best_seq = list(seq)
        best_sums = set(sums_set)
        best_len = len(seq)
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # Try adding elements at the end
            for x in range(seq[-1] + 1, MAX_VAL + 1) if seq else range(1, MAX_VAL + 1):
                if can_extend(seq, x, sums_set):
                    n = len(seq)
                    s = x
                    sums_set.add(s)
                    for i in range(n - 1, -1, -1):
                        s += seq[i]
                        sums_set.add(s)
                    seq.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing last element and trying alternatives
                if len(seq) > 1:
                    removed = seq.pop()
                    # Rebuild sums_set for safety
                    sums_set = get_block_sums(seq)
                    
                    for x in range(removed + 1, MAX_VAL + 1):
                        if can_extend(seq, x, sums_set):
                            n = len(seq)
                            s = x
                            sums_set.add(s)
                            for i in range(n - 1, -1, -1):
                                s += seq[i]
                                sums_set.add(s)
                            seq.append(x)
                            improved = True
                            break
                    if not improved:
                        seq.append(removed)
                        sums_set = get_block_sums(seq)
            
            if len(seq) > best_len:
                best_seq = list(seq)
                best_sums = set(sums_set)
                best_len = len(seq)
            
            if not improved:
                break
        
        return best_seq, best_sums
    
    best, best_sums = local_search(best, best_sums, TIME_LIMIT - 10)
    
    # Strategy 4: Try more diverse starting points
    import random
    random.seed(42)
    
    for trial in range(100):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Create a random-ish starting sequence
        seq = []
        sums_set = set()
        current = random.randint(1, 20)
        for _ in range(25):
            if current > MAX_VAL:
                break
            if not seq or current > seq[-1]:
                if can_extend(seq, current, sums_set):
                    n = len(seq)
                    s = current
                    sums_set.add(s)
                    for i in range(n - 1, -1, -1):
                        s += seq[i]
                        sums_set.add(s)
                    seq.append(current)
            current += random.randint(1, 10)
        
        if len(seq) > len(best):
            best, best_sums = local_search(seq, sums_set, TIME_LIMIT - 10)
    
    # Final greedy extension
    extended = list(best)
    ext_sums = set(best_sums)
    for x in range(extended[-1] + 1, MAX_VAL + 1) if extended else range(1, MAX_VAL + 1):
        if can_extend(extended, x, ext_sums):
            n = len(extended)
            s = x
            ext_sums.add(s)
            for i in range(n - 1, -1, -1):
                s += extended[i]
                ext_sums.add(s)
            extended.append(x)
    
    if len(extended) > len(best):
        best = extended
    
    # Verify and return
    if is_valid(best):
        return best
    else:
        # Fallback: return a known valid sequence
        return [1, 2, 4, 8, 16, 32, 64, 128]
