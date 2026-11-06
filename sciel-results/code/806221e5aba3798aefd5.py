import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # safety margin before 60s kill
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

    def can_extend(prefix, new_val, existing_sums):
        """Check if adding new_val to prefix keeps all block sums distinct."""
        n = len(prefix)
        
        # New sums involving new_val: new_val alone, and suffix sums of prefix + new_val
        new_sums = []
        s = new_val
        new_sums.append(s)
        for i in range(n - 1, -1, -1):
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

    def extend_greedy(seq, sums_set, start_val=1):
        """Greedily extend sequence from start_val upward."""
        for x in range(start_val, MAX_VAL + 1):
            if can_extend(seq, x, sums_set):
                n = len(seq)
                s = x
                sums_set.add(s)
                for i in range(n - 1, -1, -1):
                    s += seq[i]
                    sums_set.add(s)
                seq.append(x)
        return seq, sums_set

    def greedy_from_start(start_val):
        """Build sequence greedily starting from a given value."""
        seq = []
        sums_set = set()
        return extend_greedy(seq, sums_set, start_val)

    def greedy_desc_start(high_start):
        """Start with a high value and greedily add smaller values, then extend upward."""
        seq = []
        sums_set = set()
        # First, try to build a sequence ending near high_start
        # Actually, let's try a different approach: start with small values but 
        # use a smarter selection
        
        # Alternative: try to start with values near the top and work down
        # But we need strictly increasing, so let's try different strategies
        
        # Strategy: start with a specific first element and greedily extend
        seq = [high_start] if high_start <= MAX_VAL else []
        sums_set = set()
        if seq:
            sums_set.add(high_start)
        return extend_greedy(seq, sums_set, high_start + 1)

    best = []
    best_sums = set()

    # Strategy 1: Greedy from 1 (baseline)
    seq, sums_set = greedy_from_start(1)
    if len(seq) > len(best):
        best = seq
        best_sums = sums_set

    # Strategy 2: Try various starting values
    for start in range(1, 20):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        seq, sums_set = greedy_from_start(start)
        if len(seq) > len(best):
            best = seq
            best_sums = sums_set

    # Strategy 3: Greedy descending start - start with higher values
    # This explores different sum-spaces
    for high_start in range(10, 50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        seq, sums_set = greedy_desc_start(high_start)
        if len(seq) > len(best):
            best = seq
            best_sums = sums_set

    # Strategy 4: Local search / backtracking to extend
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
            if seq:
                for x in range(seq[-1] + 1, MAX_VAL + 1):
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
            else:
                for x in range(1, MAX_VAL + 1):
                    if can_extend(seq, x, sums_set):
                        sums_set.add(x)
                        seq.append(x)
                        improved = True
                        break
            
            if not improved:
                # Try removing last element and trying alternatives
                if len(seq) > 1:
                    removed = seq.pop()
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

    # Apply local search to best
    best, best_sums = local_search(best, best_sums, TIME_LIMIT - 10)

    # Strategy 5: Randomized restarts
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Random starting point
        start = random.randint(1, 30)
        seq, sums_set = greedy_from_start(start)
        
        # Apply local search
        seq, sums_set = local_search(seq, sums_set, TIME_LIMIT - 10)
        
        if len(seq) > len(best):
            best = seq
            best_sums = sums_set

    # Strategy 6: Try to build with specific patterns
    # Exponential growth patterns
    for base in range(1, 10):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
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
            current = current * 2
        # Then greedily extend
        seq, sums_set = extend_greedy(seq, sums_set, (seq[-1] + 1) if seq else 1)
        if len(seq) > len(best):
            best = seq
            best_sums = sums_set

    # Final verification
    if is_valid(best):
        return best
    else:
        # Fallback: return a known valid sequence
        return [1, 2, 4, 8, 16, 32, 64, 128]
