import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    def is_valid(seq):
        """
        Checks if all nonempty consecutive block sums are distinct.
        seq is a sorted list of ints.
        """
        n = len(seq)
        if n == 0:
            return True
        
        sums = set()
        # Calculate all subarray sums
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += seq[j]
                if current_sum in sums:
                    return False
                sums.add(current_sum)
        return True

    def greedy_extend(seq):
        """
        Tries to add the smallest possible integer to the sorted list 'seq'
        such that the new list remains valid.
        Returns the new list if successful, else None.
        """
        if time.time() > deadline:
            return None
        
        current_set = set(seq)
        last_val = seq[-1] if seq else 0
        
        # We can add any value from 1 to 256 not in current_set.
        # To maximize length, we generally prefer smaller values to leave room.
        # But inserting in the middle might be necessary.
        # Since the list is sorted, adding a value $v$ means inserting it at the correct position.
        
        # Optimization: We only need to check sums involving the new element.
        # However, for simplicity and given N is small (<20), full check is fine.
        # But we can prune: if we add $v$, the new sums are:
        # 1. Sums that were already there (unchanged)
        # 2. Sums that include $v$.
        # If any new sum equals an old sum, or two new sums equal each other, it's invalid.
        
        # For speed, let's just try candidates and use is_valid.
        # To be smarter, try candidates in increasing order.
        
        candidates = [v for v in range(1, 257) if v not in current_set]
        
        # Heuristic: Try smaller values first?
        # Or values that are "far" from existing ones?
        # Let's try smallest first to keep the sequence "compact".
        
        for v in candidates:
            new_seq = list(seq)
            # Insert v in sorted order
            import bisect
            pos = bisect.bisect_left(new_seq, v)
            new_seq.insert(pos, v)
            
            if is_valid(new_seq):
                return new_seq
        
        return None

    best_seq = []
    
    # 1. Initial Greedy Construction
    current_seq = []
    while True:
        next_seq = greedy_extend(current_seq)
        if next_seq is None:
            break
        current_seq = next_seq
        if len(current_seq) > len(best_seq):
            best_seq = current_seq[:]
        
        if time.time() > deadline:
            return best_seq

    # 2. Local Search / Randomized Restart
    # If greedy gets stuck, we try to perturb the sequence.
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline:
            break
            
        # Strategy: Take the best_seq (or current_seq if it's longer, but best is safer), 
        # remove a random element, and try to greedily extend from there.
        # Or swap an element.
        
        # Let's try: Pick a random element from best_seq, remove it, 
        # then try to greedily extend the result.
        # If the result is longer than best_seq, update.
        
        if not best_seq:
            break
            
        # Copy best_seq
        base_seq = best_seq[:]
        
        # Remove a random element
        idx = random.randint(0, len(base_seq) - 1)
        base_seq.pop(idx)
        
        # Try to extend this perturbed sequence
        perturbed_seq = base_seq[:]
        
        # Greedy extend the perturbed sequence
        while True:
            if time.time() > deadline:
                break
            next_seq = greedy_extend(perturbed_seq)
            if next_seq is None:
                break
            perturbed_seq = next_seq
            
        if len(perturbed_seq) > len(best_seq):
            best_seq = perturbed_seq[:]
            
        # Also try swapping:
        # Pick an element, replace with a random other, check validity, then extend.
        # This is more complex. The remove-and-extend is a good proxy for "moving" an element.

    return best_seq
