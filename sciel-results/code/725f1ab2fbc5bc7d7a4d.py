import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    best_set = []
    best_size = 0
    
    # Candidates 1..256
    candidates = list(range(1, 257))
    
    # To improve, we can try different orderings.
    # Larger numbers tend to be "safer" because they don't divide as many small sums.
    # But small numbers allow more combinations?
    # Actually, if we pick small numbers, the sums are small and dense, leading to many divisibility conflicts.
    # Picking large numbers spreads sums out.
    # Let's bias towards larger numbers in the shuffle.
    
    while time.time() < deadline:
        # Shuffle with bias? Or just random.
        # Let's try random shuffle.
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {0}
        
        for c in candidates:
            # Check if c can be added
            # New sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + c)
            
            all_sums = current_sums.union(new_sums)
            # Remove 0? No, 0 is not in subset sums (non-empty).
            # But 0 is in current_sums for generation.
            # We only care about non-empty sums.
            # So check all_sums - {0}
            
            valid = True
            # Sort sums for efficient check?
            # Or just check divisibility directly.
            # Since we just added c, we only need to check:
            # 1. New sums among themselves
            # 2. New sums vs old sums
            # 3. Old sums are already valid
            
            # So we can optimize.
            # Check if any new sum is divisible by any old sum or new sum.
            # And if any old sum is divisible by any new sum.
            
            # Let's just do the full check on all_sums - {0} for simplicity and correctness first.
            # If too slow, optimize.
            
            sums_to_check = list(all_sums)
            sums_to_check.remove(0)
            sums_to_check.sort()
            
            for i, s in enumerate(sums_to_check):
                # Check multiples
                # Max sum is sum(current_set) + c
                max_val = sums_to_check[-1]
                for m in range(2, max_val // s + 1):
                    if s * m in all_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(c)
                current_sums = all_sums
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
