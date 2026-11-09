"""Suite task: distinct subset sums, n=12, minimize the largest element."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return exactly 12 distinct positive integers <= 1048576 with all 4096
    subset sums distinct.  Minimizes the largest integer.

    Strategy:
    1. Start with the superincreasing sequence (powers of 2) which gives
       largest = 2048.
    2. Perform a local search to reduce the largest element while maintaining
       distinct subset sums. We try to "compress" the sequence by reducing
       elements where possible.
    
    The key insight is that we don't need a strictly superincreasing sequence.
    We just need all 2^12 = 4096 subset sums to be distinct. This is a
    Sidon-like condition but for subset sums.
    
    We use a greedy approach with backtracking or local search to find a
    sequence with smaller maximum.
    """
    _t0 = time.monotonic()
    _DEADLINE = 85.0  # safety margin before 90 s kill

    n = 12
    MAX_VAL = 1048576

    # Initial solution: powers of 2
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Largest = 2048
    
    # We want to minimize the largest element.
    # Let's try to find a sequence with largest element < 2048.
    
    # A known result: the minimal largest element for n distinct positive integers
    # with distinct subset sums is related to the problem of finding the smallest
    # maximum in a set of n integers with distinct subset sums.
    
    # For n=12, the minimal maximum is known to be less than 2048.
    # Let's search for it.
    
    # We'll use a randomized local search approach:
    # 1. Start with a valid sequence (powers of 2)
    # 2. Try to reduce elements while maintaining validity
    # 3. Use simulated annealing or hill climbing
    
    def is_valid(seq):
        """Check if all subset sums are distinct."""
        if len(seq) != n:
            return False
        if any(x <= 0 for x in seq):
            return False
        if max(seq) > MAX_VAL:
            return False
        if len(set(seq)) != n:
            return False
        
        # Check all 2^n subset sums
        sums = set()
        for mask in range(1 << n):
            s = 0
            for j in range(n):
                if mask & (1 << j):
                    s += seq[j]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    def count_collisions(seq):
        """Count the number of colliding subset sums."""
        sums = set()
        collisions = 0
        for mask in range(1 << n):
            s = 0
            for j in range(n):
                if mask & (1 << j):
                    s += seq[j]
            if s in sums:
                collisions += 1
            else:
                sums.add(s)
        return collisions
    
    # Start with powers of 2
    best_seq = [1 << i for i in range(n)]
    best_max = max(best_seq)
    
    # Try to improve by reducing the largest element
    # We'll do a local search: try changing one element at a time
    
    # First, let's try a systematic approach:
    # Fix the first k elements and search for the remaining ones
    
    # Actually, let's try a different approach:
    # Use a greedy construction with a smaller growth rate
    
    # The condition for distinct subset sums is weaker than superincreasing.
    # We can have a_k <= sum_{i<k} a_i + 1 and still have distinct sums
    # if the previous elements are chosen carefully.
    
    # Let's try to construct a sequence where each element is as small as possible
    # while maintaining distinct subset sums.
    
    def greedy_construct(max_limit):
        """Try to construct a sequence with all elements <= max_limit."""
        seq = []
        sums = {0}  # Current subset sums
        
        for i in range(n):
            found = False
            # Try values from 1 to max_limit
            for val in range(1, max_limit + 1):
                if val in seq:
                    continue
                
                # Check if adding val creates any collision
                new_sums = set()
                valid = True
                for s in sums:
                    ns = s + val
                    if ns in sums or ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if valid:
                    seq.append(val)
                    sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                return None
        
        return seq
    
    # Binary search on the maximum element
    # We know 2048 works. Let's find the minimum.
    
    # But greedy might not find the optimal solution.
    # Let's try a more sophisticated approach.
    
    # Actually, for n=12, we can afford a more exhaustive search.
    # Let's try to find the minimal maximum by checking if a sequence
    # exists with max element = M for decreasing M.
    
    # Start from 2048 and go down
    # But this could be slow. Let's use a smarter approach.
    
    # Known result: For n=12, the minimal maximum is 1346 (or similar).
    # Let's just do a local search from the powers of 2.
    
    # Local search: try to reduce each element
    import random
    random.seed(42)
    
    current_seq = list(best_seq)
    current_collisions = count_collisions(current_seq)
    
    # Simulated annealing
    T = 100.0
    T_min = 0.01
    cooling = 0.95
    
    while time.monotonic() < _DEADLINE:
        # Pick a random element to modify
        idx = random.randint(0, n - 1)
        old_val = current_seq[idx]
        
        # Try a new value
        # Bias towards smaller values
        new_val = old_val + random.randint(-50, 20)
        new_val = max(1, new_val)
        new_val = min(new_val, MAX_VAL)
        
        if new_val == old_val:
            continue
        
        # Check for duplicates
        if new_val in current_seq and current_seq.index(new_val) != idx:
            continue
        
        # Try the change
        current_seq[idx] = new_val
        new_collisions = count_collisions(current_seq)
        
        # Accept if better, or with some probability if worse
        if new_collisions < current_collisions:
            current_collisions = new_collisions
            if current_collisions == 0:
                # We have a valid solution
                if max(current_seq) < best_max:
                    best_seq = list(current_seq)
                    best_max = max(best_seq)
                # Continue searching for even better solutions
            else:
                pass
        else:
            # Reject if worse, or accept with probability exp(-delta/T)
            if new_collisions > current_collisions:
                delta = new_collisions - current_collisions
                if random.random() > pow(2.718281828, -delta / T):
                    current_seq[idx] = old_val
                    new_collisions = current_collisions
                else:
                    current_collisions = new_collisions
            else:
                current_collisions = new_collisions
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final check
    if count_collisions(best_seq) == 0:
        return best_seq
    else:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
