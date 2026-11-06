import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 256
    
    def is_valid_addition(S, new_elem):
        """Check if adding new_elem to set S maintains validity.
        S is a sorted list. new_elem must be < all elements in S for the 
        a<b<=c constraint to only involve new_elem as 'a'.
        But new_elem could also be between elements, so we need full check.
        """
        # For efficiency, assume new_elem is smaller than most elements
        # We need: for all a<b<=c in S∪{new_elem}, a ∤ (b+c)
        # Cases:
        # 1. new_elem = a: for all b,c in S with b>new_elem, c>=b, new_elem ∤ (b+c)
        # 2. new_elem = b: for all a in S with a<new_elem, c in S with c>=new_elem, a ∤ (new_elem+c)
        # 3. new_elem = c: for all a<b in S with b<new_elem, a ∤ (b+new_elem)
        
        S_set = set(S)
        S_sorted = sorted(S)
        
        # Case 1: new_elem is the smallest (a position)
        for b in S_sorted:
            if b <= new_elem:
                continue
            for c in S_sorted:
                if c < b:
                    continue
                if (b + c) % new_elem == 0:
                    return False
        
        # Case 2: new_elem is b (middle)
        for a in S_sorted:
            if a >= new_elem:
                continue
            for c in S_sorted:
                if c < new_elem:
                    continue
                if (new_elem + c) % a == 0:
                    return False
        
        # Case 3: new_elem is c (largest)
        for a in S_sorted:
            for b in S_sorted:
                if a >= b:
                    continue
                if b >= new_elem:
                    continue
                if (b + new_elem) % a == 0:
                    return False
        
        return True
    
    def full_validate(S):
        """Full validation of set S."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def greedy_extend(base_set, candidates, rng):
        """Try to add candidates to base_set greedily."""
        S = list(base_set)
        S_set = set(S)
        # Shuffle candidates
        cand = list(candidates)
        rng.shuffle(cand)
        
        for elem in cand:
            if elem in S_set:
                continue
            if is_valid_addition(S, elem):
                S.append(elem)
                S_set.add(elem)
        
        return S
    
    def greedy_extend_descending(base_set, candidates, rng):
        """Try to add candidates in descending order."""
        S = list(base_set)
        S_set = set(S)
        cand = sorted(candidates, reverse=True)
        
        for elem in cand:
            if elem in S_set:
                continue
            if is_valid_addition(S, elem):
                S.append(elem)
                S_set.add(elem)
        
        return S
    
    best_set = list(range(192, 257))  # seed: 65 elements
    best_size = len(best_set)
    
    # Try multiple strategies
    rng = random.Random(42)
    
    # Strategy 1: Start with upper tail, try to add smaller numbers
    for trial in range(20):
        if time.time() > deadline - 2.0:
            break
        
        # Different upper tail starts
        tail_start = rng.randint(120, 200)
        base = list(range(tail_start, 257))
        
        # Candidates are numbers below tail_start
        candidates = list(range(1, tail_start))
        
        S = greedy_extend(base, candidates, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Start with just the top, build up
    for trial in range(10):
        if time.time() > deadline - 2.0:
            break
        
        base = list(range(200, 257))
        candidates = list(range(1, 200))
        
        S = greedy_extend(base, candidates, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Descending greedy from full range
    for trial in range(5):
        if time.time() > deadline - 2.0:
            break
        
        # Start empty, add from large to small
        S = []
        S_set = set()
        for elem in range(256, 0, -1):
            if is_valid_addition(S, elem):
                S.append(elem)
                S_set.add(elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 4: Try various upper tails and extend
    for tail_start in range(100, 257, 5):
        if time.time() > deadline - 2.0:
            break
        
        base = list(range(tail_start, 257))
        candidates = list(range(1, tail_start))
        
        S = greedy_extend(base, candidates, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final validation
    if not full_validate(best_set):
        # Fallback to known valid set
        best_set = list(range(192, 257))
    
    return best_set
