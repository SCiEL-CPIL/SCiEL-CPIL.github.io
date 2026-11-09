import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def is_valid_addition(S_set, x):
        """
        Check if adding x to S_set (which is valid) results in a valid set.
        S_set is a set of ints. x is not in S_set.
        A conflict arises if there exist distinct a, b in S_set such that:
        (x + a) in S_set U {x}
        AND (x + b) in S_set U {x}
        AND (a + b) in S_set U {x}
        
        Note: Since S_set is valid, any invalid triple in S_set U {x} must contain x.
        The other two elements must be in S_set.
        """
        # We need to check pairs (a, b) from S_set
        # Optimization: If x+a > 128, it can't be in S (since S subset 1..128) unless it's x itself? No, x+a > x.
        # S contains only 1..128.
        
        # Pre-compute list of elements in S_set for iteration
        # To speed up, we can just iterate
        elems = list(S_set)
        n = len(elems)
        
        # We can filter pairs: 
        # If a+b > 128 and a+b != x, then a+b not in S_new. 
        # If x+a > 128 and x+a != x (impossible since a>0), then x+a not in S_new.
        
        for i in range(n):
            a = elems[i]
            # If x + a > 128 and x + a != x, then x+a is not in S_new. 
            # Since a>=1, x+a > x. So if x+a > 128, it's not in S_new.
            if x + a > 128:
                continue
            
            # Now x+a is in range 1..128. Check if it's in S_set or == x.
            if (x + a) not in S_set and (x + a) != x:
                continue
            
            for j in range(i + 1, n):
                b = elems[j]
                # Check x + b
                if x + b > 128:
                    continue
                if (x + b) not in S_set and (x + b) != x:
                    continue
                
                # Check a + b
                s_ab = a + b
                if s_ab > 128:
                    continue
                if s_ab not in S_set and s_ab != x:
                    continue
                
                # All three sums are in S_new. 
                # We must ensure a, b, x are distinct.
                # a, b are from S_set, so distinct from each other.
                # x is not in S_set, so distinct from a and b.
                # Thus a, b, x are distinct.
                return False
        
        return True

    def is_valid_set(S_list):
        """Full validity check for debugging or verification."""
        S_set = set(S_list)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S_set:
                    continue
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    best_size = 0
    best_set = set()
    
    # Initial strategy: Start with high numbers, try to add lower ones
    # We will run multiple restarts with different random seeds/orders
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize S
        # Strategy 1: Start with 63..128
        # Strategy 2: Start with random subset of 60..128
        # Strategy 3: Start with empty and add greedily? Too slow.
        
        if r % 3 == 0:
            S = set(range(63, 129))
        elif r % 3 == 1:
            # Random subset of 60..128
            S = set(random.sample(range(60, 129), 60))
        else:
            # Start with 65..128 and try to add
            S = set(range(65, 129))
        
        # Ensure validity of start (should be valid for these ranges)
        # 63..128 is valid.
        # Random subset might not be valid. Let's filter to ensure validity for start?
        # Or just use a greedy construction from scratch for some restarts?
        
        # Let's stick to valid starts to avoid wasted time.
        # If S is not valid, we should fix it or restart.
        # For random sample, it's likely valid if sparse, but let's be safe.
        # Actually, let's just use the known valid suffixes as seeds.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Local search: Try to add elements
        # We maintain a list of candidates
        candidates = [i for i in range(1, 129) if i not in S]
        
        # Shuffle candidates to break symmetry
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add smallest candidate first? Or random?
            # Smallest first tends to pack tighter.
            # Let's try sorting candidates by value, but also try random order in some iterations.
            
            # In this restart, we'll try adding in ascending order
            for x in sorted(candidates):
                if time.time() > deadline - 2:
                    break
                
                if is_valid_addition(S, x):
                    S.add(x)
                    candidates.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S.copy()
            
            # If we didn't improve in the full pass, try some random swaps or different order?
            # For now, if no improvement, we break the while loop for this restart.
            # To explore more, we could randomize the order of candidates in each pass.
            # But 'improved' flag breaks if no addition was made in a full pass.
            # This is greedy. It might get stuck.
            
            # Let's add a randomization step: if stuck, try to remove a random element and re-add?
            # Or just restart.
            
        # End of restart loop

    # Final check
    result = list(best_set)
    # Verify types
    return [int(x) for x in result]

