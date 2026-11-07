"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N_MAX = 256
ALL_CANDIDATES = list(range(2, N_MAX + 1))

def _can_add(S_list, x):
    """
    Check if x can be added to set S (given as a list of ints).
    Returns True if no triple (a, b, x) with a,b in S violates the condition.
    """
    n = len(S_list)
    # To optimize, we can break early
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            # Check a | b*x
            if (b * x) % a == 0:
                return False
            # Check b | a*x
            if (a * x) % b == 0:
                return False
            # Check x | a*b
            if (a * b) % x == 0:
                return False
    return True

def _is_valid_set(S_list):
    """Verify the full 3-wise constraint for set S (list of ints)."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Use a list for S to avoid overhead of set conversions in _can_add
    # We maintain S as a set for O(1) membership, but convert to list for checking.
    # Alternatively, maintain S as a list and use a set for membership.
    S_set = set()
    S_list = []
    
    # Greedy initialization: Try adding candidates in a specific order.
    # Larger numbers are "safer" to add first because they are less likely to divide products of smaller numbers?
    # Actually, if x is large, x | a*b is less likely.
    # a | b*x is more likely if a is small.
    # Let's try descending order first.
    candidates_desc = list(range(N_MAX, 1, -1))
    
    for x in candidates_desc:
        if time.time() > deadline:
            break
        if x not in S_set:
            if _can_add(S_list, x):
                S_set.add(x)
                S_list.append(x)
    
    best_S_list = S_list.copy()
    best_size = len(S_list)
    
    # Local search
    # We will try to add elements, then swap elements.
    # To speed up, we cache the list of candidates not in S.
    
    while time.time() < deadline:
        # 1. Try to add any missing element
        added = False
        # Shuffle candidates to avoid bias
        missing = [x for x in ALL_CANDIDATES if x not in S_set]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            if _can_add(S_list, x):
                S_set.add(x)
                S_list.append(x)
                added = True
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
        
        if added:
            continue
            
        # 2. If no add possible, try swaps
        # Pick a random element to remove and try to replace it with a missing one
        # To be efficient, we don't try all pairs, just a few random ones.
        
        if not S_list or not missing:
            break
            
        # Limit swap attempts per iteration to keep it fast
        max_swap_attempts = 20
        swap_success = False
        
        for _ in range(max_swap_attempts):
            if time.time() > deadline:
                break
            
            y = random.choice(S_list)
            # Try to find a replacement x for y
            # We can just pick a random missing x
            x = random.choice(missing)
            
            # Check if S - {y} + {x} is valid
            # This is equivalent to checking if x can be added to S - {y}
            # And that y was not needed for validity? No, we just check the new set.
            
            # Efficient check:
            # Remove y from S_list temporarily? No, just create a modified list or pass a flag.
            # Simpler: Create S_list without y.
            # Since N is small, we can just filter.
            
            # Optimization: Pre-check if x conflicts with any pair in S\{y}
            # We can iterate through S_list and skip y.
            
            valid = True
            # We need to check all pairs (a,b) in S\{y} against x.
            # Also check if x conflicts with y? No, y is removed.
            # But we must ensure that removing y doesn't break anything?
            # The condition is "for every three distinct members...".
            # If we remove y, any triple involving y is gone.
            # New triples involve x and two others from S\{y}.
            # So we just need to check if x is compatible with all pairs in S\{y}.
            
            # Let's do it efficiently.
            # We can't easily skip y in the nested loop of _can_add without modifying it.
            # Let's write a helper or inline it.
            
            # Inline check for x against S_list excluding y
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                if a == y:
                    continue
                for j in range(i + 1, n):
                    b = S_list[j]
                    if b == y:
                        continue
                    # Check a | b*x
                    if (b * x) % a == 0:
                        valid = False
                        break
                    # Check b | a*x
                    if (a * x) % b == 0:
                        valid = False
                        break
                    # Check x | a*b
                    if (a * b) % x == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Perform swap
                S_set.remove(y)
                S_set.add(x)
                # Update S_list
                for k in range(len(S_list)):
                    if S_list[k] == y:
                        S_list[k] = x
                        break
                swap_success = True
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_S_list = S_list.copy()
                break # Restart the while loop to re-evaluate adds/swaps
        
        if not swap_success:
            # No swap worked in this batch. 
            # To avoid getting stuck, we could try a more aggressive restart or just continue.
            # Given the time limit, we might just continue the loop which will try again.
            # However, if we are stuck in a local optimum, we might need to randomize more.
            # For now, we rely on the random choice of y and x.
            pass

    # Final verification
    if not _is_valid_set(best_S_list):
        # Fallback to a known valid set if something went wrong
        # The greedy set from the start was valid, so best_S_list should be valid.
        # But just in case.
        best_S_list = [x for x in range(129, 257) if x in best_S_list] # This is not necessarily valid.
        # Better fallback: empty set? No, must be nonempty.
        # Let's just return the greedy result if valid, else [256]
        if not _is_valid_set(best_S_list):
             best_S_list = [256]
    
    return [int(x) for x in best_S_list]

# EVOLVE-BLOCK-END
