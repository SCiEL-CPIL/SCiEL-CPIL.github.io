import time
import random

def _can_add(S_mask, x):
    """
    Check if x can be added to S_mask without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c in S_new.
    If x is new, any new triple must involve x.
    So we check if there exist distinct a, b in S such that:
    1. a + x in S
    2. b + x in S
    3. a + b in S or a + b == x
    
    S_mask is an integer where bit (v-1) is 1 if v is in S.
    """
    # Collect candidates a such that a+x is in S
    # a must be in S and a+x <= 128
    # a < x is not required, just a != x (which is true since x not in S)
    # a, b distinct
    
    candidates = []
    # Iterate through all possible a in 1..128-x
    # We can iterate through set bits in S that are <= 128-x
    # Or just loop 1 to 128-x and check bits
    
    # Optimization: Only check a if bit(a-1) is set AND bit(a+x-1) is set
    # We can do this by iterating over set bits of S, but filtering by value.
    
    # Let's just loop a from 1 to 128-x
    # This is at most 128 iterations.
    # Inside, we check if a is in S and a+x is in S.
    
    # To make it faster, we can precompute the list of elements in S?
    # No, S_mask changes. But we can extract elements if needed.
    # However, looping 1..128 is fast enough in Python if simple.
    
    # Let's try to be smarter.
    # We need pairs (a, b) from the set K = {a in S | a+x in S}.
    # If |K| < 2, return True.
    
    # First pass: find K
    K = []
    # We can iterate over set bits of S
    # But checking a+x in S is a bit shift.
    
    # Let's just loop a in range(1, 129-x)
    for a in range(1, 129 - x):
        if (S_mask & (1 << (a - 1))) and (S_mask & (1 << (a + x - 1))):
            K.append(a)
            
    nK = len(K)
    if nK < 2:
        return True
        
    # Now check if there exist distinct a, b in K such that a+b in S or a+b == x
    # Loop i, j in K
    for i in range(nK):
        a = K[i]
        for j in range(i + 1, nK):
            b = K[j]
            s = a + b
            if s == x:
                return False
            if (S_mask & (1 << (s - 1))):
                return False
    return True

def _is_valid(S_mask):
    """
    Check if S_mask is valid.
    """
    # Extract elements
    elements = []
    for i in range(1, 129):
        if S_mask & (1 << (i - 1)):
            elements.append(i)
            
    n = len(elements)
    # Check all triples? O(n^3) is too slow for frequent calls.
    # But we only call this on final result or initial cleanup.
    # For cleanup, we can just remove elements one by one if they cause conflict.
    # This function is for final verification.
    
    # A valid set has no forbidden triples.
    # We can check by trying to add each element to an empty set? No, that's O(N^2 * N) = O(N^3).
    # N=128, N^3 ~ 2M. Might be okay for one call.
    
    # Better: Just check all triples (a,b,c) in elements.
    # If n is large, this is slow.
    # But we can use the property: if S is valid, then for any x in S, S \ {x} is valid? No.
    
    # Let's just do a direct check for the final result.
    # Since we construct it incrementally with _can_add, it should be valid.
    # We use this just to be safe.
    
    # Actually, we can just return True if we trust _can_add.
    # But let's implement a check.
    
    # For each pair (a, b) in elements, if a+b in S, check if there is c such that a+c, b+c in S.
    # This is still complex.
    
    # Given time constraints, we will assume _can_add is correct and use a simpler final check
    # or just rely on the construction.
    # However, the prompt says "Any broken rule makes the whole result invalid".
    # So we must be sure.
    
    # Let's do a full check for the final set only.
    # O(N^3) for N=128 is ~2 million operations. In Python, this might take 0.1-0.5s. Acceptable for one call.
    
    S_set = set(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            if a + b not in S_set:
                continue
            for k in range(j + 1, n):
                c = elements[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def construct():
    deadline = time.time() + 85
    rng = random.Random(42)
    
    best_mask = 0
    best_size = 0
    
    # Strategy: Greedy construction + Local Search
    
    # 1. Initial Set: Try different random seeds and structures
    # We will run multiple restarts.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random subset or structured set
        # Let's try starting with a random set of size ~60-80 and cleaning it
        # Or start with upper half which is known to be valid.
        
        if r == 0:
            # Upper half
            current_mask = 0
            for i in range(65, 129):
                current_mask |= (1 << (i - 1))
        elif r == 1:
            # Lower half
            current_mask = 0
            for i in range(1, 65):
                current_mask |= (1 << (i - 1))
        else:
            # Random
            current_mask = 0
            # Pick random 70 elements
            elems = rng.sample(range(1, 129), 70)
            for e in elems:
                current_mask |= (1 << (e - 1))
        
        # Cleanup: Remove elements that cause conflicts
        # Iterate through elements, if removing it makes the set valid? No, we want to KEEP it if possible.
        # We want to remove the MINIMUM number of elements to make it valid.
        # Heuristic: Iterate elements in random order. If adding it to a growing set causes conflict, skip it.
        
        # Let's rebuild from scratch using greedy addition for the random case.
        if r > 1:
             current_mask = 0
             candidates = list(range(1, 129))
             rng.shuffle(candidates)
             for x in candidates:
                 if _can_add(current_mask, x):
                     current_mask |= (1 << (x - 1))
        
        # Now Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add an element
            # Get missing elements
            missing = [x for x in range(1, 129) if not (current_mask & (1 << (x - 1)))]
            rng.shuffle(missing)
            
            added = False
            for x in missing:
                if _can_add(current_mask, x):
                    current_mask |= (1 << (x - 1))
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try a swap
            # Pick random y in S, z in missing
            # Limit attempts
            swap_attempts = 20
            for _ in range(swap_attempts):
                if time.time() > deadline - 5:
                    break
                if not current_mask:
                    break
                
                # Pick y
                # Random set bit
                # We can just pick a random index 1..128 and check if in S, retry if not
                # Or maintain a list of elements in S? 
                # For speed, let's just pick random x from 1..128 and check.
                
                y_found = False
                while not y_found:
                    y = rng.randint(1, 128)
                    if current_mask & (1 << (y - 1)):
                        y_found = True
                
                z_found = False
                while not z_found:
                    z = rng.randint(1, 128)
                    if not (current_mask & (1 << (z - 1))):
                        z_found = True
                
                # Check if swap y->z is valid
                temp_mask = current_mask & ~(1 << (y - 1))
                if _can_add(temp_mask, z):
                    current_mask = temp_mask | (1 << (z - 1))
                    improved = True
                    break
            
            if improved:
                continue
            
            # 3. Stuck. Try random removal to escape?
            # Or just break and restart.
            # Let's try a few random removals
            removal_attempts = 5
            for _ in range(removal_attempts):
                if time.time() > deadline - 5:
                    break
                if not current_mask:
                    break
                y = rng.randint(1, 128)
                if current_mask & (1 << (y - 1)):
                    current_mask &= ~(1 << (y - 1))
                    improved = True # Force loop to try additions again
                    break
            
            # If still stuck, break to next restart
            if not improved:
                break
        
        # Update best
        size = bin(current_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = current_mask
    
    # Final Verification
    # Extract elements
    result = []
    for i in range(1, 129):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    # Verify
    if not _is_valid(best_mask):
        # Fallback
        result = list(range(65, 129))
        
    return [int(x) for x in result]
