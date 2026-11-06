import time
import itertools

def _is_sidon(lst):
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_4_sidon(ambient):
    for combo in itertools.combinations(ambient, 4):
        if _is_sidon(list(combo)):
            return True
    return False

def _max_sidon_size(ambient):
    # Check for 5 first? No, if we have 4, value is >=4.
    # We want to find if it's 3.
    if not _has_4_sidon(ambient):
        # Check if 3 exists (it does)
        return 3
    else:
        # Check for 5?
        # If we find 5, value is 5.
        # But we are searching for 3.
        # If we fail to find 3, we return the best candidate found so far.
        # For the purpose of the search, if _has_4_sidon is True, 
        # the value is at least 4.
        # We can just return 4 as a lower bound for sorting, 
        # or calculate exact max if needed.
        # But to save time, we can just return 4 if we find a 4-Sidon.
        # Actually, we should check if it's 5 to be safe?
        # No, 3 is the target. 4 is the fallback.
        # If we find a set with no 4-Sidon, it's 3.
        # If all sets have 4-Sidon, we return one with the smallest "excess" or just any 4.
        # Let's just return 4 if _has_4_sidon is True.
        return 4

def construct():
    deadline = time.time() + 80
    
    import random
    rng = random.Random(123)
    
    best_cand = None
    best_val = float('inf')
    
    # Try to find a set with no 4-Sidon
    for i in range(5000):
        if time.time() > deadline:
            break
        
        # Generate dense random set
        # Bias towards small numbers to increase collisions
        max_val = rng.randint(20, 40)
        cand = sorted(rng.sample(range(max_val + 1), 12))
        
        # Check if it has a 4-Sidon
        if not _has_4_sidon(cand):
            return cand
        
        # If not, track the best (lowest max Sidon size)
        # Since we know it has 4, value is at least 4.
        # We can try to check for 5 to distinguish, but 4 is good enough if we can't get 3.
        # However, if we find a set with 4, and later find one with 5, we prefer 4.
        # So we should check if it has 5.
        has_5 = False
        for combo in itertools.combinations(cand, 5):
            if _is_sidon(list(combo)):
                has_5 = True
                break
        
        val = 5 if has_5 else 4
        
        if val < best_val:
            best_val = val
            best_cand = cand
    
    if best_cand is None:
        return list(range(12))
        
    return best_cand
