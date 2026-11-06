import time
import random
from itertools import combinations

def can_add(x, S):
    # S is a set or list of selected numbers
    # Check if x can be added to S without violating constraints
    
    # 1. x must not divide sum of any 3 in S
    # Precompute sums of triples in S? No, S changes.
    # Just iterate.
    S_list = list(S)
    for b, c, d in combinations(S_list, 3):
        if (b + c + d) % x == 0:
            return False
    
    # 2. For each a in S, a must not divide (x + u + v) for any u, v in S\{a}
    for a in S_list:
        others = [v for v in S_list if v != a]
        # We need to check all pairs u, v in others
        for u, v in combinations(others, 2):
            if (x + u + v) % a == 0:
                return False
    return True

def greedy_construct(order):
    S = []
    for x in order:
        if can_add(x, S):
            S.append(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50 # 50s safety margin
    
    # Initial greedy
    best_set = greedy_construct(range(192, 0, -1))
    best_size = len(best_set)
    
    if best_size == 48:
        return best_set
    
    # Random restarts
    candidates = list(range(1, 193))
    
    while time.time() < deadline:
        # Try a random order, biased towards large numbers
        # We can just shuffle, or sort by random key
        # Let's try a mix of random and greedy
        order = candidates[:]
        random.shuffle(order)
        
        # To bias towards large numbers, we can sort by (random() - value/192)
        # Or just shuffle and let it be.
        # Actually, large numbers are safer.
        # Let's try sorting by value descending, but with some noise
        # Or just random restarts of the greedy.
        
        current_set = greedy_construct(order)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            if best_size == 48:
                return best_set
        
        # Local search on current_set
        # Try to add any missing number
        changed = True
        while changed and time.time() < deadline:
            changed = False
            missing = [x for x in candidates if x not in current_set]
            random.shuffle(missing)
            for x in missing:
                if can_add(x, current_set):
                    current_set.append(x)
                    changed = True
                    if len(current_set) > best_size:
                        best_set = current_set
                        best_size = len(best_set)
                        if best_size == 48:
                            return best_set
                    break # restart missing check? Or continue?
                    # If we add one, the set changes, so previous missing checks are invalid.
                    # So break and re-evaluate.
        
        # Try swaps if stuck
        # Pick random a in current_set, b not in current_set
        # If (current_set - {a}) + {b} is valid, do it
        # This might not increase size, but might allow future additions.
        # Given time, maybe skip swaps if we are close to 48?
        # If we are far, swaps might help escape local optima.
        
        if len(current_set) < 48 and time.time() < deadline:
            for _ in range(100): # limit swap attempts
                if time.time() > deadline: break
                if not current_set: break
                a = random.choice(current_set)
                missing = [x for x in candidates if x not in current_set]
                if not missing: break
                b = random.choice(missing)
                new_set = [x for x in current_set if x != a] + [b]
                # Check validity of new_set
                # We can just run can_add on b to (current_set - {a})?
                # No, we removed a. The validity of the rest is preserved?
                # Yes, removing an element cannot create a violation for the remaining ones.
                # So we only need to check if b can be added to (current_set - {a}).
                base_set = [x for x in current_set if x != a]
                if can_add(b, base_set):
                    current_set = new_set
                    # Update best if size increased? No, size same.
                    # But maybe this configuration allows more additions.
                    # So we continue the "add" loop in the next iteration of the while changed loop.
                    # But we are inside the restart loop.
                    # We should probably re-enter the "add" phase.
                    # Let's just break out of swap loop and go to next restart iteration?
                    # Or continue adding?
                    # Let's just break and let the outer loop handle it via the 'changed' logic?
                    # The 'changed' logic is in the while loop above.
                    # If we do a swap, we haven't increased size.
                    # We should try to add again.
                    # Let's just set a flag to retry adding.
                    pass # This logic is getting complex.
        
        # Simpler: Just rely on random restarts of greedy.
        # The greedy is fast.
        # 192 items, 48 selected.
        # ~192 * 48^3 ops ~ 20M ops?
        # 20M ops in Python is ~1-2 seconds.
        # So we can do ~20-30 restarts in 50s.
        # This might be enough to find 48 if it exists.

    return best_set
