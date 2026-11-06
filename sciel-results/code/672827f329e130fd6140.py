import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Greedy construction
    chosen = []
    counts = [0] * 2001  # counts[s] = number of representations of s as a+b, a<=b
    chosen_set = set()
    
    for x in range(1, 1001):
        # Check if adding x keeps all sums <= 2
        ok = True
        new_sums = []
        for y in chosen:
            s = x + y
            if s <= 2000:
                if counts[s] >= 2:
                    ok = False
                    break
                new_sums.append(s)
        if ok:
            s2 = 2 * x
            if s2 <= 2000 and counts[s2] >= 2:
                ok = False
            else:
                if s2 <= 2000:
                    new_sums.append(s2)
        if ok:
            for s in new_sums:
                counts[s] += 1
            chosen.append(x)
            chosen_set.add(x)
    
    best_set = set(chosen)
    best_size = len(chosen)
    
    # Local search: try to improve by removing one element and adding another
    # We'll do multiple passes of hill climbing
    
    def can_add(x, current_set, cnt):
        """Check if x can be added to current_set without violating B2[2]"""
        if x in current_set:
            return False
        for y in current_set:
            s = x + y
            if s <= 2000 and cnt[s] >= 2:
                return False
        s2 = 2 * x
        if s2 <= 2000 and cnt[s2] >= 2:
            return False
        return True
    
    def add_element(x, current_list, current_set, cnt):
        """Add x to the set, updating counts"""
        for y in current_list:
            s = x + y
            cnt[s] += 1
        cnt[2 * x] += 1
        current_list.append(x)
        current_set.add(x)
    
    def remove_element(x, current_list, current_set, cnt):
        """Remove x from the set, updating counts"""
        for y in current_list:
            if y != x:
                s = x + y
                cnt[s] -= 1
        cnt[2 * x] -= 1
        current_list.remove(x)
        current_set.discard(x)
    
    # Hill climbing: repeatedly try to find a swap that increases size
    # We try removing each element and see if we can add a different one
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        # Try each element in the set for removal
        for idx, x in enumerate(list(best_set)):
            if elapsed() > DEADLINE:
                break
            # Remove x
            cnt = [0] * 2001
            temp_set = set(best_set)
            temp_set.discard(x)
            temp_list = sorted(temp_set)
            # Rebuild counts for temp_set
            for i in range(len(temp_list)):
                for j in range(i, len(temp_list)):
                    s = temp_list[i] + temp_list[j]
                    if s <= 2000:
                        cnt[s] += 1
            
            # Try to add any number not in temp_set
            for candidate in range(1, 1001):
                if candidate in temp_set:
                    continue
                if can_add(candidate, temp_set, cnt):
                    # This swap increases or maintains size (removes 1, adds 1 = same size)
                    # But we want to INCREASE size, so this alone doesn't help
                    # Actually removing 1 and adding 1 keeps same size
                    # We need a different approach
                    pass
            # This approach doesn't increase size directly
            # Instead, let's try: remove x, then greedily add as many as possible
            # and see if we end up with more than before
            break  # Skip this expensive approach for now
    
    # Better approach: try adding elements that weren't in the greedy, 
    # by first removing some elements that block them
    
    # Let's do a simpler local search: try to add each candidate 1-1000 not in set
    # by removing one element that blocks it
    for candidate in range(1, 1001):
        if elapsed() > DEADLINE:
            break
        if candidate in best_set:
            continue
        # Find which sums are blocked
        # For candidate to be added, for all y in best_set, counts[candidate+y] must be < 2
        # and counts[2*candidate] must be < 2
        # Find blocking elements
        blockers = set()
        for y in best_set:
            s = candidate + y
            if s <= 2000:
                # If counts[s] >= 2, then y is a blocker (one of the pairs summing to s)
                # Actually we need to find which elements form the existing pairs
                pass
        # This is getting complex. Let's use a different strategy.
    
    # Simpler: just try all candidates and if blocked, try removing one blocker
    # Rebuild counts for best_set
    cnt = [0] * 2001
    best_list = sorted(best_set)
    for i in range(len(best_list)):
        for j in range(i, len(best_list)):
            s = best_list[i] + best_list[j]
            if s <= 2000:
                cnt[s] += 1
    
    # For each candidate not in set, try to add it by removing one element
    for candidate in range(1, 1001):
        if elapsed() > DEADLINE:
            break
        if candidate in best_set:
            continue
        
        # Check which sums would be violated
        blocked_by = []
        for y in best_set:
            s = candidate + y
            if s <= 2000 and cnt[s] >= 2:
                blocked_by.append(y)
        s2 = 2 * candidate
        if s2 <= 2000 and cnt[s2] >= 2:
            # The 2*candidate sum is blocked, need to remove one element z where z+z=2*candidate or z+w=2*candidate
            # Actually if candidate is not in set, 2*candidate can only be formed by pairs in set
            blocked_by.append(None)  # special marker
        
        if not blocked_by:
            # Can add directly
            for y in best_list:
                cnt[candidate + y] += 1
            cnt[2 * candidate] += 1
            best_set.add(candidate)
            best_list.append(candidate)
            best_size += 1
        else:
            # Try removing one blocker and adding candidate
            for blocker in blocked_by:
                if elapsed() > DEADLINE:
                    break
                if blocker is None:
                    # Need to handle 2*candidate case specially
                    continue
                # Try removing blocker and adding candidate
                # Check if this is beneficial (it's a swap, same size)
                # Only do this if it enables future additions
                # For now, skip swaps that don't increase size
                pass
    
    return sorted(best_set)

construct()
