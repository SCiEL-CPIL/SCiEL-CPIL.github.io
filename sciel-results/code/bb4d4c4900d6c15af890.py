"""
Maximizing a Sidon set (distinct pairwise sums) in [1, 284] containing {1, 2, 5, 11}.

Approach:
1. Greedy construction using a bitmask for fast sum conflict detection.
2. Randomized Local Search (RLS) with:
   - Add moves: Try to add a random candidate.
   - Swap moves: Remove a non-prescribed element, try to add a random candidate.
   - Remove moves: Occasionally remove an element to escape local optima (less frequent).
3. Deadline guard to return best solution found within time budget.

Key insight: A Sidon set requires all pairwise sums a+b (a<=b) to be distinct.
We track used sums in a boolean array (or bitmask) for O(1) checks.
"""

import time
import random

# Deadline: 110s wall-clock (10s safety margin for 120s limit)
_DEADLINE = 110.0
_start_time = None
_MAX_SUM = 568 # 284 + 284

def _elapsed():
    return time.time() - _start_time

def _time_up():
    return _elapsed() > _DEADLINE

def _check_sidon(a):
    """Verify that set 'a' has all distinct pairwise sums."""
    a_sorted = sorted(a)
    seen = set()
    for i in range(len(a_sorted)):
        for j in range(i, len(a_sorted)):
            s = a_sorted[i] + a_sorted[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _init_sums(a):
    """Create a boolean array of used sums for set 'a'."""
    sums = [False] * (_MAX_SUM + 1)
    for i in range(len(a)):
        for j in range(i, len(a)):
            sums[a[i] + a[j]] = True
    return sums

def _try_add(a, x, sums):
    """
    Try to add x to Sidon set a.
    Returns True if successful, False otherwise.
    Modifies sums in place if successful.
    """
    # Check conflicts
    # We need to check if x + y is already in sums for any y in a
    # AND if x + y1 == x + y2 (which implies y1 == y2, so distinctness of a handles this)
    # Actually, we just need to ensure x+y is not in sums for any y in a, 
    # and that x+y1 != x+y2 for y1!=y2 in a is guaranteed by a being distinct.
    # Wait, the condition is that ALL pairwise sums are distinct.
    # If we add x, new sums are {x+y for y in a} U {2x}.
    # We need:
    # 1. 2x not in sums
    # 2. x+y not in sums for any y in a
    # 3. x+y1 != x+y2 for any y1 != y2 in a (True since a is distinct)
    # 4. x+y1 != 2x => y1 != x (True since x not in a)
    
    if sums[2*x]:
        return False
    
    for y in a:
        if sums[x + y]:
            return False
            
    # If we reach here, it's a valid addition
    # Update sums
    sums[2*x] = True
    for y in a:
        sums[x + y] = True
        
    return True

def _remove_and_update(a, x, sums):
    """
    Remove x from Sidon set a.
    Returns new sums array.
    """
    new_sums = [False] * (_MAX_SUM + 1)
    # We must recompute sums from scratch or carefully unmark.
    # Unmarking is tricky because a sum might be formed by multiple pairs.
    # Given N is small (~20), recomputing is fast enough.
    new_a = [y for y in a if y != x]
    for i in range(len(new_a)):
        for j in range(i, len(new_a)):
            new_sums[new_a[i] + new_a[j]] = True
    return new_sums

def construct():
    global _start_time
    _start_time = time.time()
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Step 1: Greedy Construction
    # Start with prescribed elements
    a = list(prescribed)
    sums = _init_sums(a)
    
    # Candidates to add
    candidates = list(range(12, 285))
    random.shuffle(candidates)
    
    for x in candidates:
        if _time_up():
            break
        if _try_add(a, x, sums):
            a.append(x)
            
    best_a = sorted(a)
    best_size = len(best_a)
    best_sums = sums
    
    # Verify initial solution
    if not _check_sidon(best_a):
        return [1, 2, 5, 11]

    # Step 2: Randomized Local Search
    current_a = best_a
    current_sums = best_sums
    
    # Precompute list of non-prescribed elements for swap/removal
    # We will maintain current_a as a list
    
    while not _time_up():
        # Choose a move type
        # 60% Add, 30% Swap, 10% Remove (to escape local optima)
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to Add
            # Pick a random candidate not in current_a
            # Optimization: Keep a list of available candidates
            available = [x for x in range(1, 285) if x not in current_a]
            if not available:
                continue
            x = random.choice(available)
            
            # Copy sums to check validity
            # Instead of copying the whole array, we can check first
            # Check if x can be added
            can_add = True
            if current_sums[2*x]:
                can_add = False
            else:
                for y in current_a:
                    if current_sums[x + y]:
                        can_add = False
                        break
            
            if can_add:
                # Perform addition
                new_a = current_a + [x]
                # Update sums in place? No, better to copy or update carefully.
                # Since we know it's valid, we can update current_sums in place
                # and keep a backup if needed, but RLS usually keeps best separately.
                # Let's update current_sums in place for speed, but we need to be able to revert.
                # Reverting requires knowing which sums were added.
                # Added sums: {2x} U {x+y for y in current_a}
                
                added_sums = [2*x] + [x+y for y in current_a]
                
                # Apply
                for s in added_sums:
                    current_sums[s] = True
                current_a.append(x)
                
                # Update best if improved
                if len(current_a) > best_size:
                    best_size = len(current_a)
                    best_a = list(current_a)
                    best_sums = list(current_sums)
                    
        elif move_type < 0.9:
            # Try to Swap
            # Remove a non-prescribed element, add a random one
            non_prescribed = [x for x in current_a if x not in prescribed]
            if not non_prescribed:
                continue
            
            old_x = random.choice(non_prescribed)
            
            # Remove old_x
            # We need to calculate new sums after removal
            # This is expensive if done naively.
            # Alternative: Try to find a new_x that fits in current_a \ {old_x}
            
            # Let's compute the "hole" sums.
            # Sums in current_a that involve old_x:
            # {old_x + y for y in current_a, y != old_x} U {2*old_x}
            
            # We want to add new_x such that:
            # 1. new_x + y is not in (current_sums \ sums_involving_old_x) for any y in current_a \ {old_x}
            # 2. 2*new_x is not in (current_sums \ sums_involving_old_x)
            # 3. new_x + y1 != new_x + y2 (guaranteed)
            # 4. new_x + y != 2*new_x => y != new_x (guaranteed if new_x not in current_a)
            
            # This logic is complex to check without creating a temporary sums array.
            # Given the small size, let's just create a temporary sums array for the check.
            
            temp_a = [y for y in current_a if y != old_x]
            temp_sums = _init_sums(temp_a)
            
            # Now try to add a random new_x
            available = [x for x in range(1, 285) if x not in temp_a and x != old_x]
            if not available:
                continue
            
            # Try a few random candidates
            found_swap = False
            for _ in range(10): # Try up to 10 random candidates
                if not available:
                    break
                new_x = random.choice(available)
                available.remove(new_x) # Avoid retrying same one in this inner loop
                
                if _try_add(temp_a, new_x, temp_sums):
                    # Valid swap
                    # Update current state
                    current_a = temp_a + [new_x]
                    current_sums = temp_sums
                    
                    if len(current_a) > best_size:
                        best_size = len(current_a)
                        best_a = list(current_a)
                        best_sums = list(current_sums)
                    
                    found_swap = True
                    break
            
            # If no swap found, current_a/current_sums remain unchanged
            # (We didn't modify them in place during the check)
            
        else:
            # Remove a non-prescribed element to escape local optima
            non_prescribed = [x for x in current_a if x not in prescribed]
            if non_prescribed:
                old_x = random.choice(non_prescribed)
                current_a = [y for y in current_a if y != old_x]
                current_sums = _init_sums(current_a)

    # Final verification and cleanup
    if not _check_sidon(best_a):
        return [1, 2, 5, 11]
    
    # Ensure prescribed elements are present
    for p in prescribed:
        if p not in best_a:
            best_a.append(p)
            best_a.sort()
            # Re-verify
            if not _check_sidon(best_a):
                 return [1, 2, 5, 11]

    return sorted(best_a)
