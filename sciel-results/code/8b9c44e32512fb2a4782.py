import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    all_vals = list(range(1, 285))
    
    # Best found so far
    best_set = set(prescribed)
    best_size = len(best_set)

    # Helper functions
    def build_sums(s):
        """Build the set of all pairwise sums a+b for a<=b in s."""
        sl = sorted(s)
        sums = set()
        for i in range(len(sl)):
            xi = sl[i]
            for j in range(i, len(sl)):
                sums.add(xi + sl[j])
        return sums

    def can_add(x, s, sums):
        """Check if x can be added to s given the current sums set."""
        # x must not be in s
        if x in s:
            return False
        # Check new sums
        # New sums are {x + y for y in s} U {2x}
        # They must not intersect with current sums
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_sums(x, s, sums):
        """Add sums involving x to the sums set. s must contain x."""
        for y in s:
            sums.add(x + y)
        # Note: 2x is added when y=x in the loop if x is in s
        # But if we call this after adding x to s, it works.
        # If we call this before adding x to s, we need to handle 2x separately or ensure x is in s.
        # Let's assume x is already in s.

    def remove_sums(x, s, sums):
        """Remove sums involving x from the sums set. s must contain x."""
        for y in s:
            sums.discard(x + y)

    # Strategy 1: Greedy Construction
    def greedy(order, time_limit=2.0):
        s = set(prescribed)
        sums = build_sums(s)
        start = time.time()
        for x in order:
            if time.time() - start > time_limit:
                break
            if can_add(x, s, sums):
                s.add(x)
                add_sums(x, s, sums)
        return s, sums

    rng = random.Random(42)
    
    # Try several greedy orders
    # 1. Smallest first
    s, sums = greedy(range(12, 285), 1.0)
    if len(s) > best_size:
        best_set, best_size = set(s), len(s)
    
    # 2. Largest first
    s, sums = greedy(range(284, 11, -1), 1.0)
    if len(s) > best_size:
        best_set, best_size = set(s), len(s)
    
    # 3. Random shuffles
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(12, 285))
        rng.shuffle(order)
        s, sums = greedy(order, 0.5)
        if len(s) > best_size:
            best_set, best_size = set(s), len(s)

    # Strategy 2: Local Search with Swaps
    def local_search(start_s, time_limit=60.0):
        nonlocal best_set, best_size
        s = set(start_s)
        sums = build_sums(s)
        start = time.time()
        
        # Precompute candidates
        candidates = [x for x in all_vals if x not in s]
        
        while time.time() - start < time_limit and time.time() < deadline - 5:
            # Phase A: Try to add any element
            added = False
            # Shuffle candidates to avoid bias? Or just iterate?
            # Iterating is O(N). N=280. Fast.
            # We can iterate through a shuffled list of candidates
            cand_list = candidates[:]
            rng.shuffle(cand_list)
            
            for x in cand_list:
                if can_add(x, s, sums):
                    s.add(x)
                    add_sums(x, s, sums)
                    candidates.remove(x)
                    if len(s) > best_size:
                        best_set, best_size = set(s), len(s)
                    added = True
                    break
            
            if added:
                continue
            
            # Phase B: Stuck. Perform a swap.
            # Remove a non-prescribed element, add a new one.
            removable = [x for x in s if x not in prescribed_set]
            if not removable:
                break # Cannot remove prescribed, and no add possible. Stuck.
            
            # Pick a random element to remove
            u = rng.choice(removable)
            
            # Temporarily remove u
            s.remove(u)
            remove_sums(u, s | {u}, sums) # Pass s with u to remove sums, or handle carefully.
            # Let's be precise:
            # remove_sums iterates y in s. If s contains u, it removes u+y.
            # So we should call remove_sums(u, s_with_u, sums) before removing u from s?
            # Or remove u from s, then remove sums?
            # If we remove u from s first, we can't iterate s to find u's sums if we don't keep a copy.
            # Let's keep a copy of s before removal for sum calculation, or just calculate sums to remove.
            
            # Let's redo the swap logic more carefully to avoid state errors.
            # Revert the temporary removal for now and do it properly in a block.
            s.add(u)
            add_sums(u, s, sums) # Restore sums if we messed up. 
            # Actually, let's just implement a clean swap function.
            
            def try_swap(u, s, sums, candidates):
                # s contains u.
                # 1. Remove u from s and sums
                old_s = set(s)
                s.remove(u)
                # Remove sums involving u from old_s
                for y in old_s:
                    sums.discard(u + y)
                
                # 2. Try to add a candidate v
                # We want to find a v that fits in s (which is now s \ {u})
                # To be efficient, we can just try random candidates.
                # Or we can pick a candidate that has fewer conflicts.
                
                # Let's try a few random candidates
                for _ in range(20):
                    if not candidates:
                        break
                    v = rng.choice(candidates)
                    # Check if v can be added to s
                    valid = True
                    for y in s:
                        if v + y in sums:
                            valid = False
                            break
                    if valid and 2*v not in sums:
                        # Accept swap
                        s.add(v)
                        # Add sums for v
                        for y in s:
                            sums.add(v + y)
                        candidates.remove(v)
                        candidates.append(u) # u becomes a candidate again
                        return True
                # Revert
                s.add(u)
                for y in s:
                    sums.add(u + y)
                return False

            if not try_swap(u, s, sums, candidates):
                # If swap fails, try another u? Or break?
                # Let's try a few swaps before giving up on this configuration
                # If we fail repeatedly, we might be in a dead end.
                # For simplicity, if one swap fails, we just continue the loop 
                # and pick another u next time.
                pass

    # Run local search from the best greedy result
    if best_size > len(prescribed):
        local_search(best_set, 50.0)
        
    # Final polish: try to add more elements if time remains
    if time.time() < deadline - 10:
        s = set(best_set)
        sums = build_sums(s)
        candidates = [x for x in all_vals if x not in s]
        rng.shuffle(candidates)
        for x in candidates:
            if can_add(x, s, sums):
                s.add(x)
                add_sums(x, s, sums)
        if len(s) > best_size:
            best_set, best_size = set(s), len(s)

    # Verify
    def is_valid_sidon(s):
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sm = sl[i] + sl[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True

    result = sorted(best_set)
    if not is_valid_sidon(result):
        return list(prescribed)
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
