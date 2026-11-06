"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            if new_sums & sums:
                continue
            
            extendable.append(x)
        return extendable

    # Strategy: "Dominating Set" Greedy with Backtracking Repair
    # 1. Greedily build a set S where each new element is chosen to "block" (make non-extendable)
    #    the maximum number of currently unblocked candidates.
    # 2. If S is maximal and Sidon, we are done (or we try to shrink it).
    # 3. If S is not maximal (some x is extendable), we have a "hole".
    #    To fix it, we must add an element y that blocks x.
    #    We choose y to minimize the increase in size (ideally 0, but adding is required for maximality if greedy missed it).
    #    Actually, if the greedy logic was perfect, S would be maximal.
    #    The "error" in greedy is that it might pick an element that blocks many *currently* alive elements,
    #    but fails to block some that become alive later or were not considered optimally.
    #    
    #    A better structural approach:
    #    1. Generate a random Sidon set S using a simple greedy or random insertion.
    #    2. Check for "holes" (extendable elements).
    #    3. For each hole x, find a "witness" pair (a,b) in S such that a+b = 2x or x+a = b+c...
    #       Wait, x is extendable if NO such collision exists.
    #       So to block x, we need to ADD an element y such that y creates a collision with x.
    #       Collision types:
    #       - 2y = x + s (for some s in S)
    #       - y + x = s + t (for some s,t in S)
    #       - y + s = x + t
    #       - 2x = y + s
    #       - x + s = y + t
    #       
    #    This is complex. Let's stick to the "Greedy Max-Blocking" but with a more robust "alive" set maintenance.
    #    The previous attempt had bugs in the blocking logic. Let's implement a cleaner version.
    #    
    #    Algorithm:
    #    1. S = []
    #    2. Alive = {1..N}
    #    3. While Alive is not empty:
    #       a. For each x in Alive:
    #          - Check if x is valid (no collision with S). If not, remove from Alive (shouldn't happen if maintained correctly).
    #          - Compute "Block(x)": The set of y in Alive\{x} such that y would become INVALID if x were added to S.
    #            y becomes invalid if:
    #            - 2y collides with new sums from x (2x or x+s)
    #            - y+s collides with new sums from x
    #            - x+y collides with existing sums
    #       b. Pick x with largest |Block(x)|.
    #       c. Add x to S.
    #       d. Update Alive: Remove x and all y in Block(x).
    #    
    #    This guarantees S is Sidon (by construction) and Maximal (Alive is empty at end).
    #    The size is minimized by the greedy choice of max blocking.

    def greedy_max_blocking():
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        
        while alive:
            best_x = None
            max_blocked = -1
            best_blocked_set = set()
            
            # To speed up, we can limit the candidates if alive is large, but N=256 is small.
            # We iterate over all alive to find the best.
            for x in alive:
                # Check if x is still valid against S
                # New sums if x added: 2x, x+s for s in S
                new_sums_x = {2 * x}
                new_sums_x.update(x + s for s in S)
                
                if new_sums_x & Sums:
                    # x is invalid, should have been removed. Skip.
                    continue
                
                # Calculate which y in alive (y != x) are blocked by x
                blocked = set()
                for y in alive:
                    if y == x:
                        continue
                    
                    # y is blocked if it cannot be added to S ∪ {x}
                    # This happens if any sum involving y collides with Sums or new_sums_x
                    
                    # 1. 2y collision
                    if 2 * y in Sums or 2 * y in new_sums_x:
                        blocked.add(y)
                        continue
                    
                    # 2. y+s collision
                    # y+s must not be in Sums or new_sums_x for any s in S ∪ {x}
                    # Note: If y is in alive, y+s is NOT in Sums for s in S (by invariant).
                    # So we only need to check:
                    # a) y+x in Sums?
                    # b) y+s in new_sums_x? (where s in S)
                    
                    if (y + x) in Sums:
                        blocked.add(y)
                        continue
                        
                    # Check y+s in new_sums_x for s in S
                    # new_sums_x = {2x} ∪ {x+s' for s' in S}
                    # So we check if y+s == 2x  OR  y+s == x+s'
                    
                    # y+s == 2x  => s == 2x - y
                    if (2 * x - y) in S:
                        blocked.add(y)
                        continue
                        
                    # y+s == x+s' => s' == y + s - x
                    # We need to check if there exists s in S such that (y+s-x) is in S.
                    # Let d = y - x. We need s in S such that s+d in S.
                    # This is a difference check.
                    # For small S, a loop is fine.
                    blocked_y = False
                    for s in S:
                        if (y + s - x) in S:
                            blocked_y = True
                            break
                    if blocked_y:
                        blocked.add(y)
                        continue

                if len(blocked) > max_blocked:
                    max_blocked = len(blocked)
                    best_x = x
                    best_blocked_set = blocked
            
            if best_x is None:
                # Fallback: pick any valid x
                # Find first valid x
                for x in alive:
                    new_sums_x = {2 * x}
                    new_sums_x.update(x + s for s in S)
                    if not (new_sums_x & Sums):
                        best_x = x
                        break
                if best_x is None:
                    # No valid x? This means S is maximal but alive is not empty?
                    # This implies a bug in alive maintenance.
                    # Break to avoid infinite loop.
                    break

            # Add best_x to S
            S.append(best_x)
            new_sums_best = {2 * best_x}
            new_sums_best.update(best_x + s for s in S[:-1])
            Sums.update(new_sums_best)
            
            # Update alive
            alive.remove(best_x)
            for y in best_blocked_set:
                if y in alive:
                    alive.remove(y)

        return S

    # Run the greedy algorithm multiple times with different random seeds/orderings?
    # The greedy is deterministic based on the "max blocked" tie-breaking.
    # We can randomize the iteration order of `alive` to break ties or explore different paths.
    
    best_set = None
    best_size = N + 1
    
    # Try a few random restarts with shuffled initial order or tie-breaking
    for _ in range(50): # Limit restarts to ensure time
        if time.time() > deadline - 5:
            break
        
        # To make it non-deterministic, we can shuffle the list of alive candidates
        # and pick the first one with max blocked, or just shuffle and let the loop handle it.
        # Actually, the loop `for x in alive` iterates in set order (arbitrary).
        # We can force a specific order by converting to a shuffled list.
        
        # Let's modify greedy to accept a random seed for tie-breaking
        # Or just run the standard greedy. It's deterministic.
        # To get variety, we can randomly remove some "safe" elements? No, that breaks maximality.
        
        # Let's just run the deterministic greedy. If it's good, it's good.
        # If not, we need a different tie-breaking.
        # Let's implement a version that shuffles the candidates before evaluation.
        
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        
        while alive:
            # Shuffle candidates to break ties and explore different branches
            candidates = list(alive)
            random.shuffle(candidates)
            
            best_x = None
            max_blocked = -1
            best_blocked_set = set()
            
            for x in candidates:
                new_sums_x = {2 * x}
                new_sums_x.update(x + s for s in S)
                
                if new_sums_x & Sums:
                    continue
                
                blocked = set()
                for y in candidates:
                    if y == x:
                        continue
                    
                    if 2 * y in Sums or 2 * y in new_sums_x:
                        blocked.add(y)
                        continue
                    
                    if (y + x) in Sums:
                        blocked.add(y)
                        continue
                        
                    if (2 * x - y) in S:
                        blocked.add(y)
                        continue
                        
                    blocked_y = False
                    for s in S:
                        if (y + s - x) in S:
                            blocked_y = True
                            break
                    if blocked_y:
                        blocked.add(y)
                        continue

                if len(blocked) > max_blocked:
                    max_blocked = len(blocked)
                    best_x = x
                    best_blocked_set = blocked
            
            if best_x is None:
                for x in candidates:
                    new_sums_x = {2 * x}
                    new_sums_x.update(x + s for s in S)
                    if not (new_sums_x & Sums):
                        best_x = x
                        break
                if best_x is None:
                    break

            S.append(best_x)
            new_sums_best = {2 * best_x}
            new_sums_best.update(best_x + s for s in S[:-1])
            Sums.update(new_sums_best)
            
            alive.remove(best_x)
            for y in best_blocked_set:
                if y in alive:
                    alive.remove(y)
            
            if time.time() > deadline - 2:
                break

        if S:
            # Verify
            if is_sidon(S):
                sums = get_sums(S)
                if not get_extendable(S, sums):
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = S
                else:
                    # Not maximal? Try to repair by adding elements?
                    # Or discard.
                    pass

    if best_set is None:
        # Fallback: Simple greedy Sidon (maximal by construction if we remove blocked)
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        while alive:
            # Pick smallest available
            x = min(alive)
            new_sums_x = {2 * x}
            new_sums_x.update(x + s for s in S)
            if new_sums_x & Sums:
                alive.remove(x)
                continue
            
            S.append(x)
            Sums.update(new_sums_x)
            alive.remove(x)
            
            # Remove blocked
            to_remove = []
            for y in alive:
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
            for y in to_remove:
                alive.remove(y)
        
        best_set = S

    # Final check and potential local search improvement
    # Try to remove elements and see if we can re-add a different set of fewer elements to cover the same "blocking"?
    # This is hard. Let's just return the best found.
    
    # One more check: if best_set is not maximal, we must fix it.
    # The greedy algorithm above ensures maximality by emptying `alive`.
    # So best_set should be maximal.
    
    return best_set

# EVOLVE-BLOCK-END
