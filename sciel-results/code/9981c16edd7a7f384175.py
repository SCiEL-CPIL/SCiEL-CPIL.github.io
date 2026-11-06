import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    n_all = len(all_triples)
    
    # Helper to get sorted tuple
    def s_tup(t):
        return tuple(sorted(t))

    # Precompute for each triple, the list of other 7 triples that form a K(2,2,2) with it.
    # This is expensive to store fully, so we will compute conflicts on the fly or use a faster check.
    # Given the small size (16 points), we can afford a slightly heavier check if it allows better search.
    
    # Let's stick to the fast check from the parent but integrate it into a local search loop.
    
    def check_violation(triple, pair_masks):
        """
        Returns a list of triples that, if removed, would allow 'triple' to be added without violation.
        Actually, it's easier to just return True/False for validity, and for local search, 
        we need to know WHICH triples are blocking.
        
        For local search, we can do this:
        1. Try to add T.
        2. If violation, find one violating configuration (6 points).
        3. The 8 transversals in that config are all present.
        4. To add T, we must remove at least one of the other 7.
        5. We pick one of the other 7 to remove.
        """
        x, y, z = triple
        # We need to find if there exists x', y', z' such that all 8 are present.
        # If found, we return the 7 other triples.
        
        for x_prime in range(16):
            if x_prime in (x, y, z): continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime): continue
                # Check 3 transversals involving x', y', z fixed or mixed
                # We need to check if (x',y,z), (x,y',z), (x',y',z) are present.
                a, b, c = sorted((x_prime, y, z))
                if not (pair_masks[a][b] >> c) & 1: continue
                a, b, c = sorted((x, y_prime, z))
                if not (pair_masks[a][b] >> c) & 1: continue
                a, b, c = sorted((x_prime, y_prime, z))
                if not (pair_masks[a][b] >> c) & 1: continue
                
                # Now check for z_prime
                for z_prime in range(16):
                    if z_prime in (x, y, z, x_prime, y_prime): continue
                    # Check remaining 4 transversals
                    a, b, c = sorted((x, y, z_prime))
                    if not (pair_masks[a][b] >> c) & 1: continue
                    a, b, c = sorted((x_prime, y, z_prime))
                    if not (pair_masks[a][b] >> c) & 1: continue
                    a, b, c = sorted((x, y_prime, z_prime))
                    if not (pair_masks[a][b] >> c) & 1: continue
                    a, b, c = sorted((x_prime, y_prime, z_prime))
                    if not (pair_masks[a][b] >> c) & 1: continue
                    
                    # Found a conflict. The 7 other triples are:
                    return [
                        (x_prime, y, z),
                        (x, y_prime, z),
                        (x_prime, y_prime, z),
                        (x, y, z_prime),
                        (x_prime, y, z_prime),
                        (x, y_prime, z_prime),
                        (x_prime, y_prime, z_prime)
                    ]
        return None

    def add_triple(triple, system_set, pair_masks):
        x, y, z = triple
        a, b, c = sorted((x, y, z))
        pair_masks[a][b] |= (1 << c)
        system_set.add(tuple(sorted((x, y, z))))

    def remove_triple(triple, system_set, pair_masks):
        x, y, z = triple
        a, b, c = sorted((x, y, z))
        pair_masks[a][b] &= ~(1 << c)
        system_set.remove(tuple(sorted((x, y, z))))

    # Phase 1: Greedy Initialization
    best_system_set = None
    best_pair_masks = None
    best_size = 0
    
    for attempt in range(10):
        if time.time() > deadline - 30: break # Leave time for local search
        
        pair_masks = [[0] * 16 for _ in range(16)]
        system_set = set()
        
        order = list(all_triples)
        random.shuffle(order)
        
        for t in order:
            if check_violation(t, pair_masks) is None:
                add_triple(t, system_set, pair_masks)
        
        if len(system_set) > best_size:
            best_size = len(system_set)
            best_system_set = set(system_set)
            # Deep copy pair_masks
            best_pair_masks = [row[:] for row in pair_masks]

    if best_system_set is None:
        # Fallback star
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

    # Phase 2: Local Search (Kicking)
    current_set = set(best_system_set)
    current_masks = [row[:] for row in best_pair_masks]
    current_size = len(current_set)
    
    remaining_triples = [t for t in all_triples if t not in current_set]
    random.shuffle(remaining_triples)
    remaining_idx = 0
    
    # We will iterate, trying to add triples from 'remaining' by removing blockers
    # We also need to track the best solution ever seen.
    
    while time.time() < deadline - 5:
        if remaining_idx >= len(remaining_triples):
            # Reset remaining list with new random order
            remaining_triples = [t for t in all_triples if t not in current_set]
            if not remaining_triples:
                break
            random.shuffle(remaining_triples)
            remaining_idx = 0
            
        t = remaining_triples[remaining_idx]
        remaining_idx += 1
        
        conflict_list = check_violation(t, current_masks)
        
        if conflict_list is None:
            # Can add directly
            add_triple(t, current_set, current_masks)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_system_set = set(current_set)
                best_pair_masks = [row[:] for row in current_masks]
        else:
            # Conflict exists. Pick a random blocker from the list to remove.
            # Removing one blocker might not be enough if there are multiple conflicts,
            # but let's try the simplest kick first.
            blocker = random.choice(conflict_list)
            
            # Only remove if blocker is actually in current_set (it should be)
            if blocker in current_set:
                remove_triple(blocker, current_set, current_masks)
                current_size -= 1
                
                # Now try to add t again
                # It might still conflict with other configurations, but let's try.
                # If it still conflicts, we might need to remove more.
                # For simplicity, we just add it if it's valid now, otherwise we skip this kick.
                if check_violation(t, current_masks) is None:
                    add_triple(t, current_set, current_masks)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_system_set = set(current_set)
                        best_pair_masks = [row[:] for row in current_masks]
                else:
                    # Still invalid. We've lost a triple and didn't gain one.
                    # This is a "regression". We can either keep this state (if we hope to recover)
                    # or revert. Reverting is safer for maintaining quality, but less exploratory.
                    # Let's revert to avoid getting stuck in bad states.
                    add_triple(blocker, current_set, current_masks)
                    current_size += 1
                    # Note: 't' is still not in the set.
            
            # If we reverted, we just move to the next candidate.
            # If we didn't revert (i.e. we successfully added t after removing blocker), we move on.

    # Return the best found
    return [list(t) for t in best_system_set]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
