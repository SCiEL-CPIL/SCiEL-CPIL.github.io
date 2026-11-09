import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Data structures
    A = []
    A_set = set()
    counts = {}
    
    def can_add(x):
        # Check if x can be added to current A
        # Sums: x + a for a in A, and 2x
        # If any sum has count >= 2, return False
        for a in A:
            s = x + a
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2*x, 0) >= 2:
            return False
        return True

    def add(x):
        A.append(x)
        A_set.add(x)
        A.sort()
        for a in A:
            if a == x: continue
            s = x + a
            counts[s] = counts.get(s, 0) + 1
        s2 = 2*x
        counts[s2] = counts.get(s2, 0) + 1

    def remove(x):
        A.remove(x)
        A_set.remove(x)
        A.sort()
        for a in A:
            if a == x: continue
            s = x + a
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s2 = 2*x
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]

    def get_blockers(x):
        # Returns set of elements in A that must be removed to make x addable
        # Or None if x is addable
        if can_add(x):
            return None
            
        blockers = set()
        # Find conflict sums
        conflict_sums = set()
        for a in A:
            s = x + a
            if counts.get(s, 0) >= 2:
                conflict_sums.add(s)
        s2 = 2*x
        if counts.get(s2, 0) >= 2:
            conflict_sums.add(s2)
            
        # Find elements in A participating in these sums
        for s in conflict_sums:
            for u in A:
                v = s - u
                if v in A_set:
                    blockers.add(u)
        return blockers

    # 1. Greedy Initialization
    for x in range(1, 1001):
        if can_add(x):
            add(x)

    # 2. Local Search
    # We iterate until deadline
    while time.time() < deadline:
        # Try to add any candidate
        added = False
        # To avoid always trying the same small numbers first,
        # we can shuffle the candidate list occasionally.
        # But greedy order (small to large) is often good for packing.
        # Let's stick to sorted order for determinism in the "add" phase,
        # but use randomization in the "swap" phase.
        
        # Check all candidates for direct add
        # Optimization: Only check candidates that were previously blocked?
        # For simplicity, check all.
        
        # We can speed up by maintaining a list of "active" candidates.
        # But 1000 is small.
        
        for b in range(1, 1001):
            if b in A_set:
                continue
            if can_add(b):
                add(b)
                added = True
                break # Restart loop to re-evaluate
                
        if added:
            continue
            
        # No direct add possible. Try Swap-to-Add.
        
        # Find the "best" blocked candidate
        # Best = fewest blockers
        best_b = None
        best_blockers = None
        min_blockers = float('inf')
        
        # To save time, we can sample candidates if N is large, but N=1000 is fine.
        for b in range(1, 1001):
            if b in A_set:
                continue
            blockers = get_blockers(b)
            if blockers is None:
                # Should have been added
                add(b)
                break
            if len(blockers) < min_blockers:
                min_blockers = len(blockers)
                best_b = b
                best_blockers = blockers
                
        if best_b is None:
            # No candidates to try?
            break
            
        # Try to swap a blocker with a candidate
        swap_done = False
        blocker_list = list(best_blockers)
        random.shuffle(blocker_list)
        
        # Limit the number of swap attempts to avoid wasting time on one b
        attempts = 0
        max_attempts = 500 # Arbitrary limit per b
        
        for a in blocker_list:
            if attempts > max_attempts:
                break
                
            # Try to find a c to swap with a
            # c must not be in A, c != best_b
            # We can shuffle candidates to try different ones
            # But we want c to be valid.
            
            # Heuristic: Try c that are "similar" to a? Or random?
            # Random is safer to explore.
            
            # To speed up, we can pre-check c against a subset of constraints?
            # Just try a subset of candidates.
            
            # Pick 50 random candidates to try
            candidates = [c for c in range(1, 1001) if c not in A_set and c != best_b]
            if not candidates:
                break
            random.shuffle(candidates)
            candidates_to_try = candidates[:50]
            
            for c in candidates_to_try:
                attempts += 1
                # Save state
                old_A = A.copy()
                old_A_set = A_set.copy()
                old_counts = counts.copy()
                
                # Remove a
                remove(a)
                
                # Try add c
                if can_add(c):
                    add(c)
                    
                    # Try add best_b
                    if can_add(best_b):
                        add(best_b)
                        swap_done = True
                        break
                    else:
                        # Rollback c
                        remove(c)
                else:
                    # Rollback a
                    pass # a is removed, need to re-add
                
                # If not successful, restore a
                if not swap_done:
                    add(a)
                    
            if swap_done:
                break
                
        if not swap_done:
            # No swap worked for best_b.
            # To prevent getting stuck in a local minimum,
            # perform a random valid swap (size-preserving) to change the landscape.
            if random.random() < 0.2:
                # Pick random a in A
                a = random.choice(A)
                # Pick random c not in A
                candidates = [c for c in range(1, 1001) if c not in A_set]
                if candidates:
                    c = random.choice(candidates)
                    # Check if swap is valid
                    old_A = A.copy()
                    old_A_set = A_set.copy()
                    old_counts = counts.copy()
                    
                    remove(a)
                    if can_add(c):
                        add(c)
                        # Swap successful (size unchanged)
                    else:
                        # Rollback
                        add(a)
            
    return A
