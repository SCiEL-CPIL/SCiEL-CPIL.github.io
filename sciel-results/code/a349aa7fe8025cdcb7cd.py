import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 55  # safety margin before 60s (evaluator kills at 90s, but we want margin)
    
    N = 256
    # Precompute conflicts: conflict[i][j] = True if (i*j) % (i+j) == 0 for distinct i,j
    # Use 1-indexed: elements are 1..256
    
    # For speed, compute conflicts as a set of frozensets or bitmasks
    # conflict_mask[i] = bitmask of j such that i,j conflict
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def degree(mask):
        # Count bits in mask for elements 1..256
        return bin(mask & ((1 << (N+1)) - 2)).count('1')  # exclude bit 0
    
    def is_conflict_free(selected, x):
        # Check if x conflicts with any element in selected
        # selected is a set of ints
        for s in selected:
            if conflict_mask[x] & (1 << s):
                return False
        return True
    
    # Phase 1: Greedy - start with all, remove highest degree iteratively
    selected = set(range(1, N + 1))
    
    # Greedy removal: repeatedly remove the element with most conflicts within selected
    while True:
        if time.time() > deadline - 5:
            break
        # Find element with max conflicts within selected
        best_elem = None
        best_count = -1
        for e in selected:
            cnt = degree(conflict_mask[e] & 0)  # need to count only within selected
            # Better: count conflicts within selected
            cnt = 0
            for s in selected:
                if s != e and (conflict_mask[e] & (1 << s)):
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_elem = e
        if best_count <= 0:
            break
        selected.remove(best_elem)
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Phase 2: Local search - try to add back removed elements
    removed = set(range(1, N + 1)) - selected
    
    # Try adding back each removed element
    for _ in range(50):  # multiple passes
        if time.time() > deadline - 5:
            break
        improved = False
        for r in list(removed):
            if is_conflict_free(selected, r):
                selected.add(r)
                removed.discard(r)
                improved = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
        if not improved:
            break
    
    # Phase 3: Local search - try swaps (remove one from selected, add one from removed)
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        improved = False
        # Try: for each removed element r, for each selected element s that conflicts with r,
        # remove s and add r if it doesn't conflict with others
        for r in list(removed):
            if time.time() > deadline - 5:
                break
            # Find which selected elements conflict with r
            conflicting = [s for s in selected if conflict_mask[r] & (1 << s)]
            if len(conflicting) == 0:
                # Can add directly
                selected.add(r)
                removed.discard(r)
                improved = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
                continue
            # Try removing one conflicting element and adding r
            for s in conflicting:
                # Check if r conflicts with any other selected element (besides s)
                ok = True
                for other in selected:
                    if other != s and (conflict_mask[r] & (1 << other)):
                        ok = False
                        break
                if ok:
                    selected.discard(s)
                    selected.add(r)
                    removed.discard(r)
                    removed.add(s)
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = set(selected)
                    break
        if not improved:
            break
    
    # Phase 4: Randomized local search with restarts
    for restart in range(20):
        if time.time() > deadline - 3:
            break
        
        # Start from best_set or random perturbation
        if restart == 0:
            cur = set(best_set)
        else:
            cur = set(best_set)
            # Remove k random elements
            k = random.randint(1, min(5, len(cur)))
            to_remove = random.sample(list(cur), k)
            for x in to_remove:
                cur.discard(x)
        
        removed_cur = set(range(1, N + 1)) - cur
        
        # Try to add back
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            improved = False
            for r in list(removed_cur):
                if is_conflict_free(cur, r):
                    cur.add(r)
                    removed_cur.discard(r)
                    improved = True
            if not improved:
                break
        
        # Try swaps
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            improved = False
            for r in list(removed_cur):
                conflicting = [s for s in cur if conflict_mask[r] & (1 << s)]
                if len(conflicting) == 0:
                    cur.add(r)
                    removed_cur.discard(r)
                    improved = True
                    continue
                for s in conflicting:
                    ok = True
                    for other in cur:
                        if other != s and (conflict_mask[r] & (1 << other)):
                            ok = False
                            break
                    if ok:
                        cur.discard(s)
                        cur.add(r)
                        removed_cur.discard(r)
                        removed_cur.add(s)
                        improved = True
                        break
            if not improved:
                break
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = set(cur)
    
    return list(best_set)

# EVOLVE-BLOCK-END
