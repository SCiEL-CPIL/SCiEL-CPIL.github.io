import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _find_witnesses(a, b, c, presence, max_witnesses=10):
    """Find up to max_witnesses K(2,2,2)s that block triple (a,b,c)."""
    witnesses = []
    excluded = {a, b, c}
    
    for a_p in range(V):
        if a_p in excluded:
            continue
        if not _has_presence(presence, _tidx(a_p, b, c)):
            continue
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b_p, c)):
                continue
            if not _has_presence(presence, _tidx(a_p, b_p, c)):
                continue
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                if not _has_presence(presence, _tidx(a, b, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a, b_p, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b_p, c_p)):
                    continue
                
                # Found a witness
                witness = [
                    _tidx(a_p, b, c),
                    _tidx(a, b_p, c),
                    _tidx(a_p, b_p, c),
                    _tidx(a, b, c_p),
                    _tidx(a_p, b, c_p),
                    _tidx(a, b_p, c_p),
                    _tidx(a_p, b_p, c_p)
                ]
                witnesses.append(witness)
                if len(witnesses) >= max_witnesses:
                    return witnesses
    return witnesses

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        # Quick check: if any part of a potential K222 is missing, it's safe?
        # No, we need to check if adding it creates a K222.
        # We can use _find_witnesses with max_witnesses=1 to check conflict quickly.
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _local_search(presence, count, deadline):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    max_iters = 100
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # 1. Try to add any absent triple
        random.shuffle(absent_list)
        added = False
        for idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[idx]
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                absent_list.remove(idx)
                count += 1
                improved = True
                added = True
                break # Restart loop to re-evaluate
        
        if added:
            continue
        
        # 2. If no add possible, try swaps (remove 1, add 1)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        swapped = False
        for new_idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[new_idx]
            
            # Find witnesses for this new triple
            witnesses = _find_witnesses(a, b, c, presence, 10)
            if not witnesses:
                # Should have been added in step 1, but maybe race condition or logic error?
                # Or maybe it was addable but we missed it? 
                # Actually, if witnesses is empty, it is addable.
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                improved = True
                swapped = True
                break

            # Collect all candidate removals from these witnesses
            candidates = set()
            for w in witnesses:
                for t in w:
                    candidates.add(t)
            
            # Try removing each candidate
            for t_rem in candidates:
                if t_rem not in present_set:
                    continue
                
                # Temporarily remove t_rem
                presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                present_set.remove(t_rem)
                
                # Check if new_idx is now addable
                # We need to check if there are *any* witnesses left
                # Since we only removed one, we can just re-check
                # But _find_witnesses is expensive.
                # Alternative: Check if the removed edge was in ALL witnesses.
                # If t_rem is in all witnesses, then all those witnesses are broken.
                # Are there other witnesses?
                # If we found max_witnesses=10, there might be more.
                # So we must re-check.
                remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                
                if not remaining_witnesses:
                    # Swap successful
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    # Count remains same
                    swapped = True
                    improved = True
                    break
                else:
                    # Restore t_rem
                    presence[t_rem >> 6] |= (1 << (t_rem & 63))
                    present_set.add(t_rem)
                    
            if swapped:
                break
        
        if not swapped:
            # No swap found in this pass
            # We could try a "kick" move: remove a random present triple to unblock things?
            # Or just break out of this iteration and let the outer loop restart with a new shuffle.
            # To avoid infinite loop if no improvement is possible, we break.
            # But maybe we should try a few more absent triples?
            # The loop `for new_idx in absent_list` already tries all (shuffled) absent triples.
            # If none worked, we are stuck.
            break

    return count, presence

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        
        # Local search
        count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
        if time.time() > deadline - 5:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
