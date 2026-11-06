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
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _local_search(presence, count, deadline):
    """Local search with Add, Swap, and 2-Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    max_iters = 200
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
                break
        
        if added:
            continue
        
        # 2. Try single swaps (remove 1, add 1)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        swapped = False
        for new_idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[new_idx]
            
            witnesses = _find_witnesses(a, b, c, presence, 20)
            if not witnesses:
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                improved = True
                swapped = True
                break

            candidates = set()
            for w in witnesses:
                for t in w:
                    candidates.add(t)
            
            for t_rem in candidates:
                if t_rem not in present_set:
                    continue
                
                presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                present_set.remove(t_rem)
                
                remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                
                if not remaining_witnesses:
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    swapped = True
                    improved = True
                    break
                else:
                    presence[t_rem >> 6] |= (1 << (t_rem & 63))
                    present_set.add(t_rem)
                    
            if swapped:
                break
        
        if swapped:
            continue
        
        # 3. Try 2-swaps (remove 2, add 1) - net loss of 1, but might unblock future adds
        # Actually, let's try 2-remove 1-add only if it leads to a net gain later, 
        # but that's hard to predict. Let's try 1-remove 2-add (net gain +1).
        # Or 2-remove 2-add (net 0, but changes structure).
        
        # Let's try 2-remove 1-add to break complex blocks, hoping to add 2 later.
        # This is a "kick" move.
        kicked = False
        if not swapped:
            random.shuffle(absent_list)
            for new_idx in absent_list[:20]:  # Limit to speed up
                if time.time() >= deadline:
                    return count, presence
                a, b, c = ALL_TRIPLES[new_idx]
                witnesses = _find_witnesses(a, b, c, presence, 10)
                if not witnesses:
                    continue
                
                candidates = set()
                for w in witnesses:
                    for t in w:
                        candidates.add(t)
                
                # Try removing 2 candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                for i_rem in range(min(len(cand_list), 30)):
                    for j_rem in range(i_rem + 1, min(len(cand_list), 30)):
                        t1 = cand_list[i_rem]
                        t2 = cand_list[j_rem]
                        if t1 not in present_set or t2 not in present_set:
                            continue
                        
                        # Remove both
                        presence[t1 >> 6] &= ~(1 << (t1 & 63))
                        present_set.remove(t1)
                        presence[t2 >> 6] &= ~(1 << (t2 & 63))
                        present_set.remove(t2)
                        
                        # Check if new_idx is addable
                        remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                        
                        if not remaining_witnesses:
                            # Add new_idx
                            presence[new_idx >> 6] |= (1 << (new_idx & 63))
                            present_set.add(new_idx)
                            absent_list.remove(new_idx)
                            # Count decreases by 1
                            count -= 1
                            kicked = True
                            improved = True
                            break
                        else:
                            # Restore
                            presence[t1 >> 6] |= (1 << (t1 & 63))
                            present_set.add(t1)
                            presence[t2 >> 6] |= (1 << (t2 & 63))
                            present_set.add(t2)
                        
                        if kicked:
                            break
                    if kicked:
                        break
                if kicked:
                    break
            
            if kicked:
                continue

        if not swapped and not kicked:
            break

    return count, presence

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        
        count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
        if time.time() > deadline - 10:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
