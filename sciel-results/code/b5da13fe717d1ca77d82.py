import time
import random
import math
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

def _simulated_annealing(presence, count, deadline):
    """Simulated annealing with add, remove, and swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    best_count = count
    best_presence = presence[:]
    best_present_set = set(present_set)
    
    # Precompute for speed
    triples = ALL_TRIPLES
    
    while T > T_min and time.time() < deadline:
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Add move: try to add a random absent triple
            if not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = triples[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                # Accept
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
            else:
                # Reject (can't add if it creates a K222)
                pass
        elif move_type < 0.75:
            # Remove move: remove a random present triple
            if not present_set:
                T *= cooling_rate
                continue
            old_idx = random.choice(list(present_set))
            # Remove
            presence[old_idx >> 6] &= ~(1 << (old_idx & 63))
            present_set.remove(old_idx)
            absent_list.append(old_idx)
            count -= 1
            # This is always "accepted" in the sense that we allow it, but it worsens the objective
            # We'll rely on subsequent add moves to recover
        else:
            # Swap move: remove one, add one
            if not present_set or not absent_list:
                T *= cooling_rate
                continue
            new_idx = random.choice(absent_list)
            a, b, c = triples[new_idx]
            witnesses = _find_witnesses(a, b, c, presence, 10)
            if not witnesses:
                # Just add
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                if count > best_count:
                    best_count = count
                    best_presence = presence[:]
                    best_present_set = set(present_set)
            else:
                # Try swapping: remove a candidate from witnesses
                candidates = set()
                for w in witnesses:
                    for t in w:
                        candidates.add(t)
                candidates = [t for t in candidates if t in present_set]
                if candidates:
                    t_rem = random.choice(candidates)
                    # Remove t_rem
                    presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                    present_set.remove(t_rem)
                    # Check if new_idx is now addable
                    remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                    if not remaining_witnesses:
                        # Swap successful
                        presence[new_idx >> 6] |= (1 << (new_idx & 63))
                        present_set.add(new_idx)
                        absent_list.remove(new_idx)
                        # count stays the same
                        if count > best_count:
                            best_count = count
                            best_presence = presence[:]
                            best_present_set = set(present_set)
                    else:
                        # Restore t_rem
                        presence[t_rem >> 6] |= (1 << (t_rem & 63))
                        present_set.add(t_rem)
        
        T *= cooling_rate
    
    # Return best solution found during SA
    # Restore best
    for i in range(N_TRIPLES):
        bit = 1 << (i & 63)
        word = i >> 6
        if i in best_present_set:
            presence[word] |= bit
        else:
            presence[word] &= ~bit
    
    return best_count, presence

def _local_search(presence, count, deadline):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    max_iters = 50
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
        
        # 2. If no add possible, try swaps (remove 1, add 1)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        swapped = False
        for new_idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[new_idx]
            
            witnesses = _find_witnesses(a, b, c, presence, 10)
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
        
        if not swapped:
            break

    return count, presence

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        
        # Local search first
        count, presence = _local_search(presence, count, deadline)
        
        # Then simulated annealing
        if time.time() < deadline - 10:
            count, presence = _simulated_annealing(presence, count, deadline)
        
        # Final local search polish
        if time.time() < deadline - 5:
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
