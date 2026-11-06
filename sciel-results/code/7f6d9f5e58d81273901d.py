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

# Precompute a 3D array for _tidx to make it O(1) without sorting
_tidx_lookup = [[[0]*V for _ in range(V)] for _ in range(V)]
for a in range(V):
    for b in range(V):
        if b == a: continue
        for c in range(V):
            if c == a or c == b: continue
            _tidx_lookup[a][b][c] = _tidx(a, b, c)

def _get_idx(a, b, c):
    return _tidx_lookup[a][b][c]

# Precompute conflict bitmasks: for each triple i, conflict_mask[i] is a bitmask
# of all triples j such that if all triples in some K(2,2,2) witness for i are present,
# then i cannot be added. Actually, we need: for triple i, which triples j are "in conflict"
# meaning j is part of some K(2,2,2) that would block i.
# More precisely: i is blocked if there exist a',b',c' such that the 7 other transversals
# are all present. So i conflicts with any triple that appears in such a witness.
# But actually for efficient checking, we want: given current presence, can we add i?
# i can be added iff there is NO witness (a',b',c') such that all 7 other transversals are present.
# 
# Alternative: precompute for each triple i, the set of "witness configurations" and for each
# witness, the 7 triples that need to be present. Then i is blocked if ANY witness has all 7 present.
# This is still expensive to check dynamically.
#
# Better approach: For each pair of triples (i, j), determine if they can coexist in a way that
# helps form a K(2,2,2). Actually, let's think differently.
#
# Key insight: i is blocked iff ∃ a',b',c' (distinct from i's points and each other) such that
# all 7 transversals (other than i itself) are in the current set.
#
# For efficiency, precompute for each triple i, a list of "witnesses" where each witness is
# a frozenset of 7 triple indices. Then checking if i is blocked = checking if any witness's
# 7 triples are all present.
#
# Even better: For each triple i, precompute the conflict structure. When we add/remove a triple,
# we can update which triples become blocked/unblocked. But this is complex.
#
# Let's use a different strategy: precompute for each triple i, a list of "critical sets" -
# minimal sets of triples whose presence blocks i. Then use bit operations to check.
#
# Actually, the most practical approach: precompute the full conflict graph where edge (i,j) 
# means j is in some minimal blocking set for i. Then use a smarter search.
#
# For now, let's precompute witness structures for each triple to make _find_witnesses faster.

# For each triple, precompute all possible witnesses (a',b',c' combinations)
# and the 7 triple indices for each witness
_witness_data = {}  # idx -> list of (witness_triples_list, (a_p, b_p, c_p))
for _i, (a, b, c) in enumerate(ALL_TRIPLES):
    _witnesses = []
    excluded = {a, b, c}
    for a_p in range(V):
        if a_p in excluded:
            continue
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                w = [
                    _get_idx(a_p, b, c),
                    _get_idx(a, b_p, c),
                    _get_idx(a_p, b_p, c),
                    _get_idx(a, b, c_p),
                    _get_idx(a_p, b, c_p),
                    _get_idx(a, b_p, c_p),
                    _get_idx(a_p, b_p, c_p)
                ]
                _witnesses.append(w)
    _witness_data[_i] = _witnesses

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _is_blocked(idx, presence):
    """Check if triple idx is blocked by current presence (using precomputed witnesses)."""
    witnesses = _witness_data[idx]
    for w in witnesses:
        all_present = True
        for t in w:
            if not _has_presence(presence, t):
                all_present = False
                break
        if all_present:
            return True
    return False

def _find_blocking_witness(idx, presence):
    """Find a specific blocking witness for triple idx, or None if not blocked."""
    witnesses = _witness_data[idx]
    for w in witnesses:
        all_present = True
        for t in w:
            if not _has_presence(presence, t):
                all_present = False
                break
        if all_present:
            return w
    return None

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        if not _is_blocked(idx, presence):
            presence[idx >> 6] |= (1 << (idx & 63))
            count += 1
    return count, presence

def _local_search(presence, count, deadline):
    """
    Local search: 
    1. Try to add any absent triple.
    2. If no add possible, try 1-swap moves (remove one, add one) to escape local optima.
    """
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add
        random.shuffle(absent_list)
        for idx in absent_list:
            if not _is_blocked(idx, presence):
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                absent_list.remove(idx)
                count += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try to swap (if no add possible)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        for new_idx in absent_list:
            blocking_w = _find_blocking_witness(new_idx, presence)
            if not blocking_w:
                # Should have been caught in Phase 1
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                improved = True
                break
            
            # Candidates for removal are the triples in the blocking witness
            candidates = list(set(blocking_w))
            candidates = [t for t in candidates if t in present_set]
            random.shuffle(candidates)
            
            swap_success = False
            for t_rem in candidates:
                presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                present_set.remove(t_rem)
                
                if not _is_blocked(new_idx, presence):
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    improved = True
                    swap_success = True
                    break
                else:
                    presence[t_rem >> 6] |= (1 << (t_rem & 63))
                    present_set.add(t_rem)
            
            if swap_success:
                break

    return count, presence

def _simulated_annealing(presence, count, deadline):
    """
    Simulated Annealing.
    Moves:
    1. Add (if possible)
    2. Remove (random)
    3. Swap (remove random present, try to add random absent)
    """
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    best_count = count
    best_presence = presence[:]
    best_present_set = set(present_set)
    
    while T > T_min and time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            if absent_list:
                new_idx = random.choice(absent_list)
                if not _is_blocked(new_idx, presence):
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    count += 1
                    if count > best_count:
                        best_count = count
                        best_presence = presence[:]
                        best_present_set = set(present_set)
                        
        elif move_type < 0.5:
            # Remove
            if present_set:
                old_idx = random.choice(list(present_set))
                presence[old_idx >> 6] &= ~(1 << (old_idx & 63))
                present_set.remove(old_idx)
                absent_list.append(old_idx)
                count -= 1
                
        else:
            # Swap
            if present_set and absent_list:
                new_idx = random.choice(absent_list)
                blocking_w = _find_blocking_witness(new_idx, presence)
                
                if not blocking_w:
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
                    candidates = list(set(blocking_w))
                    candidates = [t for t in candidates if t in present_set]
                    
                    if candidates:
                        random.shuffle(candidates)
                        for t_rem in candidates[:5]:
                            presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                            present_set.remove(t_rem)
                            
                            if not _is_blocked(new_idx, presence):
                                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                                present_set.add(new_idx)
                                absent_list.remove(new_idx)
                                if count > best_count:
                                    best_count = count
                                    best_presence = presence[:]
                                    best_present_set = set(present_set)
                                break
                            else:
                                presence[t_rem >> 6] |= (1 << (t_rem & 63))
                                present_set.add(t_rem)
        
        T *= cooling_rate
    
    # Restore best
    for i in range(N_TRIPLES):
        bit = 1 << (i & 63)
        word = i >> 6
        if i in best_present_set:
            presence[word] |= bit
        else:
            presence[word] &= ~bit
            
    return best_count, presence

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    # Run multiple attempts with different random seeds/orders
    while time.time() < deadline:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        # 1. Greedy
        count, presence = _greedy(order, deadline)
        
        # 2. Local Search
        count, presence = _local_search(presence, count, deadline)
        
        # 3. Simulated Annealing
        if time.time() < deadline - 10:
            count, presence = _simulated_annealing(presence, count, deadline)
            
        # 4. Final Polish
        if time.time() < deadline - 5:
            count, presence = _local_search(presence, count, deadline)
            
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
        # If we are close to the deadline, stop starting new attempts
        if time.time() > deadline - 5:
            break

    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
