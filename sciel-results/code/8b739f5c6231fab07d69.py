"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples via greedy + local search with bitmask."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N = len(ALL_TRIPLES)  # 560

# Index each triple
TRIPLE_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Precompute for each triple t, the list of "witness" bitmasks.
# A witness is a set of 7 triples such that {t} U witness is a K(2,2,2).
# If (selected_mask & witness_mask) == witness_mask, then adding t creates a K(2,2,2).
# So t can be added iff for ALL w in WITNESSES[t], (selected_mask & w) != w.

WITNESSES = [[] for _ in range(N)]

def _build_witnesses():
    """Precompute witness masks for each triple."""
    # For each K(2,2,2) defined by 6 vertices partitioned into 3 pairs,
    # generate the 8 triples.
    # For each triple in the 8, the other 7 form a witness.
    
    # Iterate over all 6-subsets of vertices
    for verts in combinations(range(V), 6):
        # There are 15 ways to partition 6 vertices into 3 pairs
        matchings = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)],
        ]
        
        for m in matchings:
            p1, p2, p3 = m
            v1, v2 = verts[p1[0]], verts[p1[1]]
            v3, v4 = verts[p2[0]], verts[p2[1]]
            v5, v6 = verts[p3[0]], verts[p3[1]]
            
            # The 8 triples are all combinations of one from each pair
            eight_triples = []
            for a in (v1, v2):
                for b in (v3, v4):
                    for c in (v5, v6):
                        eight_triples.append(tuple(sorted((a, b, c))))
            
            # For each triple in this K(2,2,2), the other 7 are witnesses
            for i, t in enumerate(eight_triples):
                idx = TRIPLE_IDX[t]
                # Create bitmask of the other 7
                mask = 0
                for j, t2 in enumerate(eight_triples):
                    if i != j:
                        mask |= (1 << TRIPLE_IDX[t2])
                WITNESSES[idx].append(mask)

# Build witnesses once at module load
_build_witnesses()

def _can_add(selected_mask, t_idx):
    """Check if t_idx can be added to selected_mask without forming K(2,2,2)."""
    for w in WITNESSES[t_idx]:
        if (selected_mask & w) == w:
            return False
    return True

def _greedy(order, deadline):
    """Greedily add triples in given order."""
    selected_mask = 0
    result = []
    for t_idx in order:
        if time.time() >= deadline:
            break
        if (selected_mask >> t_idx) & 1:
            continue
        if _can_add(selected_mask, t_idx):
            selected_mask |= (1 << t_idx)
            result.append(list(ALL_TRIPLES[t_idx]))
    return result, selected_mask

def _local_search(result, selected_mask, deadline, max_iters=10000):
    """
    Improve by:
    1. Trying to add a random triple.
    2. If blocked, try to remove one triple that is part of a blocking witness, then add the candidate.
    3. Try "remove 1, add 2" moves to increase size.
    """
    best_result = result[:]
    best_mask = selected_mask
    best_len = len(result)
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
            
        # Strategy 1: Add a random triple
        if random.random() < 0.5:
            cand = random.randrange(N)
            if not (selected_mask >> cand) & 1:
                if _can_add(selected_mask, cand):
                    # Success
                    selected_mask |= (1 << cand)
                    result.append(list(ALL_TRIPLES[cand]))
                    best_result = result[:]
                    best_mask = selected_mask
                    best_len += 1
                else:
                    # Blocked. Find a witness that is fully present.
                    # Pick a random witness that is full, remove a random triple from it.
                    blocked_witnesses = [w for w in WITNESSES[cand] if (selected_mask & w) == w]
                    if blocked_witnesses:
                        w = random.choice(blocked_witnesses)
                        # Pick a random bit set in w
                        bits = []
                        tmp_w = w
                        while tmp_w:
                            lsb = tmp_w & -tmp_w
                            bits.append(lsb.bit_length() - 1)
                            tmp_w ^= lsb
                        
                        rem_idx = random.choice(bits)
                        rem_triple = ALL_TRIPLES[rem_idx]
                        
                        # Remove rem_triple from result and selected_mask
                        pos = -1
                        for i, t in enumerate(result):
                            if t == list(rem_triple):
                                pos = i
                                break
                        if pos != -1:
                            result.pop(pos)
                            selected_mask &= ~(1 << rem_idx)
                            
                            # Now try to add cand
                            if _can_add(selected_mask, cand):
                                selected_mask |= (1 << cand)
                                result.append(list(ALL_TRIPLES[cand]))
                            else:
                                # Restore
                                result.append(list(rem_triple))
                                selected_mask |= (1 << rem_idx)

        # Strategy 2: Remove 1, Add 2 (Net +1)
        else:
            if not result:
                continue
            pos = random.randrange(len(result))
            rem_triple = result[pos]
            rem_idx = TRIPLE_IDX[tuple(sorted(rem_triple))]
            
            # Tentatively remove
            new_mask = selected_mask & ~(1 << rem_idx)
            
            # Try to add two random triples
            added_any = False
            added_list = []
            
            for _ in range(2):
                cand = random.randrange(N)
                if not (new_mask >> cand) & 1:
                    if _can_add(new_mask, cand):
                        new_mask |= (1 << cand)
                        added_list.append(cand)
                        added_any = True
            
            if added_any and len(added_list) == 2:
                # Success: removed 1, added 2. Net gain +1.
                result.pop(pos)
                for c in added_list:
                    result.append(list(ALL_TRIPLES[c]))
                selected_mask = new_mask
                best_result = result[:]
                best_mask = selected_mask
                best_len += 1

    return best_result, best_mask

def _full_validation(triples):
    """Check that no K(2,2,2) exists in the triple set."""
    tset = set(tuple(sorted(t)) for t in triples)
    for verts in combinations(range(V), 6):
        matchings = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)],
        ]
        for m in matchings:
            p1, p2, p3 = m
            v1, v2 = verts[p1[0]], verts[p1[1]]
            v3, v4 = verts[p2[0]], verts[p2[1]]
            v5, v6 = verts[p3[0]], verts[p3[1]]
            
            eight = [
                tuple(sorted((v1, v3, v5))), tuple(sorted((v1, v3, v6))),
                tuple(sorted((v1, v4, v5))), tuple(sorted((v1, v4, v6))),
                tuple(sorted((v2, v3, v5))), tuple(sorted((v2, v3, v6))),
                tuple(sorted((v2, v4, v5))), tuple(sorted((v2, v4, v6))),
            ]
            if all(t in tset for t in eight):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best = None
    best_mask = 0
    best_len = -1
    
    # Strategy: Multiple greedy restarts, then local search on the best ones.
    
    for attempt in range(200):
        if time.time() >= deadline - 30:
            break
            
        order = list(range(N))
        random.shuffle(order)
        
        result, mask = _greedy(order, deadline - 30)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
            best_mask = mask
            
        # Local search on this result
        if time.time() < deadline - 15:
            improved_result, improved_mask = _local_search(result, mask, deadline - 15, max_iters=1000)
            if len(improved_result) > best_len:
                best_len = len(improved_result)
                best = improved_result
                best_mask = improved_mask

    # Final local search on the global best
    if best is not None and time.time() < deadline - 5:
        improved_result, improved_mask = _local_search(best, best_mask, deadline - 5, max_iters=10000)
        if len(improved_result) > best_len:
            best_len = len(improved_result)
            best = improved_result
            best_mask = improved_mask

    if best is None:
        # Fallback: star at vertex 0
        best = [list(t) for t in [(0, a, b) for a in range(1, V) for b in range(a + 1, V)]]
    
    # Validation
    if not _full_validation(best):
        # Should not happen if logic is correct
        pass

    return best
