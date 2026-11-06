"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples via bitmask + local search."""
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
        # There are 15 perfect matchings on 6 vertices.
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
    for t_idx in order:
        if time.time() >= deadline:
            break
        if (selected_mask >> t_idx) & 1:
            continue
        if _can_add(selected_mask, t_idx):
            selected_mask |= (1 << t_idx)
    return selected_mask

def _local_search(selected_mask, deadline, max_iters=1000):
    """
    Improve selected_mask using local search.
    Returns the improved mask.
    """
    in_set = [False] * N
    for i in range(N):
        if (selected_mask >> i) & 1:
            in_set[i] = True
    
    # List of currently selected triple indices for random removal
    selected_indices = [i for i in range(N) if in_set[i]]
    
    best_mask = selected_mask
    best_len = bin(selected_mask).count('1')
    
    for iteration in range(max_iters):
        if time.time() >= deadline:
            break
        
        r = random.random()
        
        if r < 0.5:
            # Try to add a random triple
            cand = random.randrange(N)
            if not in_set[cand]:
                if _can_add(selected_mask, cand):
                    # Success
                    in_set[cand] = True
                    selected_mask |= (1 << cand)
                    selected_indices.append(cand)
                    best_mask = selected_mask
                    best_len += 1
                else:
                    # Blocked. Try to unblock by removing a triple from a blocking witness.
                    blocked_witnesses = [w for w in WITNESSES[cand] if (selected_mask & w) == w]
                    if blocked_witnesses:
                        w = random.choice(blocked_witnesses)
                        # Find bits in w
                        bits = []
                        tmp_w = w
                        while tmp_w:
                            lsb = tmp_w & -tmp_w
                            bits.append(lsb.bit_length() - 1)
                            tmp_w ^= lsb
                        
                        rem_idx = random.choice(bits)
                        
                        # Tentatively remove rem_idx
                        in_set[rem_idx] = False
                        selected_mask &= ~(1 << rem_idx)
                        selected_indices.remove(rem_idx)
                        
                        # Try to add cand
                        if _can_add(selected_mask, cand):
                            # Swap successful
                            in_set[cand] = True
                            selected_mask |= (1 << cand)
                            selected_indices.append(cand)
                            # Size unchanged, so best_len doesn't change.
                            # We accept this state to explore further.
                        else:
                            # Failed to add cand. Revert.
                            in_set[rem_idx] = True
                            selected_mask |= (1 << rem_idx)
                            selected_indices.append(rem_idx)

        elif r < 0.8:
            # Swap (1-out, 1-in) random
            if not selected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            
            in_set[rem_idx] = False
            selected_mask &= ~(1 << rem_idx)
            selected_indices.remove(rem_idx)
            
            cand = random.randrange(N)
            if not in_set[cand] and _can_add(selected_mask, cand):
                in_set[cand] = True
                selected_mask |= (1 << cand)
                selected_indices.append(cand)
            else:
                # Revert
                in_set[rem_idx] = True
                selected_mask |= (1 << rem_idx)
                selected_indices.append(rem_idx)

        else:
            # Improve (1-out, 2-in)
            if not selected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            
            in_set[rem_idx] = False
            selected_mask &= ~(1 << rem_idx)
            selected_indices.remove(rem_idx)
            
            added_list = []
            # Try to add 2 random triples
            for _ in range(2):
                cand = random.randrange(N)
                if not in_set[cand] and _can_add(selected_mask, cand):
                    in_set[cand] = True
                    selected_mask |= (1 << cand)
                    selected_indices.append(cand)
                    added_list.append(cand)
            
            if len(added_list) == 2:
                # Success
                best_mask = selected_mask
                best_len += 1
            else:
                # Revert
                in_set[rem_idx] = True
                selected_mask |= (1 << rem_idx)
                selected_indices.append(rem_idx)
                for c in added_list:
                    in_set[c] = False
                    selected_mask &= ~(1 << c)
                    selected_indices.remove(c)

    return best_mask

def _mask_to_list(mask):
    """Convert bitmask to list of triples."""
    result = []
    for i in range(N):
        if (mask >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
    return result

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
    deadline = start_time + 115  # Safety margin
    
    best_mask = 0
    best_len = -1
    
    # Strategy: Multiple greedy restarts, then local search on the best ones.
    
    local_search_budget = 1000
    
    for attempt in range(200):
        if time.time() >= deadline - 30:
            break
            
        order = list(range(N))
        random.shuffle(order)
        
        mask = _greedy(order, deadline - 30)
        
        if bin(mask).count('1') > best_len:
            best_len = bin(mask).count('1')
            best_mask = mask
            
        # Local search on this result
        if time.time() < deadline - 10:
            improved_mask = _local_search(mask, deadline - 10, max_iters=local_search_budget)
            if bin(improved_mask).count('1') > best_len:
                best_len = bin(improved_mask).count('1')
                best_mask = improved_mask

    # Final local search on the global best
    if best_mask != 0 and time.time() < deadline - 2:
        improved_mask = _local_search(best_mask, deadline - 2, max_iters=5000)
        if bin(improved_mask).count('1') > best_len:
            best_len = bin(improved_mask).count('1')
            best_mask = improved_mask

    if best_mask == 0:
        # Fallback: star at vertex 0
        mask = 0
        for a in range(1, V):
            for b in range(a + 1, V):
                t = (0, a, b)
                mask |= (1 << TRIPLE_IDX[t])
        best_mask = mask

    # Convert to list and validate
    result = _mask_to_list(best_mask)
    
    # Final validation and cleanup if necessary
    if not _full_validation(result):
        # This should not happen if logic is correct.
        # Fallback to a smaller valid set if needed.
        # We can try to remove triples causing conflict.
        # For safety, we return the star which is always valid.
        result = []
        mask = 0
        for a in range(1, V):
            for b in range(a + 1, V):
                t = (0, a, b)
                result.append(list(t))
                mask |= (1 << TRIPLE_IDX[t])

    return result
