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

def _precompute_k222s():
    """Precompute all K(2,2,2) configurations and their associated triples."""
    k222_list = []
    seen = set()
    
    # A K(2,2,2) is determined by choosing:
    # - 2 points for part A: a1, a2
    # - 2 points for part B: b1, b2  
    # - 2 points for part C: c1, c2
    # where all 6 points are distinct
    # The 8 transversal triples are:
    # (a1,b1,c1), (a1,b1,c2), (a1,b2,c1), (a1,b2,c2),
    # (a2,b1,c1), (a2,b1,c2), (a2,b2,c1), (a2,b2,c2)
    
    points = list(range(V))
    for a1 in range(V):
        for a2 in range(a1+1, V):
            for b1 in range(V):
                if b1 == a1 or b1 == a2:
                    continue
                for b2 in range(V):
                    if b2 == a1 or b2 == a2 or b2 == b1:
                        continue
                    for c1 in range(V):
                        if c1 in (a1, a2, b1, b2):
                            continue
                        for c2 in range(V):
                            if c2 in (a1, a2, b1, b2, c1):
                                continue
                            
                            # Generate the 8 transversal triples
                            triples_in_k222 = [
                                _tidx(a1, b1, c1),
                                _tidx(a1, b1, c2),
                                _tidx(a1, b2, c1),
                                _tidx(a1, b2, c2),
                                _tidx(a2, b1, c1),
                                _tidx(a2, b1, c2),
                                _tidx(a2, b2, c1),
                                _tidx(a2, b2, c2)
                            ]
                            
                            # Use frozenset as key to avoid duplicates
                            key = frozenset(triples_in_k222)
                            if key not in seen:
                                seen.add(key)
                                k222_list.append(tuple(sorted(triples_in_k222)))
    
    return k222_list

# Precompute all K(2,2,2) configurations
K222_LIST = _precompute_k222s()
N_K222 = len(K222_LIST)

# For each triple, precompute which K(2,2,2)s it belongs to
triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for k_idx, k222 in enumerate(K222_LIST):
    for t_idx in k222:
        triple_to_k222[t_idx].append(k_idx)

# Precompute for each K(2,2,2), a bitmask of which triples are present (for fast counting)
# We'll use a different approach: for each K(2,2,2), store the 8 triple indices

def _count_k222_violations(presence):
    """Count number of K(2,2,2) configurations where all 8 triples are present."""
    violations = 0
    for k222 in K222_LIST:
        all_present = True
        for t_idx in k222:
            if not _has_presence(presence, t_idx):
                all_present = False
                break
        if all_present:
            violations += 1
    return violations

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
    """Simulated annealing with bit-parallel conflict tracking."""
    # Energy = -count + penalty * violations
    # We want to minimize energy, so maximize count while minimizing violations
    
    penalty = 100  # Large penalty to strongly discourage violations
    T_start = 50.0
    T_end = 0.1
    cooling = 0.9995
    
    T = T_start
    best_presence = presence[:]
    best_count = count
    best_energy = -count + penalty * _count_k222_violations(presence)
    
    # Track conflicts per K(2,2,2) for fast updates
    # conflict_count[k] = number of present triples in K(2,2,2) k
    conflict_count = [0] * N_K222
    for k_idx, k222 in enumerate(K222_LIST):
        cnt = 0
        for t_idx in k222:
            if _has_presence(presence, t_idx):
                cnt += 1
        conflict_count[k_idx] = cnt
    
    def get_energy(pres, cnt, conflicts):
        violations = sum(1 for c in conflicts if c == 8)
        return -cnt + penalty * violations
    
    current_energy = get_energy(presence, count, conflict_count)
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        if random.random() < 0.3:
            # Try to add an absent triple
            absent = [i for i in range(N_TRIPLES) if not _has_presence(presence, i)]
            if not absent:
                continue
            idx = random.choice(absent)
            a, b, c = ALL_TRIPLES[idx]
            
            # Check if adding creates violation
            # Find all K(2,2,2)s containing this triple
            affected_k222s = triple_to_k222[idx]
            
            # Check if any K(2,2,2) would become fully present
            would_violate = False
            for k_idx in affected_k222s:
                if conflict_count[k_idx] == 7:  # 7 others present, adding makes 8
                    would_violate = True
                    break
            
            if would_violate:
                # Compute delta energy
                delta_e = 1 + penalty  # +1 count, +1 violation
                if random.random() < min(1.0, 2.71828 ** (-delta_e / T)):
                    # Accept
                    presence[idx >> 6] |= (1 << (idx & 63))
                    count += 1
                    for k_idx in affected_k222s:
                        conflict_count[k_idx] += 1
                    current_energy = get_energy(presence, count, conflict_count)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_presence = presence[:]
                        best_count = count
            else:
                # Always accept (improves or neutral)
                presence[idx >> 6] |= (1 << (idx & 63))
                count += 1
                for k_idx in affected_k222s:
                    conflict_count[k_idx] += 1
                current_energy = get_energy(presence, count, conflict_count)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_presence = presence[:]
                    best_count = count
        else:
            # Try to remove a present triple
            present = [i for i in range(N_TRIPLES) if _has_presence(presence, i)]
            if not present:
                continue
            idx = random.choice(present)
            
            affected_k222s = triple_to_k222[idx]
            
            # Check if removing fixes a violation
            fixes_violation = False
            for k_idx in affected_k222s:
                if conflict_count[k_idx] == 8:
                    fixes_violation = True
                    break
            
            if fixes_violation:
                # Always accept (removes violation)
                presence[idx >> 6] &= ~(1 << (idx & 63))
                count -= 1
                for k_idx in affected_k222s:
                    conflict_count[k_idx] -= 1
                current_energy = get_energy(presence, count, conflict_count)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_presence = presence[:]
                    best_count = count
            else:
                # Compute delta energy
                delta_e = -1  # -1 count, no change in violations
                if random.random() < min(1.0, 2.71828 ** (-delta_e / T)):
                    # Accept
                    presence[idx >> 6] &= ~(1 << (idx & 63))
                    count -= 1
                    for k_idx in affected_k222s:
                        conflict_count[k_idx] -= 1
                    current_energy = get_energy(presence, count, conflict_count)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_presence = presence[:]
                        best_count = count
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    # Restore best solution
    for i in range(N_TRIPLES):
        if _has_presence(best_presence, i):
            presence[i >> 6] |= (1 << (i & 63))
        else:
            presence[i >> 6] &= ~(1 << (i & 63))
    
    return best_count, best_presence

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

def _perturb_and_improve(presence, count, deadline):
    """Perturb the current solution and run local search + SA."""
    # Perturb: remove some random triples
    present_list = [i for i in range(N_TRIPLES) if _has_presence(presence, i)]
    n_remove = max(1, len(present_list) // 10)
    to_remove = random.sample(present_list, min(n_remove, len(present_list)))
    
    for idx in to_remove:
        presence[idx >> 6] &= ~(1 << (idx & 63))
        count -= 1
    
    # Run local search
    count, presence = _local_search(presence, count, deadline)
    
    # Run simulated annealing
    count, presence = _simulated_annealing(presence, count, deadline)
    
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
        
        # Simulated annealing
        count, presence = _simulated_annealing(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
        
        # Perturb and improve from best
        if attempts % 3 == 0 and time.time() < deadline - 10:
            perturbed_presence = best_presence[:]
            perturbed_count = best_count
            perturbed_count, perturbed_presence = _perturb_and_improve(perturbed_presence, perturbed_count, deadline)
            
            if perturbed_count > best_count:
                best_count = perturbed_count
                best_presence = perturbed_presence[:]
        
        if time.time() > deadline - 5:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
