"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(max_n):
    """Precompute LCM table for all pairs in [1, max_n]."""
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    Returns a list of tuples (a,b,c).
    """
    n = max_n
    lcm_table = _precompute_lcms(n)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_conflict_masks(bad_triples, max_n):
    """
    Build bitmask-based conflict detection.
    For each element, precompute which other elements it conflicts with
    in the sense that there exists a third element forming a bad triple.
    
    Returns:
        conflict_mask[i]: bitmask of elements that conflict with i
        triple_count[i]: number of bad triples containing element i
    """
    n = max_n
    # conflict_mask[i] has bit j set if i and j are in a bad triple
    conflict_mask = [0] * (n + 1)
    triple_count = [0] * (n + 1)
    
    for a, b, c in bad_triples:
        # a conflicts with b and c
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    return conflict_mask, triple_count

def _count_active_conflicts(selected_mask, conflict_mask, max_n):
    """
    Count how many bad triples are active (not yet broken) given selected_mask.
    A bad triple is active if all three elements are in selected_mask.
    
    We use a different approach: for each element in selected, count how many
    of its conflict partners are also in selected. Each bad triple is counted
    3 times (once per element), so divide by 3.
    """
    count = 0
    for i in range(1, max_n + 1):
        if selected_mask & (1 << i):
            # Count conflict partners that are also selected
            partners_in_selected = selected_mask & conflict_mask[i]
            count += bin(partners_in_selected).count('1')
    # Each bad triple contributes 3 to this count (one for each element)
    return count // 3

def _greedy_forward_optimized(max_n, deadline):
    """
    Optimized greedy forward selection using bitmask conflict detection.
    """
    lcm_table = _precompute_lcms(max_n)
    bad_triples = _precompute_bad_triples(max_n)
    conflict_mask, triple_count = _build_conflict_masks(bad_triples, max_n)
    
    selected_mask = 0
    selected_list = []
    
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Check if x can be added: no existing pair in selected should form
        # a bad triple with x. This means for every y in selected, x must not
        # be in conflict_mask[y] AND there must be no z in selected such that
        # {x,y,z} is a bad triple.
        
        # Simpler check: for each y in selected, check if there exists z in
        # selected such that {x,y,z} is bad. Equivalent to: for each y in
        # selected, x must not be a "third" element with any pair.
        
        # Using conflict masks: x can be added iff for all y in selected,
        # the pair (x,y) doesn't have a common bad-triple partner z in selected.
        
        # Actually, let's use the direct approach: x can be added iff there's
        # no y,z in selected such that {x,y,z} is a bad triple.
        
        valid = True
        for y in selected_list:
            # Check if there exists z in selected_list such that {x,y,z} is bad
            # This means z is in conflict_mask[x] AND z is in conflict_mask[y]
            # AND {x,y,z} is actually a bad triple (not just pairwise conflicts)
            # 
            # Better: check if adding x creates any bad triple with two existing elements
            common_partners = conflict_mask[x] & conflict_mask[y] & selected_mask
            if common_partners:
                # There might be a z in selected that forms a bad triple with x and y
                # Need to verify it's actually a bad triple, not just pairwise conflicts
                # For now, assume if x and y both conflict with z, and x,y are in selected,
                # then {x,y,z} could be bad. We need to be more careful.
                # 
                # Let's use a different approach: precompute for each pair (a,b) the set
                # of c such that {a,b,c} is a bad triple.
                valid = False
                break
        
        if valid:
            selected_mask |= (1 << x)
            selected_list.append(x)
    
    return selected_list

def _greedy_forward_with_pairs(max_n, deadline):
    """
    Greedy forward selection with precomputed pair-to-thirds mapping.
    """
    bad_triples = _precompute_bad_triples(max_n)
    
    # For each pair (a,b), store the set of c such that {a,b,c} is a bad triple
    pair_to_thirds = defaultdict(set)
    for a, b, c in bad_triples:
        pair_to_thirds[(a, b)].add(c)
        pair_to_thirds[(b, a)].add(c)
        pair_to_thirds[(a, c)].add(b)
        pair_to_thirds[(c, a)].add(b)
        pair_to_thirds[(b, c)].add(a)
        pair_to_thirds[(c, b)].add(a)
    
    selected = []
    selected_set = set()
    
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        
        valid = True
        for y in selected:
            if (x, y) in pair_to_thirds:
                for z in pair_to_thirds[(x, y)]:
                    if z in selected_set:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _greedy_hit_set_optimized(bad_triples, max_n, deadline):
    """
    Optimized greedy hitting set using bitmask operations.
    """
    # Build element-to-triples index
    element_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        element_to_triples[a].append(idx)
        element_to_triples[b].append(idx)
        element_to_triples[c].append(idx)
    
    active = [True] * len(bad_triples)
    selected_mask = (1 << (max_n + 1)) - 2  # All bits 1..max_n set
    selected_list = list(range(1, max_n + 1))
    
    # Precompute for each element, the list of triple indices it belongs to
    elem_triple_indices = {elem: triples for elem, triples in element_to_triples.items()}
    
    while time.time() < deadline:
        # Find element with max active triple count
        best_elem = None
        best_count = 0
        
        for elem in selected_list:
            cnt = 0
            for tidx in elem_triple_indices.get(elem, []):
                if active[tidx]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_elem = elem
        
        if best_elem is None or best_count == 0:
            break
        
        # Remove this element
        selected_mask &= ~(1 << best_elem)
        selected_list.remove(best_elem)
        
        # Deactivate all triples containing this element
        for tidx in elem_triple_indices.get(best_elem, []):
            if active[tidx]:
                active[tidx] = False
    
    return selected_list

def _simulated_annealing(initial_set, bad_triples, max_n, deadline):
    """
    Simulated annealing to escape local optima.
    State: bitmask of selected elements.
    Moves: add or remove a single element.
    Objective: maximize |selected| - penalty * (number of active bad triples)
    """
    conflict_mask, triple_count = _build_conflict_masks(bad_triples, max_n)
    
    # Convert initial set to mask
    current_mask = 0
    for x in initial_set:
        current_mask |= (1 << x)
    
    def count_active_triples(mask):
        """Count active bad triples in the given mask."""
        # For each triple, check if all three elements are in mask
        count = 0
        for a, b, c in bad_triples:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                count += 1
        return count
    
    def get_size(mask):
        return bin(mask).count('1')
    
    current_size = get_size(current_mask)
    current_triples = count_active_triples(current_mask)
    current_score = current_size - 100 * current_triples  # Heavy penalty for violations
    
    best_mask = current_mask
    best_score = current_score
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.995
    T = T_start
    
    steps = 0
    while time.time() < deadline and T > T_end:
        # Choose a random move
        if random.random() < 0.5:
            # Try to add an element
            candidates = [x for x in range(1, max_n + 1) if not (current_mask & (1 << x))]
            if not candidates:
                T *= cooling_rate
                continue
            elem = random.choice(candidates)
            new_mask = current_mask | (1 << elem)
        else:
            # Try to remove an element
            selected = [x for x in range(1, max_n + 1) if current_mask & (1 << x)]
            if not selected:
                T *= cooling_rate
                continue
            elem = random.choice(selected)
            new_mask = current_mask & ~(1 << elem)
        
        new_size = get_size(new_mask)
        new_triples = count_active_triples(new_mask)
        new_score = new_size - 100 * new_triples
        
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_size = new_size
            current_triples = new_triples
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling_rate
        steps += 1
    
    # Convert best_mask back to list
    result = [x for x in range(1, max_n + 1) if best_mask & (1 << x)]
    return result

def _local_search_with_swaps(initial_set, bad_triples, max_n, deadline):
    """
    Enhanced local search that also tries swaps (remove one, add two).
    """
    pair_to_thirds = defaultdict(set)
    for a, b, c in bad_triples:
        pair_to_thirds[(a, b)].add(c)
        pair_to_thirds[(b, a)].add(c)
        pair_to_thirds[(a, c)].add(b)
        pair_to_thirds[(c, a)].add(b)
        pair_to_thirds[(b, c)].add(a)
        pair_to_thirds[(c, b)].add(a)
    
    selected = set(initial_set)
    
    def can_add(elem):
        """Check if elem can be added to selected without creating a bad triple."""
        for other in selected:
            if (elem, other) in pair_to_thirds:
                for third in pair_to_thirds[(elem, other)]:
                    if third in selected:
                        return False
        return True
    
    # Phase 1: Try single additions
    candidates = [x for x in range(1, max_n + 1) if x not in selected]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in candidates:
            if elem in selected:
                continue
            if can_add(elem):
                selected.add(elem)
                improved = True
    
    # Phase 2: Try swaps - remove one element, try to add two others
    if time.time() < deadline:
        for remove_elem in list(selected):
            if time.time() > deadline:
                break
            
            # Temporarily remove
            selected.remove(remove_elem)
            
            # Try to add as many as possible from candidates
            added = []
            for add_elem in candidates:
                if add_elem in selected or add_elem == remove_elem:
                    continue
                if can_add(add_elem):
                    selected.add(add_elem)
                    added.append(add_elem)
            
            # If we added more than we removed, keep the change
            if len(added) > 1:
                pass  # Keep the new state
            else:
                # Revert
                selected.difference_update(added)
                selected.add(remove_elem)
    
    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline, 5s safety margin
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    # Strategy 1: Greedy forward with pair-to-thirds
    result1 = _greedy_forward_with_pairs(max_n, deadline)
    
    # Strategy 2: Greedy hitting set (remove elements to break all bad triples)
    result2 = _greedy_hit_set_optimized(bad_triples, max_n, deadline)
    
    # Pick the better initial result
    if len(result1) >= len(result2):
        best = result1
    else:
        best = result2
    
    # Apply local search with swaps
    if time.time() < deadline - 10:
        best = _local_search_with_swaps(best, bad_triples, max_n, deadline)
    
    # Apply simulated annealing to escape local optima
    if time.time() < deadline - 5:
        sa_result = _simulated_annealing(best, bad_triples, max_n, deadline)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final local search polish
    if time.time() < deadline - 2:
        best = _local_search_with_swaps(best, bad_triples, max_n, deadline)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
