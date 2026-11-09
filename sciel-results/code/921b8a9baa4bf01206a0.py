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

def _build_pair_to_thirds(bad_triples):
    """For each unordered pair (a,b), store set of c such that {a,b,c} is bad."""
    pair_to_thirds = defaultdict(set)
    for a, b, c in bad_triples:
        pair_to_thirds[(a, b)].add(c)
        pair_to_thirds[(b, a)].add(c)
        pair_to_thirds[(a, c)].add(b)
        pair_to_thirds[(c, a)].add(b)
        pair_to_thirds[(b, c)].add(a)
        pair_to_thirds[(c, b)].add(a)
    return pair_to_thirds

def _build_element_to_triples(bad_triples):
    """Map each element to list of triple indices it belongs to."""
    elem_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    return elem_triples

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

def _count_active_triples_fast(selected_mask, bad_triples):
    """Count active bad triples using bitmask check."""
    count = 0
    for a, b, c in bad_triples:
        if (selected_mask & (1 << a)) and (selected_mask & (1 << b)) and (selected_mask & (1 << c)):
            count += 1
    return count

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
    pair_to_thirds = _build_pair_to_thirds(bad_triples)
    
    selected_mask = 0
    selected_list = []
    
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        
        valid = True
        for y in selected_list:
            if (x, y) in pair_to_thirds:
                for z in pair_to_thirds[(x, y)]:
                    if selected_mask & (1 << z):
                        valid = False
                        break
            if not valid:
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
    pair_to_thirds = _build_pair_to_thirds(bad_triples)
    
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
    elem_triple_indices = _build_element_to_triples(bad_triples)
    
    active = [True] * len(bad_triples)
    selected_mask = (1 << (max_n + 1)) - 2  # All bits 1..max_n set
    selected_list = list(range(1, max_n + 1))
    
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
    Uses incremental triple counting for speed.
    """
    pair_to_thirds = _build_pair_to_thirds(bad_triples)
    elem_triple_indices = _build_element_to_triples(bad_triples)
    
    # Convert initial set to mask
    current_mask = 0
    for x in initial_set:
        current_mask |= (1 << x)
    
    # Track active triples count incrementally
    current_triples = _count_active_triples_fast(current_mask, bad_triples)
    current_size = bin(current_mask).count('1')
    current_score = current_size - 100 * current_triples
    
    best_mask = current_mask
    best_score = current_score
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.995
    T = T_start
    
    # Precompute bit masks for each element
    elem_bits = {x: (1 << x) for x in range(1, max_n + 1)}
    
    steps = 0
    while time.time() < deadline and T > T_end:
        # Choose a random move
        if random.random() < 0.5:
            # Try to add an element
            not_selected = [x for x in range(1, max_n + 1) if not (current_mask & elem_bits[x])]
            if not not_selected:
                T *= cooling_rate
                continue
            elem = random.choice(not_selected)
            new_mask = current_mask | elem_bits[elem]
            
            # Count new triples created by adding elem
            new_triples_delta = 0
            for other in range(1, max_n + 1):
                if current_mask & elem_bits[other]:
                    if (elem, other) in pair_to_thirds:
                        for third in pair_to_thirds[(elem, other)]:
                            if current_mask & elem_bits[third]:
                                new_triples_delta += 1
            new_triples = current_triples + new_triples_delta
            new_size = current_size + 1
        else:
            # Try to remove an element
            selected = [x for x in range(1, max_n + 1) if current_mask & elem_bits[x]]
            if not selected:
                T *= cooling_rate
                continue
            elem = random.choice(selected)
            new_mask = current_mask & ~elem_bits[elem]
            
            # Count triples removed by removing elem
            removed_triples = 0
            for other in range(1, max_n + 1):
                if current_mask & elem_bits[other]:
                    if (elem, other) in pair_to_thirds:
                        for third in pair_to_thirds[(elem, other)]:
                            if current_mask & elem_bits[third]:
                                removed_triples += 1
            new_triples = current_triples - removed_triples
            new_size = current_size - 1
        
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
    result = [x for x in range(1, max_n + 1) if best_mask & elem_bits[x]]
    return result

def _local_search_with_swaps(initial_set, bad_triples, max_n, deadline):
    """
    Enhanced local search that also tries swaps (remove one, add two).
    """
    pair_to_thirds = _build_pair_to_thirds(bad_triples)
    
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
    
    # Phase 2: Try swaps - remove one element, try to add others
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
            
            # Keep if we added at least as many as we removed (net non-negative)
            if len(added) < 1:
                # Revert: nothing was added, put back removed element
                selected.add(remove_elem)
    
    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline, 8s safety margin before 90s kill
    
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
    if time.time() < deadline - 15:
        best = _local_search_with_swaps(best, bad_triples, max_n, deadline - 15)
    
    # Apply simulated annealing to escape local optima
    if time.time() < deadline - 8:
        sa_result = _simulated_annealing(best, bad_triples, max_n, deadline - 8)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final local search polish
    if time.time() < deadline - 2:
        best = _local_search_with_swaps(best, bad_triples, max_n, deadline - 2)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
