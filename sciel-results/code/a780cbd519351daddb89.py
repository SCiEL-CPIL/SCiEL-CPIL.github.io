"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """Precompute all bad triples where lcm(a,b)==lcm(a,c)==lcm(b,c)."""
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_pair_to_thirds(bad_triples):
    """For each pair (a,b), store set of c such that {a,b,c} is a bad triple."""
    pair_to_thirds = defaultdict(set)
    for a, b, c in bad_triples:
        pair_to_thirds[(a, b)].add(c)
        pair_to_thirds[(b, a)].add(c)
        pair_to_thirds[(a, c)].add(b)
        pair_to_thirds[(c, a)].add(b)
        pair_to_thirds[(b, c)].add(a)
        pair_to_thirds[(c, b)].add(a)
    return pair_to_thirds

def _build_elem_to_triples(bad_triples):
    """For each element, store list of triple indices it belongs to."""
    elem_to_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
    return elem_to_triples

def _can_add(elem, selected, pair_to_thirds):
    """Check if elem can be added to selected without creating a bad triple."""
    for other in selected:
        if (elem, other) in pair_to_thirds:
            for third in pair_to_thirds[(elem, other)]:
                if third in selected:
                    return False
    return True

def _greedy_forward_with_pairs(max_n, pair_to_thirds, deadline):
    """Greedy forward selection with precomputed pair-to-thirds mapping."""
    selected = []
    selected_set = set()
    for x in range(1, max_n + 1):
        if time.time() > deadline:
            break
        if _can_add(x, selected, pair_to_thirds):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_hit_set(bad_triples, max_n, deadline):
    """Greedy hitting set: remove elements to break all bad triples."""
    elem_to_triples = _build_elem_to_triples(bad_triples)
    active = [True] * len(bad_triples)
    selected = set(range(1, max_n + 1))
    
    while time.time() < deadline:
        best_elem = None
        best_count = 0
        for elem in selected:
            cnt = sum(1 for tidx in elem_to_triples.get(elem, []) if active[tidx])
            if cnt > best_count:
                best_count = cnt
                best_elem = elem
        if best_elem is None or best_count == 0:
            break
        selected.discard(best_elem)
        for tidx in elem_to_triples.get(best_elem, []):
            if active[tidx]:
                active[tidx] = False
    
    return list(selected)

def _local_search_polish(selected, bad_triples, max_n, pair_to_thirds, deadline):
    """Iterative local search: add all possible, then try 1-for-2 swaps."""
    selected_set = set(selected)
    candidates = [x for x in range(1, max_n + 1) if x not in selected_set]
    
    # Phase 1: Add all possible elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in candidates:
            if elem in selected_set:
                continue
            if _can_add(elem, selected_set, pair_to_thirds):
                selected_set.add(elem)
                improved = True
        candidates = [x for x in range(1, max_n + 1) if x not in selected_set]
    
    # Phase 2: Try 1-for-2 swaps
    if time.time() < deadline:
        for remove_elem in list(selected_set):
            if time.time() > deadline:
                break
            selected_set.discard(remove_elem)
            added = []
            for add_elem in candidates:
                if add_elem in selected_set:
                    continue
                if _can_add(add_elem, selected_set, pair_to_thirds):
                    selected_set.add(add_elem)
                    added.append(add_elem)
            if len(added) <= 1:
                selected_set.difference_update(added)
                selected_set.add(remove_elem)
            else:
                candidates = [x for x in range(1, max_n + 1) if x not in selected_set]
    
    return list(selected_set)

def _simulated_annealing(initial_set, bad_triples, max_n, pair_to_thirds, deadline):
    """Simulated annealing with efficient active-triple counting."""
    elem_to_triples = _build_elem_to_triples(bad_triples)
    n_triples = len(bad_triples)
    
    current_mask = 0
    for x in initial_set:
        current_mask |= (1 << x)
    
    # Track active triples incrementally
    active_count = 0
    triple_active = [False] * n_triples
    for idx, (a, b, c) in enumerate(bad_triples):
        if (current_mask >> a) & 1 and (current_mask >> b) & 1 and (current_mask >> c) & 1:
            triple_active[idx] = True
            active_count += 1
    
    current_size = bin(current_mask).count('1')
    penalty = 100
    current_score = current_size - penalty * active_count
    best_mask = current_mask
    best_score = current_score
    
    T = 5.0
    T_end = 0.01
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_end:
        if random.random() < 0.5:
            not_selected = [x for x in range(1, max_n + 1) if not (current_mask >> x) & 1]
            if not not_selected:
                T *= cooling_rate
                continue
            elem = random.choice(not_selected)
            new_mask = current_mask | (1 << elem)
        else:
            selected_elems = [x for x in range(1, max_n + 1) if (current_mask >> x) & 1]
            if not selected_elems:
                T *= cooling_rate
                continue
            elem = random.choice(selected_elems)
            new_mask = current_mask & ~(1 << elem)
        
        # Compute new active count efficiently
        if new_mask != current_mask:
            # Determine what changed
            diff = current_mask ^ new_mask
            changed_elem = diff.bit_length() - 1
            
            if (new_mask >> changed_elem) & 1:
                # Added changed_elem: count new active triples
                new_triples = 0
                for tidx in elem_to_triples.get(changed_elem, []):
                    a, b, c = bad_triples[tidx]
                    if (new_mask >> a) & 1 and (new_mask >> b) & 1 and (new_mask >> c) & 1:
                        new_triples += 1
                delta_triples = new_triples
            else:
                # Removed changed_elem: count deactivated triples
                deactivated = 0
                for tidx in elem_to_triples.get(changed_elem, []):
                    a, b, c = bad_triples[tidx]
                    if triple_active[tidx] and (current_mask >> a) & 1 and (current_mask >> b) & 1 and (current_mask >> c) & 1:
                        deactivated += 1
                delta_triples = -deactivated
            
            new_active_count = active_count + delta_triples
            new_size = current_size + (1 if (new_mask >> changed_elem) & 1 else -1)
            new_score = new_size - penalty * new_active_count
        else:
            new_active_count = active_count
            new_size = current_size
            new_score = current_score
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < (2.718281828) ** (delta / T):
            # Update state
            if new_mask != current_mask:
                changed_elem = (current_mask ^ new_mask).bit_length() - 1
                if (new_mask >> changed_elem) & 1:
                    for tidx in elem_to_triples.get(changed_elem, []):
                        a, b, c = bad_triples[tidx]
                        if (new_mask >> a) & 1 and (new_mask >> b) & 1 and (new_mask >> c) & 1:
                            if not triple_active[tidx]:
                                triple_active[tidx] = True
                                active_count += 1
                else:
                    for tidx in elem_to_triples.get(changed_elem, []):
                        if triple_active[tidx]:
                            a, b, c = bad_triples[tidx]
                            if not ((new_mask >> a) & 1 and (new_mask >> b) & 1 and (new_mask >> c) & 1):
                                triple_active[tidx] = False
                                active_count -= 1
                current_size += 1 if (new_mask >> changed_elem) & 1 else -1
            
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling_rate
    
    result = [x for x in range(1, max_n + 1) if (best_mask >> x) & 1]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    pair_to_thirds = _build_pair_to_thirds(bad_triples)
    
    # Strategy 1: Greedy forward
    result1 = _greedy_forward_with_pairs(max_n, pair_to_thirds, deadline)
    
    # Strategy 2: Greedy hitting set
    result2 = _greedy_hit_set(bad_triples, max_n, deadline)
    
    # Pick better initial result
    best = result1 if len(result1) >= len(result2) else result2
    
    # Local search polish
    if time.time() < deadline - 20:
        best = _local_search_polish(best, bad_triples, max_n, pair_to_thirds, deadline)
    
    # Simulated annealing
    if time.time() < deadline - 5:
        sa_result = _simulated_annealing(best, bad_triples, max_n, pair_to_thirds, deadline)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final polish
    if time.time() < deadline - 2:
        best = _local_search_polish(best, bad_triples, max_n, pair_to_thirds, deadline)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
