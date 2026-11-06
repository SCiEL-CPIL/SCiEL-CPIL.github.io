"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max or c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _greedy_independent_set(triples, elem_to_triples, n_max, deadline):
    """Greedy: start with all, remove highest-conflict element iteratively."""
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        best_elem = max(S, key=lambda x: elem_conflict[x])
        if elem_conflict[best_elem] == 0:
            break
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return sorted(S)

def _local_search(triples, elem_to_triples, n_max, start_set, deadline, max_iter=5000):
    """Local search: 1-opt (add) and 2-opt (remove one, add up to two)."""
    S = set(start_set)
    removed = set(range(1, n_max + 1)) - S
    
    def count_conflicts(s):
        cnt = 0
        for a, b, c in triples:
            if a in s and b in s and c in s:
                cnt += 1
        return cnt
    
    if count_conflicts(S) > 0:
        S = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
        removed = set(range(1, n_max + 1)) - S
    
    best_size = len(S)
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        
        # Phase 1: try to add any single removed element (1-opt)
        candidates = list(removed)
        random.shuffle(candidates)
        for elem in candidates:
            if time.time() > deadline:
                break
            new_conflicts = 0
            for tidx in elem_to_triples[elem]:
                a, b, c = triples[tidx]
                others = [x for x in (a, b, c) if x != elem]
                if all(x in S for x in others):
                    new_conflicts += 1
            if new_conflicts == 0:
                S.add(elem)
                removed.discard(elem)
                improved = True
                best_size = max(best_size, len(S))
                break
        
        if improved:
            no_improve = 0
            continue
        
        # Phase 2: 2-opt — remove one element from S, then try to add up to 2 from removed
        # Only try if S has elements to remove
        if not S:
            break
            
        # Pick elements to remove: prioritize those with high conflict count
        s_list = list(S)
        random.shuffle(s_list)
        
        found_2opt = False
        for rem_elem in s_list[:min(30, len(s_list))]:
            if time.time() > deadline:
                break
            
            # Temporarily remove rem_elem
            S.remove(rem_elem)
            removed.add(rem_elem)
            
            # Now try to add elements from removed (including rem_elem itself won't help since we just removed it,
            # but other elements might now be addable)
            added_count = 0
            added_elems = []
            
            # First try adding rem_elem back (to check if it was actually needed)
            # Actually, skip rem_elem, try other removed elements
            candidates2 = [x for x in removed if x != rem_elem]
            random.shuffle(candidates2)
            
            for elem in candidates2:
                if time.time() > deadline:
                    break
                new_conflicts = 0
                for tidx in elem_to_triples[elem]:
                    a, b, c = triples[tidx]
                    others = [x for x in (a, b, c) if x != elem]
                    if all(x in S for x in others):
                        new_conflicts += 1
                if new_conflicts == 0:
                    S.add(elem)
                    removed.discard(elem)
                    added_elems.append(elem)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            # Net gain: added_count - 1 (we removed 1)
            if added_count > 1:
                # Net positive: keep the change
                best_size = max(best_size, len(S))
                improved = True
                found_2opt = True
                break
            else:
                # Revert: put rem_elem back, remove added_elems
                for ae in added_elems:
                    S.discard(ae)
                    removed.add(ae)
                S.add(rem_elem)
                removed.discard(rem_elem)
        
        if not found_2opt:
            no_improve += 1
    
    return sorted(S)

def _greedy_random(triples, elem_to_triples, n_max, deadline, seed=None):
    """Greedy removal with random tie-breaking."""
    if seed is not None:
        random.seed(seed)
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        max_conf = max((elem_conflict[x] for x in S), default=0)
        if max_conf == 0:
            break
        # Random tie-breaking among max-conflict elements
        candidates = [x for x in S if elem_conflict[x] == max_conf]
        best_elem = random.choice(candidates)
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return sorted(S)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n_max = 180
    triples = _build_triples(n_max)
    
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Strategy 1: Deterministic greedy removal (highest conflict, smallest tie-break)
    best_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    best_size = len(best_set)
    
    # Strategy 2: Local search from greedy result
    if time.time() < deadline - 10.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 5.0, max_iter=3000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(best_set)
    
    # Strategy 3: Multiple random greedy restarts
    if time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            alt_set = _greedy_random(triples, elem_to_triples, n_max, deadline - 10.0, seed=restart * 12345 + 7)
            if len(alt_set) > best_size:
                best_set = alt_set
                best_size = len(alt_set)
                # Quick local search on improved set
                if time.time() < deadline - 10.0:
                    ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 10.0, max_iter=1000)
                    if len(ls_set) > best_size:
                        best_set = ls_set
                        best_size = len(best_set)
    
    # Strategy 4: Greedy with lowest-element tie-breaking
    if time.time() < deadline - 10.0:
        S = set(range(1, n_max + 1))
        triple_incount = [3] * len(triples)
        elem_conflict = {i: 0 for i in range(1, n_max + 1)}
        for idx, (a, b, c) in enumerate(triples):
            elem_conflict[a] += 1
            elem_conflict[b] += 1
            elem_conflict[c] += 1
        total_conflicts = len(triples)
        while total_conflicts > 0:
            if time.time() > deadline - 10.0:
                break
            max_conf = max((elem_conflict[x] for x in S), default=0)
            if max_conf == 0:
                break
            best_elem = min(x for x in S if elem_conflict[x] == max_conf)
            S.remove(best_elem)
            for tidx in elem_to_triples[best_elem]:
                if triple_incount[tidx] == 3:
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    for x in (a, b, c):
                        if x in S:
                            elem_conflict[x] -= 1
                triple_incount[tidx] -= 1
        alt_set = sorted(S)
        if len(alt_set) > best_size:
            best_set = alt_set
            best_size = len(best_set)
            if time.time() < deadline - 5.0:
                ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 5.0, max_iter=1000)
                if len(ls_set) > best_size:
                    best_set = ls_set
                    best_size = len(best_set)
    
    # Final local search on best_set with remaining time
    if time.time() < deadline - 2.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 1.0, max_iter=5000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(best_set)
    
    if not best_set:
        best_set = list(range(91, 181))
    
    return best_set

# EVOLVE-BLOCK-END
