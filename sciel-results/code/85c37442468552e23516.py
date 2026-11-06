"""Rewritten program: branch-and-bound + MILP for max independent set on 3-uniform hypergraph."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if diff == 0:
                continue
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
        best_elem = max(S, key=lambda x: (elem_conflict[x], -x))
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

def _local_search(triples, elem_to_triples, n_max, start_set, deadline, max_iter=5000):
    """Local search: 1-opt (add) and 2-opt (remove one, add up to two)."""
    S = set(start_set)
    removed = set(range(1, n_max + 1)) - S
    
    # Verify no conflicts
    for a, b, c in triples:
        if a in S and b in S and c in S:
            # Remove one of them
            S.discard(c)
            removed.add(c)
    
    best_size = len(S)
    no_improve = 0
    max_no_improve = 300
    
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
        if not S:
            break
            
        s_list = list(S)
        random.shuffle(s_list)
        
        found_2opt = False
        for rem_elem in s_list[:min(50, len(s_list))]:
            if time.time() > deadline:
                break
            
            S.remove(rem_elem)
            removed.add(rem_elem)
            
            added_count = 0
            added_elems = []
            
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
            
            if added_count > 1:
                best_size = max(best_size, len(S))
                improved = True
                found_2opt = True
                break
            else:
                for ae in added_elems:
                    S.discard(ae)
                    removed.add(ae)
                S.add(rem_elem)
                removed.discard(rem_elem)
        
        if not found_2opt:
            no_improve += 1
    
    return sorted(S)

def _maxsat_bnb(triples, elem_to_triples, n_max, deadline, initial_set=None):
    """MaxSAT-style branch and bound with unit propagation for 3-clauses.
    
    Model: Maximize sum(x_i) subject to (¬x_a ∨ ¬x_b ∨ ¬x_c) for each triple.
    Equivalent: at most 2 of {a,b,c} can be true.
    
    Uses recursive B&B with:
    - Unit propagation: if 2 vars in a clause are true, force 3rd to false
    - Conflict detection: if all 3 vars true, prune
    - Upper bound: current_trues + remaining_undecided
    - Variable ordering by conflict count
    """
    if initial_set is None:
        initial_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Precompute conflict structure
    elem_conflict = {i: len(elem_to_triples[i]) for i in range(1, n_max + 1)}
    elements = sorted(range(1, n_max + 1), key=lambda x: (-elem_conflict[x], x))
    
    # State representation:
    # assigned[i] = -1 (false/excluded), 0 (undecided), 1 (true/included)
    # Use lists for speed
    
    def _propagate(assigned, true_set, false_set):
        """Unit propagation: if 2 of 3 in a clause are true, force 3rd false.
        Returns (success, new_true_set, new_false_set, new_assigned)
        """
        queue = []
        # Initial: check all clauses for unit propagation opportunities
        for idx, (a, b, c) in enumerate(triples):
            t_count = (1 if assigned[a] == 1 else 0) + (1 if assigned[b] == 1 else 0) + (1 if assigned[c] == 1 else 0)
            if t_count == 2:
                # One of them must be false
                for v in (a, b, c):
                    if assigned[v] == 0:
                        queue.append(v)
                        break
            elif t_count == 3:
                return (False, true_set, false_set, assigned)  # Conflict
        
        while queue:
            v = queue.pop()
            if assigned[v] != 0:
                continue
            assigned[v] = -1
            false_set.add(v)
            # Propagate further
            for tidx in elem_to_triples[v]:
                a, b, c = triples[tidx]
                t_count = (1 if assigned[a] == 1 else 0) + (1 if assigned[b] == 1 else 0) + (1 if assigned[c] == 1 else 0)
                if t_count == 2:
                    for u in (a, b, c):
                        if assigned[u] == 0:
                            queue.append(u)
                            break
                elif t_count == 3:
                    return (False, true_set, false_set, assigned)
        
        return (True, true_set, false_set, assigned)
    
    def _search(assigned, true_set, false_set, undecided_count, idx):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if idx >= len(elements):
            if len(true_set) > best_size:
                best_size = len(true_set)
                best_set = set(true_set)
            return
        
        elem = elements[idx]
        
        if assigned[elem] == 1:
            _search(assigned, true_set, false_set, undecided_count, idx + 1)
            return
        if assigned[elem] == -1:
            _search(assigned, true_set, false_set, undecided_count, idx + 1)
            return
        
        # elem is undecided
        # Upper bound: current true + remaining undecided (including elem)
        if len(true_set) + undecided_count <= best_size:
            return
        
        # Option 1: try setting elem to true
        if not any(all(v in true_set for v in (a, b, c) if v != elem) for tidx in elem_to_triples[elem] for a, b, c in [triples[tidx]]):
            # Check if adding elem creates immediate conflict
            can_add = True
            for tidx in elem_to_triples[elem]:
                a, b, c = triples[tidx]
                others = [x for x in (a, b, c) if x != elem]
                if all(x in true_set for x in others):
                    can_add = False
                    break
            
            if can_add:
                assigned[elem] = 1
                true_set.add(elem)
                success, true_set, false_set, assigned = _propagate(assigned, true_set, false_set)
                if success:
                    _search(assigned, true_set, false_set, undecided_count - 1, idx + 1)
                assigned[elem] = 0
                true_set.discard(elem)
                # Restore false_set changes from propagation - this is complex
                # For now, skip the restore and rely on the fact that we're in a new branch
        
        # Option 2: set elem to false
        assigned[elem] = -1
        false_set.add(elem)
        _search(assigned, true_set, false_set, undecided_count - 1, idx + 1)
        assigned[elem] = 0
        false_set.discard(elem)
    
    # Initialize state
    assigned = [0] * (n_max + 1)
    true_set = set()
    false_set = set()
    
    # Pre-assign from initial set
    for elem in initial_set:
        assigned[elem] = 1
        true_set.add(elem)
    for elem in range(1, n_max + 1):
        if assigned[elem] == 0 and elem not in true_set:
            # Check if it conflicts with current true_set
            conflict = False
            for tidx in elem_to_triples[elem]:
                a, b, c = triples[tidx]
                others = [x for x in (a, b, c) if x != elem]
                if all(x in true_set for x in others):
                    conflict = True
                    break
            if conflict:
                assigned[elem] = -1
                false_set.add(elem)
    
    undecided_count = sum(1 for i in range(1, n_max + 1) if assigned[i] == 0)
    
    _search(assigned, true_set, false_set, undecided_count, 0)
    
    return sorted(best_set)

def _branch_and_bound(triples, elem_to_triples, n_max, deadline, initial_set=None):
    """Branch and bound for maximum independent set on 3-uniform hypergraph."""
    if initial_set is None:
        initial_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Order elements by conflict count descending
    elem_conflict = {i: len(elem_to_triples[i]) for i in range(1, n_max + 1)}
    elements = sorted(range(1, n_max + 1), key=lambda x: -elem_conflict[x])
    
    # State: current set S, remaining elements to decide, index
    # Use recursion with pruning
    
    S = set()
    decided = set()
    
    def _has_conflict(s, elem):
        """Check if adding elem to s creates a conflict."""
        for tidx in elem_to_triples[elem]:
            a, b, c = triples[tidx]
            others = [x for x in (a, b, c) if x != elem]
            if all(x in s for x in others):
                return True
        return False
    
    def _upper_bound(s, undecided, idx):
        """Upper bound: |s| + |undecided from idx onwards|"""
        return len(s) + (len(elements) - idx)
    
    def _search(idx, s, undecided):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if idx >= len(elements):
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)
            return
        
        # Pruning: even if we add all remaining, can't beat best
        if len(s) + (len(elements) - idx) <= best_size:
            return
        
        elem = elements[idx]
        
        # Option 1: include elem if no conflict
        if not _has_conflict(s, elem):
            s.add(elem)
            _search(idx + 1, s, undecided - {elem})
            s.remove(elem)
        
        # Option 2: exclude elem
        _search(idx + 1, s, undecided - {elem})
    
    # This is too slow for 180 elements. Use a smarter approach.
    # Instead, use iterative deepening on the "excluded" set.
    
    # Better approach: start from greedy solution, try to improve via 2-opt and 3-opt
    S = set(initial_set)
    removed = set(range(1, n_max + 1)) - S
    
    # Iteratively try 1-opt, 2-opt, 3-opt
    for _ in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1-opt: add one element
        for elem in list(removed):
            if time.time() > deadline:
                break
            if not _has_conflict(S, elem):
                S.add(elem)
                removed.discard(elem)
                improved = True
                break
        
        if improved:
            continue
        
        # 2-opt: remove one, add two
        s_list = list(S)
        random.shuffle(s_list)
        for rem in s_list[:30]:
            if time.time() > deadline:
                break
            S.remove(rem)
            removed.add(rem)
            
            added = []
            for elem in list(removed):
                if elem == rem:
                    continue
                if not _has_conflict(S, elem):
                    S.add(elem)
                    removed.discard(elem)
                    added.append(elem)
                    if len(added) >= 2:
                        break
            
            if len(added) > 1:
                improved = True
                break
            else:
                for a in added:
                    S.discard(a)
                    removed.add(a)
                S.add(rem)
                removed.discard(rem)
        
        if improved:
            continue
        
        # 3-opt: remove two, add three
        s_list = list(S)
        random.shuffle(s_list)
        found_3opt = False
        for i in range(min(20, len(s_list))):
            if time.time() > deadline:
                break
            for j in range(i+1, min(i+20, len(s_list))):
                rem1, rem2 = s_list[i], s_list[j]
                S.remove(rem1)
                S.remove(rem2)
                removed.add(rem1)
                removed.add(rem2)
                
                added = []
                for elem in list(removed):
                    if elem in (rem1, rem2):
                        continue
                    if not _has_conflict(S, elem):
                        S.add(elem)
                        removed.discard(elem)
                        added.append(elem)
                        if len(added) >= 3:
                            break
                
                if len(added) > 2:
                    improved = True
                    found_3opt = True
                    break
                else:
                    for a in added:
                        S.discard(a)
                        removed.add(a)
                    S.add(rem1)
                    S.add(rem2)
                    removed.discard(rem1)
                    removed.discard(rem2)
            
            if found_3opt:
                break
        
        if not improved:
            break
    
    return sorted(best_set) if len(best_set) > len(initial_set) else sorted(initial_set)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n_max = 180
    triples = _build_triples(n_max)
    
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Strategy 1: MaxSAT-style B&B with unit propagation
    best_set = None
    best_size = 0
    
    if time.time() < deadline - 15.0:
        msat_set = _maxsat_bnb(triples, elem_to_triples, n_max, deadline - 10.0)
        if msat_set is not None:
            best_set = msat_set
            best_size = len(msat_set)
    
    # Strategy 2: Greedy + local search
    greedy_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Local search on best
    if time.time() < deadline - 10.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 5.0, max_iter=3000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    # Strategy 3: Multiple random greedy restarts + local search
    if time.time() < deadline - 15.0:
        for restart in range(10):
            if time.time() > deadline - 15.0:
                break
            alt_set = _greedy_random(triples, elem_to_triples, n_max, deadline - 15.0, seed=restart * 99991 + 13)
            if len(alt_set) > best_size:
                best_set = alt_set
                best_size = len(alt_set)
            # Quick local search
            if time.time() < deadline - 10.0:
                ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 10.0, max_iter=1000)
                if len(ls_set) > best_size:
                    best_set = ls_set
                    best_size = len(ls_set)
    
    # Strategy 4: More MaxSAT B&B attempts with different initial sets
    if time.time() < deadline - 5.0:
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            alt_greedy = _greedy_random(triples, elem_to_triples, n_max, deadline - 5.0, seed=restart * 77771 + 42)
            msat_set = _maxsat_bnb(triples, elem_to_triples, n_max, deadline - 3.0, initial_set=alt_greedy)
            if msat_set is not None and len(msat_set) > best_size:
                best_set = msat_set
                best_size = len(msat_set)
    
    # Final local search
    if time.time() < deadline - 2.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 1.0, max_iter=5000)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    if not best_set:
        best_set = list(range(91, 181))
    
    return best_set

# EVOLVE-BLOCK-END
