import time
from math import gcd
from collections import defaultdict
import random
import sys

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    random.seed(42)
    
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    num_triples = len(bad_triples)
    
    # For each element, list of triple indices it participates in
    elem_to_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(i)
        elem_to_triples[b].append(i)
        elem_to_triples[c].append(i)
    
    # Build conflict structure: for each pair of elements, list of triples containing both
    # This helps with efficient violation checking
    pair_conflicts = defaultdict(list)  # (x,y) -> list of triple indices containing both x and y
    for i, (a, b, c) in enumerate(bad_triples):
        pair_conflicts[(a, b)].append(i)
        pair_conflicts[(b, a)].append(i)
        pair_conflicts[(a, c)].append(i)
        pair_conflicts[(c, a)].append(i)
        pair_conflicts[(b, c)].append(i)
        pair_conflicts[(c, b)].append(i)
    
    # Greedy initial solution: start with all, remove to break triples
    selected = [True] * (n + 1)
    active = set(range(num_triples))
    
    while active:
        if time.time() > deadline - 40:
            break
        best_x = None
        best_cov = 0
        for x in range(1, n+1):
            if not selected[x]:
                continue
            cov = 0
            for ti in elem_to_triples[x]:
                if ti in active:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None or best_cov == 0:
            break
        selected[best_x] = False
        to_remove = set()
        for ti in elem_to_triples[best_x]:
            if ti in active:
                to_remove.add(ti)
        active -= to_remove
    
    def is_valid(sel):
        for (a, b, c) in bad_triples:
            if sel[a] and sel[b] and sel[c]:
                return False
        return True
    
    assert is_valid(selected), "Greedy failed"
    
    best_size = sum(1 for x in range(1, n+1) if selected[x])
    best_sel = selected[:]
    
    # Phase 1: Greedy additions
    changed = True
    while changed and time.time() < deadline - 35:
        changed = False
        for x in range(1, n+1):
            if selected[x]:
                continue
            # Check if adding x creates any violation
            can_add = True
            for ti in elem_to_triples[x]:
                a, b, c = bad_triples[ti]
                others = [y for y in (a, b, c) if y != x]
                if selected[others[0]] and selected[others[1]]:
                    can_add = False
                    break
            if can_add:
                selected[x] = True
                best_size = sum(1 for x2 in range(1, n+1) if selected[x2])
                changed = True
    
    # Phase 2: Branch and bound with multiple orderings
    # We'll try several different element orderings for the B&B
    
    def add_violations(sel, x):
        """Count how many bad triples would be violated if x is added to sel."""
        cnt = 0
        for ti in elem_to_triples[x]:
            a, b, c = bad_triples[ti]
            others = [y for y in (a, b, c) if y != x]
            if sel[others[0]] and sel[others[1]]:
                cnt += 1
        return cnt
    
    def branch_and_bound(start_order, deadline_bnb):
        """Branch and bound on a given ordering of elements."""
        nonlocal best_size, best_sel
        
        # Elements in the given order
        order = list(start_order)
        # Elements already selected
        selected_bnb = best_sel[:]
        current_size = best_size
        
        # For B&B, we process elements in order, deciding include/exclude
        # Upper bound: current_size + number of remaining elements that could be added
        # (optimistic: all remaining can be added)
        
        def rec(idx, sel, size):
            nonlocal best_size, best_sel
            
            if time.time() > deadline_bnb:
                return
            
            if idx == len(order):
                if size > best_size:
                    best_size = size
                    best_sel = sel[:]
                return
            
            x = order[idx]
            
            # Upper bound: size + (len(order) - idx)
            if size + (len(order) - idx) <= best_size:
                return
            
            # Option 1: Exclude x
            rec(idx + 1, sel, size)
            
            # Option 2: Include x (if valid)
            if add_violations(sel, x) == 0:
                sel[x] = True
                rec(idx + 1, sel, size + 1)
                sel[x] = False
        
        # Start B&B
        # Only consider elements not yet in best_sel for the "free" choices
        # Elements in best_sel are fixed as included
        # Elements not in best_sel are candidates for inclusion/exclusion
        
        # Actually, let's do a cleaner B&B:
        # Start from scratch with a different strategy
        pass
    
    # Better approach: Use a maximum independent set style B&B
    # The hypergraph has vertices 1..128 and hyperedges (bad triples)
    # We want max subset with no hyperedge fully contained
    
    # Let's use a different structural approach:
    # Group elements by their "type" based on prime factorization patterns
    # Elements that are pairwise coprime or have specific LCM relationships
    
    # Actually, let's try a smarter local search with restarts
    
    # Phase 3: Multi-start local search with diverse initial solutions
    def greedy_solution(seed_val):
        """Generate a greedy solution with a given seed."""
        rng = random.Random(seed_val)
        sel = [True] * (n + 1)
        act = set(range(num_triples))
        
        # Randomize the order of processing
        while act:
            if time.time() > deadline - 10:
                break
            # Pick element with max coverage, with some randomness
            candidates = []
            for x in range(1, n+1):
                if not sel[x]:
                    continue
                cov = 0
                for ti in elem_to_triples[x]:
                    if ti in act:
                        cov += 1
                candidates.append((cov, x))
            
            if not candidates:
                break
            
            candidates.sort(reverse=True)
            # Pick from top few with randomness
            top_k = min(5, len(candidates))
            best_cov, best_x = rng.choice(candidates[:top_k])
            
            if best_cov == 0:
                break
            
            sel[best_x] = False
            to_remove = set()
            for ti in elem_to_triples[best_x]:
                if ti in act:
                    to_remove.add(ti)
            act -= to_remove
        
        # Greedy additions
        changed = True
        while changed:
            changed = False
            for x in range(1, n+1):
                if sel[x]:
                    continue
                can_add = True
                for ti in elem_to_triples[x]:
                    a, b, c = bad_triples[ti]
                    others = [y for y in (a, b, c) if y != x]
                    if sel[others[0]] and sel[others[1]]:
                        can_add = False
                        break
                if can_add:
                    sel[x] = True
                    changed = True
        
        return sel
    
    def local_search(sel, deadline_ls):
        """Local search: try swaps and 2-for-1 exchanges."""
        nonlocal best_size, best_sel
        
        cur_sel = sel[:]
        cur_size = sum(1 for x in range(1, n+1) if cur_sel[x])
        
        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            
            # Try single additions
            for x in range(1, n+1):
                if cur_sel[x]:
                    continue
                if add_violations(cur_sel, x) == 0:
                    cur_sel[x] = True
                    cur_size += 1
                    improved = True
            
            # Try swaps: remove one, add one
            unselected = [x for x in range(1, n+1) if not cur_sel[x]]
            for x_out in range(1, n+1):
                if not cur_sel[x_out]:
                    continue
                for x_in in unselected:
                    # Check if removing x_out and adding x_in is valid
                    # Temporarily remove x_out
                    cur_sel[x_out] = False
                    if add_violations(cur_sel, x_in) == 0:
                        cur_sel[x_in] = True
                        cur_size += 0  # size unchanged
                        # Now try to add more
                        for x2 in range(1, n+1):
                            if not cur_sel[x2] and x2 != x_out:
                                if add_violations(cur_sel, x2) == 0:
                                    cur_sel[x2] = True
                                    cur_size += 1
                                    improved = True
                        if cur_size > best_size:
                            best_size = cur_size
                            best_sel = cur_sel[:]
                    else:
                        cur_sel[x_out] = True
            
            # Try 2-for-1: remove two, add three
            # This is expensive, so only do it occasionally
            if time.time() < deadline_ls - 5:
                selected_list = [x for x in range(1, n+1) if cur_sel[x]]
                for i in range(min(10, len(selected_list))):
                    x_out1 = selected_list[i]
                    for j in range(i+1, min(10, len(selected_list))):
                        x_out2 = selected_list[j]
                        cur_sel[x_out1] = False
                        cur_sel[x_out2] = False
                        added = 0
                        for x_in in range(1, n+1):
                            if not cur_sel[x_in] and x_in != x_out1 and x_in != x_out2:
                                if add_violations(cur_sel, x_in) == 0:
                                    cur_sel[x_in] = True
                                    added += 1
                        if added > 2:
                            cur_size = sum(1 for x in range(1, n+1) if cur_sel[x])
                            if cur_size > best_size:
                                best_size = cur_size
                                best_sel = cur_sel[:]
                            improved = True
                        else:
                            # Revert
                            for x_in in range(1, n+1):
                                if cur_sel[x_in] and x_in != x_out1 and x_in != x_out2:
                                    # Check if this was added
                                    pass
                            cur_sel[x_out1] = True
                            cur_sel[x_out2] = True
                            # Revert additions - complex, skip for now
        
        return cur_sel
    
    # Multi-start
    num_starts = 20
    time_per_start = (deadline - 15 - time.time()) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        seed_val = 1000 + s * 137
        start_sel = greedy_solution(seed_val)
        start_size = sum(1 for x in range(1, n+1) if start_sel[x])
        
        if start_size > best_size:
            best_size = start_size
            best_sel = start_sel[:]
        
        # Local search from this start
        ls_deadline = min(deadline - 15, time.time() + max(1, time_per_start))
        local_search(start_sel, ls_deadline)
    
    # Final greedy additions
    for x in range(1, n+1):
        if not best_sel[x]:
            if add_violations(best_sel, x) == 0:
                best_sel[x] = True
    
    final_size = sum(1 for x in range(1, n+1) if best_sel[x])
    result = [x for x in range(1, n+1) if best_sel[x]]
    if not result:
        result = [1]
    
    return result
