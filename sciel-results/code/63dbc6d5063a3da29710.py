import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    # Precompute conflict: conflict[a][b] = set of c where (a,b,c) is bad
    conflict = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    conflict[a][b].add(c)
                    conflict[b][a].add(c)
                    conflict[a][c].add(b)
                    conflict[c][a].add(b)
                    conflict[b][c].add(a)
                    conflict[c][b].add(a)
                    bad_triples.append((a, b, c))

    # Hypergraph degree: number of bad triples each vertex is in
    degrees = [0] * (N + 1)
    for a, b, c in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1

    def is_compatible(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        sel_list = list(selected_set)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if x in conflict[a][b]:
                    return False
        return True

    def is_compatible_excl(x, selected_set, excl_elem):
        """Check if x is compatible with selected_set minus excl_elem."""
        sel_list = [s for s in selected_set if s != excl_elem]
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if x in conflict[a][b]:
                    return False
        return True

    def is_compatible_excl2(x, selected_set, excl1, excl2):
        """Check if x is compatible with selected_set minus excl1 and excl2."""
        sel_list = [s for s in selected_set if s != excl1 and s != excl2]
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if x in conflict[a][b]:
                    return False
        return True

    def greedy_build(order):
        sel = set()
        for x in order:
            if is_compatible(x, sel):
                sel.add(x)
        return sel

    def local_search(selected, time_limit):
        """Local search with 1-for-1, 2-for-1, and 2-for-2 swaps."""
        improved = True
        last_improvement = time.time()
        
        while time.time() < time_limit and improved:
            improved = False
            current_size = len(selected)
            
            # Phase 1: Try to add any compatible number
            for x in numbers:
                if x not in selected:
                    if is_compatible(x, selected):
                        selected.add(x)
                        improved = True
                        last_improvement = time.time()
                        break
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-1 swaps
            sel_list = list(selected)
            for i in range(len(sel_list)):
                if time.time() > time_limit:
                    break
                y = sel_list[i]
                temp_sel = selected - {y}
                for x in numbers:
                    if x == y or x in selected:
                        continue
                    if is_compatible_excl(x, selected, y):
                        selected = temp_sel | {x}
                        improved = True
                        last_improvement = time.time()
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Phase 3: Try 2-for-1 swaps (remove 1, add 2) - KEY IMPROVEMENT
            sel_list = list(selected)
            for i in range(len(sel_list)):
                if time.time() > time_limit:
                    break
                y = sel_list[i]
                temp_sel = selected - {y}
                
                # Find all candidates that are compatible with temp_sel
                candidates = [x for x in numbers if x not in selected and x != y and is_compatible(x, temp_sel)]
                
                # Sort by degree (lower degree first)
                candidates.sort(key=lambda x: degrees[x])
                
                # Try to find two candidates that are mutually compatible with temp_sel
                found_pair = False
                for idx1 in range(min(len(candidates), 80)):
                    x1 = candidates[idx1]
                    for idx2 in range(idx1 + 1, min(len(candidates), 80)):
                        x2 = candidates[idx2]
                        # Check if x1 and x2 are compatible with each other and temp_sel
                        if x2 in conflict[x1][y] or x1 in conflict[x2][y]:
                            continue
                        # Check x1 and x2 together with temp_sel
                        if is_compatible(x1, temp_sel) and is_compatible(x2, temp_sel | {x1}):
                            selected = temp_sel | {x1, x2}
                            improved = True
                            last_improvement = time.time()
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break
            
            if improved:
                continue
            
            # Phase 4: Try 2-for-2 swaps
            sel_list = list(selected)
            n_sel = len(sel_list)
            found_2swap = False
            pair_count = 0
            max_pairs = min(50, n_sel * (n_sel - 1) // 2)
            
            for i in range(n_sel):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n_sel):
                    if pair_count >= max_pairs:
                        break
                    pair_count += 1
                    y1 = sel_list[i]
                    y2 = sel_list[j]
                    temp_sel = selected - {y1, y2}
                    
                    candidates = [x for x in numbers if x not in selected and x != y1 and x != y2]
                    candidates.sort(key=lambda x: degrees[x])
                    candidates = candidates[:60]
                    
                    found_two = False
                    for idx1 in range(len(candidates)):
                        x1 = candidates[idx1]
                        if not is_compatible(x1, temp_sel):
                            continue
                        for idx2 in range(idx1 + 1, len(candidates)):
                            x2 = candidates[idx2]
                            if is_compatible(x2, temp_sel | {x1}):
                                selected = temp_sel | {x1, x2}
                                improved = True
                                last_improvement = time.time()
                                found_two = True
                                found_2swap = True
                                break
                        if found_two:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if improved:
                continue
            
            # Stagnation: try random perturbation
            if time.time() - last_improvement > 3:
                if len(selected) > 5:
                    perturb = set(selected)
                    to_remove = random.sample(list(perturb), min(3, len(perturb)))
                    for r in to_remove:
                        perturb.discard(r)
                    
                    # Try to rebuild
                    for x in numbers:
                        if x not in perturb and is_compatible(x, perturb):
                            perturb.add(x)
                    
                    if len(perturb) > len(selected):
                        selected = perturb
                        improved = True
                        last_improvement = time.time()
                    elif time.time() - last_improvement > 8:
                        break
        
        return selected

    # Multiple strategies
    best_selected = set()
    best_size = 0

    # Strategy 1: Greedy by degree ascending
    sorted_by_degree = sorted(numbers, key=lambda x: degrees[x])
    sel = greedy_build(sorted_by_degree)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = sel

    # Strategy 2: Greedy by value descending (larger numbers have fewer multiples)
    sorted_desc = sorted(numbers, reverse=True)
    sel = greedy_build(sorted_desc)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = sel

    # Strategy 3: Greedy by value ascending
    sel = greedy_build(numbers)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = sel

    # Strategy 4: Random restarts
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        sel = greedy_build(shuffled)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel

    # Local search on best found
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5:
        best_selected = local_search(best_selected, time.time() + remaining_time)

    # Try local search from a few different starting points
    if time.time() < deadline - 10:
        # Start from degree-ordered greedy
        sel2 = greedy_build(sorted_by_degree)
        time_left = deadline - time.time() - 3
        if time_left > 5:
            sel2 = local_search(sel2, time.time() + time_left)
            if len(sel2) > len(best_selected):
                best_selected = sel2

    # Final polish: try to add any remaining compatible elements
    for x in numbers:
        if x not in best_selected and is_compatible(x, best_selected):
            best_selected.add(x)

    return list(best_selected)
