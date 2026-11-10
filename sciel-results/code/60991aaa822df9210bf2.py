import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM matrix
    lcm_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_mat[i][j] = _lcm(i, j)
            lcm_mat[j][i] = lcm_mat[i][j]
    
    # Build conflict structure: for each element x, conflict_pairs[x] = set of frozenset({a,b})
    # where {a,b,x} forms a bad triple (all three pairwise LCMs equal)
    conflict_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_mat[a][b]
            for c in range(b + 1, n + 1):
                if lcm_mat[a][c] == lab and lcm_mat[b][c] == lab:
                    conflict_pairs[a].add(frozenset((b, c)))
                    conflict_pairs[b].add(frozenset((a, c)))
                    conflict_pairs[c].add(frozenset((a, b)))
    
    def count_conflicts(x, selected_set):
        """Count how many bad triples would be formed if x were added to selected_set."""
        count = 0
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                count += 1
        return count
    
    def count_conflicts_excluding(x, selected_set, excluded_y):
        """Count conflicts for x in selected_set, ignoring any pair involving excluded_y."""
        count = 0
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                if a != excluded_y and b != excluded_y:
                    count += 1
        return count
    
    def is_valid(selected_set):
        """Check if the set has no bad triples."""
        for x in selected_set:
            if count_conflicts(x, selected_set - {x}) > 0:
                return False
        return True
    
    # Greedy add function
    def greedy_add(order):
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        return selected
    
    # Try multiple deterministic orders
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        sorted(range(1, n + 1), key=lambda x: len(conflict_pairs[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict_pairs[x])),
        # Focus on upper range which tends to have fewer conflicts
        list(range(65, 129)) + list(range(1, 65)),
        sorted(range(1, n + 1), key=lambda x: (x < 65, x)),
    ]
    
    best_result = set()
    
    for order in orders:
        if time.time() > deadline - 20:
            break
        selected = greedy_add(order)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Randomized greedy restarts
    rng = random.Random(42)
    for restart in range(40):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = greedy_add(order)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Local search with multiple phases
    current = best_result.copy()
    
    # Phase 1: Try to add any element that fits (multiple passes)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    # Phase 2: 1-for-1 Swaps
    swap_attempts = 0
    max_swap_attempts = 15000
    while time.time() < deadline - 10 and swap_attempts < max_swap_attempts:
        swap_found = False
        for x in range(1, n + 1):
            if x in current:
                continue
            for y in list(current):
                swap_attempts += 1
                if swap_attempts > max_swap_attempts or time.time() > deadline - 10:
                    break
                if count_conflicts_excluding(x, current, y) == 0:
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found or swap_attempts > max_swap_attempts:
                break
        if not swap_found:
            break
    
    # Phase 3: 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 8:
        improved_1for2 = True
        while improved_1for2 and time.time() < deadline - 8:
            improved_1for2 = False
            for y in list(current):
                if time.time() > deadline - 8:
                    break
                temp = current - {y}
                candidates = [x for x in range(1, n + 1) if x not in temp]
                for i, x1 in enumerate(candidates):
                    if time.time() > deadline - 8:
                        break
                    if count_conflicts(x1, temp) == 0:
                        temp_with_x1 = temp | {x1}
                        for x2 in candidates[i+1:]:
                            if time.time() > deadline - 8:
                                break
                            if count_conflicts(x2, temp_with_x1) == 0:
                                current = temp_with_x1 | {x2}
                                improved_1for2 = True
                                break
                    if improved_1for2:
                        break
                if improved_1for2:
                    break
    
    # Phase 4: 2-for-3 swaps (remove 2, add 3) - key innovation for escaping local optima
    if time.time() < deadline - 5:
        improved_2for3 = True
        while improved_2for3 and time.time() < deadline - 5:
            improved_2for3 = False
            current_list = list(current)
            for i, y1 in enumerate(current_list):
                if time.time() > deadline - 5:
                    break
                for y2 in current_list[i+1:]:
                    if time.time() > deadline - 5:
                        break
                    temp = current - {y1, y2}
                    candidates = [x for x in range(1, n + 1) if x not in temp]
                    found_triple = False
                    for a_idx, x1 in enumerate(candidates):
                        if time.time() > deadline - 5:
                            break
                        if count_conflicts(x1, temp) == 0:
                            temp_with_x1 = temp | {x1}
                            for b_idx, x2 in enumerate(candidates[a_idx+1:], start=a_idx+1):
                                if time.time() > deadline - 5:
                                    break
                                if count_conflicts(x2, temp_with_x1) == 0:
                                    temp_with_x1x2 = temp_with_x1 | {x2}
                                    for x3 in candidates[b_idx+1:]:
                                        if time.time() > deadline - 5:
                                            break
                                        if count_conflicts(x3, temp_with_x1x2) == 0:
                                            current = temp_with_x1x2 | {x3}
                                            improved_2for3 = True
                                            found_triple = True
                                            break
                                if found_triple:
                                    break
                        if found_triple:
                            break
                    if found_triple:
                        break
                if improved_2for3:
                    break
    
    # Phase 5: Kicks - random 2-swap moves to escape local optima, then re-optimize
    if time.time() < deadline - 3:
        for kick_round in range(5):
            if time.time() > deadline - 3:
                break
            # Perform a random 2-element swap (remove 2, add 2 random candidates)
            if len(current) >= 2:
                y1, y2 = rng.sample(list(current), 2)
                temp = current - {y1, y2}
                candidates = [x for x in range(1, n + 1) if x not in temp]
                if len(candidates) >= 2:
                    x1, x2 = rng.sample(candidates, 2)
                    # Only accept if it doesn't increase conflicts too much
                    new_set = temp | {x1, x2}
                    if len(new_set) == len(current):
                        current = new_set
            
            # Re-optimize after kick
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for x in range(1, n + 1):
                    if x in current:
                        continue
                    if count_conflicts(x, current) == 0:
                        current.add(x)
                        improved = True
                        break
            
            # Quick 1-for-1 swap pass
            for x in range(1, n + 1):
                if x in current or time.time() > deadline - 3:
                    continue
                for y in list(current):
                    if count_conflicts_excluding(x, current, y) == 0:
                        current.remove(y)
                        current.add(x)
                        break
    
    # Phase 6: Final add pass
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    # Validate final result
    if not is_valid(current):
        # Fallback to best valid result found
        current = best_result
    
    return list(current) if current else list(best_result)
