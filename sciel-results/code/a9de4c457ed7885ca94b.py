"""Optimized search for no-element-divides-sum-of-3-others problem."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill
    
    def time_ok():
        return time.time() < deadline
    
    def check_candidate_divides_triples(cand, selected):
        """Check if cand divides sum of any 3 distinct elements in selected."""
        n = len(selected)
        # Precompute pair sums for speed
        for i in range(n):
            si = selected[i]
            for j in range(i+1, n):
                sj = selected[j]
                pair_sum = si + sj
                for k in range(j+1, n):
                    if (pair_sum + selected[k]) % cand == 0:
                        return False
        return True
    
    def check_existing_with_new(selected, new_elem):
        """Check if any existing element divides sum of 3 from (selected without it) + new_elem."""
        n = len(selected)
        for i, a in enumerate(selected):
            # Triples from (selected \ {a}) ∪ {new_elem} of size 3:
            # Case 1: 2 from selected\{a} + new_elem
            for j in range(n):
                if j == i:
                    continue
                sj = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    if (sj + selected[k] + new_elem) % a == 0:
                        return False
        return True
    
    def can_add(selected, cand):
        """Full check: can cand be added to selected?"""
        if cand in selected:
            return False
        # Check cand doesn't divide any triple sum in selected
        if not check_candidate_divides_triples(cand, selected):
            return False
        # Check existing elements don't divide new triple sums involving cand
        if not check_existing_with_new(selected, cand):
            return False
        return True
    
    def greedy(order):
        """Greedy: add elements in given order if valid."""
        selected = []
        selected_set = set()
        for x in order:
            if not time_ok():
                break
            if len(selected) >= 48:
                break
            if x in selected_set:
                continue
            if can_add(selected, x):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_search(selected, max_iters=500):
        """Try 1-out-2-in swaps to increase size."""
        best = list(selected)
        best_len = len(best)
        best_set = set(best)
        
        for _ in range(max_iters):
            if not time_ok():
                break
            if best_len >= 48:
                break
            
            improved = False
            # Try removing one element and adding two others
            for i in range(len(best)):
                if not time_ok():
                    break
                removed = best[i]
                without = [v for j, v in enumerate(best) if j != i]
                without_set = set(without)
                
                for b in range(192, 0, -1):
                    if b in without_set or b == removed:
                        continue
                    if can_add(without, b):
                        with_b = without + [b]
                        with_b_set = without_set | {b}
                        for c in range(192, 0, -1):
                            if c in with_b_set:
                                continue
                            if can_add(with_b, c):
                                new_best = with_b + [c]
                                if len(new_best) > best_len:
                                    best = new_best
                                    best_set = set(new_best)
                                    best_len = len(best)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try single swaps (remove one, add one different)
                for i in range(len(best)):
                    if not time_ok():
                        break
                    removed = best[i]
                    without = [v for j, v in enumerate(best) if j != i]
                    without_set = set(without)
                    for b in range(192, 0, -1):
                        if b in without_set or b == removed:
                            continue
                        if can_add(without, b):
                            # Same size but might enable future improvements
                            break
                break
        
        return best
    
    # Strategy 1: Greedy from high to low
    best_result = greedy(range(192, 0, -1))
    best_len = len(best_result)
    
    # Strategy 2: Greedy from low to high
    if time_ok():
        result2 = greedy(range(1, 193))
        if len(result2) > best_len:
            best_result = result2
            best_len = len(result2)
    
    # Strategy 3: Multiple randomized greedy restarts with bias toward larger numbers
    rng = random.Random(42)
    for restart in range(30):
        if not time_ok():
            break
        if best_len >= 48:
            break
        order = list(range(1, 193))
        # Weighted random order: bias toward larger numbers
        for i in range(len(order) - 1, 0, -1):
            j = rng.randint(0, i)
            order[i], order[j] = order[j], order[i]
        result = greedy(order)
        if len(result) > best_len:
            best_result = result
            best_len = len(result)
    
    # Strategy 4: Local search from best result
    if time_ok() and best_len < 48:
        improved = local_search(best_result)
        if len(improved) > best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Strategy 5: Try more randomized restarts with local search
    for restart in range(15):
        if not time_ok():
            break
        if best_len >= 48:
            break
        order = list(range(1, 193))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) < best_len:
            continue
        improved = local_search(result, max_iters=100)
        if len(improved) > best_len:
            best_result = improved
            best_len = len(improved)
    
    # Strategy 6: Final polish - try to add one more by removing one
    if time_ok() and best_len < 48:
        best_set = set(best_result)
        for i in range(len(best_result)):
            if not time_ok():
                break
            removed = best_result[i]
            without = [v for j, v in enumerate(best_result) if j != i]
            without_set = set(without)
            for b in range(192, 0, -1):
                if b in without_set or b == removed:
                    continue
                if can_add(without, b):
                    # Try to add another element too
                    with_b = without + [b]
                    with_b_set = without_set | {b}
                    for c in range(192, 0, -1):
                        if c in with_b_set:
                            continue
                        if can_add(with_b, c):
                            new_best = with_b + [c]
                            if len(new_best) > best_len:
                                best_result = new_best
                                best_len = len(best_result)
                            break
                    break
    
    # Final: if we have more than 48, truncate; otherwise return as-is
    if len(best_result) > 48:
        return best_result[:48]
    return best_result

# EVOLVE-BLOCK-END
