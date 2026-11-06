"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

def _greedy_repair(chosen_set, counts, deadline):
    """Given an initial random set, greedily repair violations by removing elements."""
    if time.time() > deadline:
        return list(chosen_set)
    
    # Build sum -> list of pairs mapping
    sum_pairs = defaultdict(list)
    elems = sorted(chosen_set)
    for i, a in enumerate(elems):
        for j in range(i, len(elems)):
            b = elems[j]
            s = a + b
            sum_pairs[s].append((a, b))
    
    # Find violations (sums with > 2 representations)
    violations = [s for s, pairs in sum_pairs.items() if len(pairs) > 2]
    
    # Greedily remove elements to fix violations
    # Strategy: for each violating sum, remove the element that appears in the most pairs
    # to maximize damage reduction per removal
    while violations and time.time() < deadline:
        # Pick the most violated sum
        violations.sort(key=lambda s: -len(sum_pairs[s]))
        target_sum = violations[0]
        pairs = sum_pairs[target_sum]
        
        if len(pairs) <= 2:
            violations.remove(target_sum)
            continue
        
        # Count how many pairs each element participates in for this sum
        elem_count = defaultdict(int)
        for a, b in pairs:
            elem_count[a] += 1
            if b != a:
                elem_count[b] += 1
        
        # Also count how many other violations each element is involved in
        # (to prefer removing elements that help fix multiple violations)
        elem_other_violations = defaultdict(int)
        for s, pairs_list in sum_pairs.items():
            if s == target_sum or len(pairs_list) <= 2:
                continue
            for a, b in pairs_list:
                elem_other_violations[a] += 1
                if b != a:
                    elem_other_violations[b] += 1
        
        # Score each element: prefer removing one that appears in many pairs of target_sum
        # and also in other violations
        best_elem = None
        best_score = -1
        for elem in elem_count:
            score = elem_count[elem] * 2 + elem_other_violations.get(elem, 0)
            if score > best_score:
                best_score = score
                best_elem = elem
        
        if best_elem is None:
            break
        
        # Remove best_elem from the set
        chosen_set.discard(best_elem)
        
        # Update all sums that involved best_elem
        for other in elems:
            if other == best_elem:
                continue
            s = best_elem + other
            if s in sum_pairs:
                # Remove all pairs involving best_elem
                sum_pairs[s] = [(a, b) for a, b in sum_pairs[s] if a != best_elem and b != best_elem]
                if not sum_pairs[s]:
                    del sum_pairs[s]
        
        # Update violations list
        violations = [s for s in violations if s in sum_pairs and len(sum_pairs[s]) > 2]
        # Also check if removing created no new issues (it can't, we're only removing)
    
    return list(chosen_set)

def _try_probabilistic(p, deadline):
    """Try a probabilistic construction with probability p, then repair."""
    chosen_set = set()
    for x in range(1, 1001):
        if random.random() < p:
            chosen_set.add(x)
    
    if not chosen_set:
        return []
    
    # Build counts
    counts = defaultdict(int)
    elems = sorted(chosen_set)
    for i, a in enumerate(elems):
        for j in range(i, len(elems)):
            b = elems[j]
            s = a + b
            counts[s] += 1
    
    # Check if already valid
    if all(c <= 2 for c in counts.values()):
        return list(chosen_set)
    
    # Repair
    return _greedy_repair(set(chosen_set), counts, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, run the deterministic greedy to get a baseline
    chosen, counts = [], {}
    for x in range(1, 1001):
        new = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        for s in new:
            trial[s] = trial.get(s, 0) + 1
        if max(trial[s] for s in new) <= 2:
            chosen.append(x)
            counts = trial
    
    best_result = chosen
    best_size = len(chosen)
    
    # Try various probabilities with random restarts
    # Theoretical density ~3/2 * sqrt(N) / N... actually for B2[2], density is ~ c/sqrt(N)
    # For N=1000, we expect size around 30-50 based on known results
    # Let's try p values that give initial sets of various sizes
    
    random.seed(42)
    
    # Try a range of probabilities
    p_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    
    for p in p_values:
        if time.time() > deadline:
            break
        
        # Multiple restarts per probability
        for restart in range(5):
            if time.time() > deadline:
                break
            
            result = _try_probabilistic(p, deadline)
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
    
    # Try to improve best_result with local search: try adding elements
    if time.time() < deadline - 5:
        current = set(best_result)
        current_counts = defaultdict(int)
        elems = sorted(current)
        for i, a in enumerate(elems):
            for j in range(i, len(elems)):
                s = a + elems[j]
                current_counts[s] += 1
        
        # Try adding each unselected element
        for x in range(1, 1001):
            if time.time() > deadline - 2:
                break
            if x in current:
                continue
            
            # Check if adding x would violate
            new_sums = [x + y for y in current] + [2 * x]
            valid = True
            for s in new_sums:
                if current_counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            
            if valid:
                current.add(x)
                for s in new_sums:
                    current_counts[s] = current_counts.get(s, 0) + 1
                best_size = len(current)
                best_result = list(current)
    
    return best_result
# EVOLVE-BLOCK-END
