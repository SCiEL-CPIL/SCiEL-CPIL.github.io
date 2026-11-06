import time
import random
from itertools import combinations

def _can_add(selected, x):
    """
    Check if x can be added to selected while maintaining the constraint.
    Constraint: No element a in (selected + [x]) divides the sum of any 3 distinct OTHER elements.
    """
    if not selected:
        return True
    
    # 1. Check if x divides the sum of any 3 distinct elements in selected
    # We need to check all triples from selected
    n = len(selected)
    if n >= 3:
        # Optimization: if x is large, it's unlikely to divide small sums, but we must check.
        # Sum of 3 max elements is 3 * max(selected).
        # If x > 3 * max(selected), it's safe.
        if x > 3 * max(selected):
            pass # Safe for Part 1
        else:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if (selected[i] + selected[j] + selected[k]) % x == 0:
                            return False

    # 2. Check if any a in selected divides a sum involving x
    # Sum must be x + s_i + s_j where s_i, s_j are distinct elements from selected.
    # So for each a in selected, we check if (x + s_i + s_j) % a == 0 for any distinct s_i, s_j in selected.
    
    # Optimization: Precompute residues?
    # For each a, we want to know if there exist distinct u, v in selected such that (x + u + v) % a == 0.
    # This is equivalent to (u + v) % a == (-x) % a.
    
    for a in selected:
        if n < 2:
            continue
            
        target = (-x) % a
        
        # We need distinct u, v from selected.
        # Let's collect residues of all elements in selected mod a.
        # But we need to ensure u and v are distinct elements, not just distinct residues (though if residues are same, values might be different).
        # Actually, if u % a == r1 and v % a == r2, we need r1 + r2 == target.
        # We can iterate through pairs of indices in selected to be safe, or use a set of residues if we handle duplicates carefully.
        # Given n is small (<50), iterating pairs is O(n^2) per a. Total O(n^3).
        # n=48, n^3 ~ 110,000. This is fast.
        
        # Let's try a slightly optimized pair check.
        # Create a list of residues
        residues = [s % a for s in selected]
        
        # We need i != j such that (residues[i] + residues[j]) % a == target
        # We can use a set to store seen residues.
        seen_residues = {}
        # To handle distinct elements, we should store counts or indices.
        # Since we just need existence of ANY pair, we can do:
        # For each residue r in residues:
        #   complement = (target - r) % a
        #   If complement is in seen_residues (from previous iterations), we found a pair.
        #   But we must ensure the element corresponding to complement is NOT the same as the current element.
        #   If r == complement, we need at least 2 elements with residue r.
        
        # Let's just do the O(n^2) loop for simplicity and reliability, it's fast enough.
        for i in range(n):
            for j in range(i+1, n):
                if (x + selected[i] + selected[j]) % a == 0:
                    return False
    return True

def _greedy(order):
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75.0 # Safety margin
    
    best_set = []
    
    # Strategy 1: Descending (Standard)
    order1 = list(range(192, 0, -1))
    best_set = _greedy(order1)
    if time.time() > deadline:
        return best_set
        
    # Strategy 2: Randomized Descending
    # We want to keep large numbers early, but allow some variation to escape local optima.
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        # Create a weighted order: larger numbers have higher probability of being earlier
        # Or just shuffle the bottom half.
        candidates = list(range(192, 0, -1))
        # Keep top 60 mostly ordered, shuffle the rest
        # Actually, let's just shuffle the whole thing but bias towards large numbers?
        # A simple effective heuristic: Sort by value, then randomly perturb the order slightly.
        
        # Let's try: Start with descending, but randomly swap adjacent elements with some probability?
        # Or: Generate a random permutation of 1..192, but sort it by a score = value + noise.
        
        noise = [rng.gauss(0, 10) for _ in range(192)]
        # value is 192..1
        # score = value + noise
        # Sort by score descending
        items = list(range(1, 193))
        scored = [(v, v + noise[v-1]) for v in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        order2 = [v for v, s in scored]
        
        res = _greedy(order2)
        if len(res) > len(best_set):
            best_set = res
            
    # Strategy 3: Try to add remaining elements to best_set
    # This is a local improvement step.
    best_set_set = set(best_set)
    for x in range(192, 0, -1):
        if x in best_set_set:
            continue
        if time.time() > deadline:
            break
        # Try to insert x into best_set
        # We need to check if x can be added to the current best_set
        if _can_add(best_set, x):
            best_set.append(x)
            best_set_set.add(x)
            
    return best_set
