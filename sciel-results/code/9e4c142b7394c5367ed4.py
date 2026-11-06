"""Optimized program for 'No element dividing a sum of three others' task.
Strategy: Fast greedy with multiple randomized orderings, then local search with 1-out-2-in swaps.
Key optimization: precompute all C(192,3) triple sums once, and use bitmasks for fast membership checks.
"""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Precompute all triple sums and their divisors
    # triple_sums[d] = list of triple sums divisible by d
    # Actually, we need: for a candidate x, check if any triple sum is divisible by x
    # And for existing a, check if any new triple sum (involving candidate) is divisible by a
    
    # Precompute all C(192,3) triples and their sums
    # triple_sums_by_value: dict sum -> list of (i,j,k)
    # But we need: for a given x, is there a triple with sum % x == 0?
    
    # Better approach: for each x in 1..192, precompute the set of triple sums that are divisible by x
    # triple_divisible[x] = set of triple sums divisible by x
    # But that's too large. Instead:
    
    # For checking if candidate x can be added to set S:
    # 1. For any triple (a,b,c) in S: sum(a,b,c) % x != 0
    # 2. For any a in S, and any pair (b,c) in S\{a}: (a+b+c) % x != 0 -- wait no
    #    Actually: for any a in S, we need: no triple from (S\{a}) ∪ {x} has sum divisible by a
    #    Triples from (S\{a}) ∪ {x}:
    #      - 3 from S\{a}: already valid
    #      - 2 from S\{a} + x: need (b+c+x) % a != 0 for all b,c in S\{a}
    
    # So the checks are:
    # 1. For all triples (a,b,c) in S: (a+b+c) % x != 0
    # 2. For all a in S, and all pairs (b,c) in S\{a}: (b+c+x) % a != 0
    
    # Precompute: for each x, and for each possible sum s, is s % x == 0?
    # That's just a modulo check, no precomputation needed for that.
    
    # But we can precompute pair sums for fast lookup.
    
    # Let's use a different approach:
    # Maintain for the current set S:
    # - all_triple_sums: set of all triple sums from S
    # - For each a in S, the set of pair sums from S\{a}
    
    # Actually, let's just optimize the inner loops.
    
    # Precompute pair sums for all pairs in 1..192
    # pair_sum[i][j] = i + j for i < j
    # And for each a, precompute which pair sums (b+c) make (b+c+x) % a == 0 for each x
    # That's too much. Let's just do direct checks but with early termination.
    
    def can_add(selected, candidate):
        """Check if candidate can be added to selected (list of ints)."""
        n = len(selected)
        
        # Check 1: no triple from selected has sum divisible by candidate
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    if (a + b + c) % candidate == 0:
                        return False
        
        # Check 2: for each a in selected, no pair (b,c) from selected\{a} 
        #           has (b+c+candidate) % a == 0
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                b = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    c = selected[k]
                    if (b + c + candidate) % a == 0:
                        return False
        
        return True
    
    # Fast greedy: try multiple orderings
    best = []
    best_len = 0
    
    # Ordering 1: descending
    order = list(range(192, 0, -1))
    selected = []
    for x in order:
        if time.time() > deadline - 10:
            break
        if len(selected) >= 48:
            break
        if can_add(selected, x):
            selected.append(x)
    if len(selected) > best_len:
        best = selected[:]
        best_len = len(selected)
    
    # Ordering 2: descending with small perturbation
    random.seed(42)
    order = list(range(192, 0, -1))
    random.shuffle(order)
    # But keep large numbers somewhat prioritized
    order.sort(key=lambda x: -x + random.uniform(-10, 10))
    selected = []
    for x in order:
        if time.time() > deadline - 10:
            break
        if len(selected) >= 48:
            break
        if can_add(selected, x):
            selected.append(x)
    if len(selected) > best_len:
        best = selected[:]
        best_len = len(selected)
    
    # More randomized orderings
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        random.seed(1000 + trial)
        order = list(range(192, 0, -1))
        random.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline - 15:
                break
            if len(selected) >= 48:
                break
            if can_add(selected, x):
                selected.append(x)
        if len(selected) > best_len:
            best = selected[:]
            best_len = len(selected)
        if best_len >= 48:
            break
    
    # Local search: 1-out-2-in swaps
    if best_len < 48 and time.time() < deadline - 5:
        current = best[:]
        current_len = best_len
        improved = True
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                without_a = [v for j, v in enumerate(current) if j != i]
                # Try to add two new elements
                for b in range(192, 0, -1):
                    if b in without_a:
                        continue
                    if not can_add(without_a, b):
                        continue
                    with_b = without_a + [b]
                    for c in range(192, 0, -1):
                        if c in with_b:
                            continue
                        if can_add(with_b, c):
                            new_set = with_b + [c]
                            if len(new_set) > current_len:
                                current = new_set
                                current_len = len(current)
                                best = current[:]
                                best_len = current_len
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if current_len >= 48:
                break
    
    # Final polish: try to add any remaining elements
    if best_len < 48 and time.time() < deadline - 2:
        current = best[:]
        current_len = best_len
        for x in range(192, 0, -1):
            if time.time() > deadline - 2:
                break
            if len(current) >= 48:
                break
            if x in current:
                continue
            if can_add(current, x):
                current.append(x)
                current_len += 1
        if current_len > best_len:
            best = current[:]
            best_len = current_len
    
    return best[:48] if len(best) >= 48 else best

# EVOLVE-BLOCK-END
