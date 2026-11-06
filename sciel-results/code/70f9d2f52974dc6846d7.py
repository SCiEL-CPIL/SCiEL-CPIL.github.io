"""
Maximize cardinality of S ⊆ [1,256] such that for all a<b≤c in S, a ∤ (b+c).

Strategy:
1. Start with the known valid interval [171, 256] (size 86), guaranteed by 2*max < 3*min.
2. Try to extend downward: for each candidate k from 170 down to 1,
   greedily remove minimal elements from current set to make k admissible,
   and accept the change if net size doesn't decrease (prefer increase).
3. Time-bounded local search: random swaps and additions to find larger sets.
"""

import time
import random

def _check(a, b, c):
    """Return True if a divides b+c (violation), False otherwise."""
    return (b + c) % a == 0

def _is_valid(S):
    """Check if set S satisfies the property: for all a<b<=c in S, a !| (b+c)."""
    elems = sorted(S)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i+1, n):
            b = elems[j]
            for k in range(j, n):
                c = elems[k]
                if (b + c) % a == 0:
                    return False
    return True

def _violations_for_a(a, S):
    """
    Given a candidate element a to add to S, find all pairs (b,c) in S with b<=c, b>=a (actually b>a since a not in S yet,
    but we consider b,c in S) such that a | (b+c).
    Returns list of (b,c) pairs.
    """
    viol = []
    for b in S:
        if b < a:
            continue
        for c in S:
            if c < b:
                continue
            if (b + c) % a == 0:
                viol.append((b, c))
    return viol

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # leave 10s margin for 60s limit

    # Step 1: Start with [171, 256]
    S = set(range(171, 257))
    best_size = len(S)
    
    # Step 2: Greedy downward extension
    for k in range(170, 0, -1):
        if time.time() > deadline:
            break
        if k in S:
            continue
        
        # Find violations if we add k
        viol = _violations_for_a(k, S)
        if not viol:
            S.add(k)
            best_size = len(S)
            continue
        
        # Need to remove elements to resolve violations.
        # Each violation (b,c) means we need to remove b or c.
        # Use greedy: repeatedly pick the element involved in most violations, remove it.
        S_work = set(S)
        S_work.add(k)
        changed = True
        while changed and len(S_work) > 0:
            changed = False
            # Recompute violations for a=k in S_work (excluding k itself from b,c since a<b)
            elems = sorted(x for x in S_work if x > k)
            # Count how many times each element appears in violations
            from collections import Counter
            cnt = Counter()
            for b in elems:
                for c in elems:
                    if c < b:
                        continue
                    if (b + c) % k == 0:
                        cnt[b] += 1
                        cnt[c] += 1
            if not cnt:
                break
            # Remove the most frequently involved element
            worst = max(cnt, key=lambda x: cnt[x])
            S_work.remove(worst)
            changed = True
        
        if len(S_work) >= len(S):
            S = S_work
            if len(S) > best_size:
                best_size = len(S)
    
    # Step 3: Local search for improvements
    random.seed(42)
    current_size = len(S)
    
    while time.time() < deadline:
        # Try adding a random element from 1..256 not in S
        candidates = [x for x in range(1, 257) if x not in S]
        if not candidates:
            break
        k = random.choice(candidates)
        S_work = set(S)
        S_work.add(k)
        
        # Resolve violations by greedy removal
        changed = True
        while changed:
            changed = False
            elems = sorted(x for x in S_work if x > k)
            from collections import Counter
            cnt = Counter()
            for b in elems:
                for c in elems:
                    if c < b:
                        continue
                    if (b + c) % k == 0:
                        cnt[b] += 1
                        cnt[c] += 1
            if not cnt:
                break
            worst = max(cnt, key=lambda x: cnt[x])
            S_work.remove(worst)
            changed = True
        
        if len(S_work) > len(S):
            S = S_work
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
        
        # Also try removing a random element and adding two smaller ones
        if len(S) > 2:
            to_remove = random.choice(sorted(S))
            S_work = set(S)
            S_work.discard(to_remove)
            
            # Try to add one or two elements from 1..to_remove-1
            candidates = [x for x in range(1, to_remove) if x not in S_work]
            random.shuffle(candidates)
            for k in candidates[:3]:
                S_temp = set(S_work)
                S_temp.add(k)
                # Check validity quickly
                valid = True
                for a in sorted(S_temp):
                    for b in S_temp:
                        if b <= a:
                            continue
                        for c in S_temp:
                            if c < b:
                                continue
                            if (b + c) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid and len(S_temp) > len(S):
                    S = S_temp
                    break
    
    return sorted(S)
