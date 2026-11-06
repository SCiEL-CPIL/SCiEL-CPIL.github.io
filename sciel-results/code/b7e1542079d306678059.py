"""Find large subset of [1,128] with no equal-lcm triples."""
import time
from math import lcm

_DEADLINE = 85.0

def construct():
    start_time = time.time()
    n = 128
    elems = list(range(1, n + 1))
    
    # Precompute all bad triples: (a,b,c) with a<b<c, lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == L and lcm(b, c) == L:
                    bad_triples.append((a, b, c))
    
    # Map values to indices 0..127
    val_to_idx = {v: i for i, v in enumerate(elems)}
    
    # Convert bad triples to index form
    bad_idx_triples = []
    for (a, b, c) in bad_triples:
        ia, ib, ic = val_to_idx[a], val_to_idx[b], val_to_idx[c]
        bad_idx_triples.append((ia, ib, ic))
    
    # For each element, list of (j, k) pairs such that (i, j, k) is a bad triple (i != j, i != k, j != k)
    elem_conflict_pairs = [[] for _ in range(n)]
    for (ia, ib, ic) in bad_idx_triples:
        # For ia: conflict pair is (ib, ic)
        elem_conflict_pairs[ia].append((ib, ic))
        elem_conflict_pairs[ib].append((ia, ic))
        elem_conflict_pairs[ic].append((ia, ib))
    
    # Count conflicts per element
    conflict_count = [0] * n
    for (ia, ib, ic) in bad_idx_triples:
        conflict_count[ia] += 1
        conflict_count[ib] += 1
        conflict_count[ic] += 1
    
    def greedy(order):
        current = [False] * n
        count = 0
        for i in order:
            valid = True
            for (j, k) in elem_conflict_pairs[i]:
                if current[j] and current[k]:
                    valid = False
                    break
            if valid:
                current[i] = True
                count += 1
        return count, current
    
    # Try multiple orderings
    best_count = 0
    best_current = [False] * n
    
    # Order 1: ascending conflict count
    order1 = sorted(range(n), key=lambda i: (conflict_count[i], i))
    c, cur = greedy(order1)
    if c > best_count:
        best_count = c
        best_current = cur[:]
    
    # Order 2: descending conflict count
    order2 = sorted(range(n), key=lambda i: (-conflict_count[i], i))
    c, cur = greedy(order2)
    if c > best_count:
        best_count = c
        best_current = cur[:]
    
    # Order 3: ascending value
    order3 = list(range(n))
    c, cur = greedy(order3)
    if c > best_count:
        best_count = c
        best_current = cur[:]
    
    # Order 4: descending value
    order4 = list(range(n - 1, -1, -1))
    c, cur = greedy(order4)
    if c > best_count:
        best_count = c
        best_current = cur[:]
    
    # Order 5: random-ish (use a fixed permutation)
    import random
    rng = random.Random(42)
    order5 = list(range(n))
    rng.shuffle(order5)
    c, cur = greedy(order5)
    if c > best_count:
        best_count = c
        best_current = cur[:]
    
    # Local improvement
    def local_improve(current, count, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_list = [i for i in range(n) if current[i]]
            non_cur_list = [i for i in range(n) if not current[i]]
            
            # Try single swap: remove one, add one
            for rem in cur_list:
                if time.time() >= deadline:
                    break
                for add in non_cur_list:
                    if add == rem:
                        continue
                    # Check validity of new set: (current \ {rem}) ∪ {add}
                    # For add to be valid, no (j,k) in elem_conflict_pairs[add] should both be in (current \ {rem})
                    valid = True
                    for (j, k) in elem_conflict_pairs[add]:
                        if current[j] and current[k] and j != rem and k != rem:
                            valid = False
                            break
                    if valid:
                        current[rem] = False
                        current[add] = True
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding one element
            if not improved and time.time() < deadline:
                for add in non_cur_list:
                    if time.time() >= deadline:
                        break
                    valid = True
                    for (j, k) in elem_conflict_pairs[add]:
                        if current[j] and current[k]:
                            valid = False
                            break
                    if valid:
                        current[add] = True
                        count += 1
                        improved = True
                        break
        
        return count, current
    
    deadline = min(start_time + _DEADLINE - 5, time.time() + 70)
    best_count, best_current = local_improve(best_current, best_count, deadline)
    
    # Try to add more elements greedily
    if time.time() < deadline:
        for i in range(n):
            if time.time() >= deadline:
                break
            if not best_current[i]:
                valid = True
                for (j, k) in elem_conflict_pairs[i]:
                    if best_current[j] and best_current[k]:
                        valid = False
                        break
                if valid:
                    best_current[i] = True
                    best_count += 1
    
    result = [elems[i] for i in range(n) if best_current[i]]
    if not result:
        result = [1]
    return result
