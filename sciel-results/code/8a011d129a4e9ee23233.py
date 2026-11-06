import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    N = 256
    # Perfect squares up to 2*N
    max_sum = 2 * N
    squares = set()
    k = 2
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Build conflict: conflicts[i] = set of j where i+j is a perfect square
    # Also include diagonal: 2*i is a perfect square
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):  # j >= i, a<=b
            if i + j in squares:
                conflicts[i].add(j)
                if j != i:
                    conflicts[j].add(i)
    
    # Diagonal constraint: 2*i must not be a perfect square
    diagonal_bad = set()
    for i in range(1, N + 1):
        if 2 * i in squares:
            diagonal_bad.add(i)
    
    # Greedy: sort by degree ascending, add if compatible
    def greedy_construct(order):
        selected = set()
        for i in order:
            if i in diagonal_bad:
                continue
            # Check if i conflicts with any in selected
            ok = True
            for s in selected:
                if s in conflicts[i]:
                    ok = False
                    break
            if ok:
                selected.add(i)
        return selected
    
    # Try multiple orderings
    all_numbers = list(range(1, N + 1))
    
    # Order 1: by degree ascending
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    order1 = sorted(all_numbers, key=lambda x: degrees[x])
    best_set = greedy_construct(order1)
    
    # Order 2: reverse
    order2 = sorted(all_numbers, key=lambda x: -degrees[x])
    s2 = greedy_construct(order2)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Local search: try to add elements, then try 1-for-2 swaps
    def local_search(s):
        s = set(s)
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            # Try adding any element not in s
            for i in range(1, N + 1):
                if i in s or i in diagonal_bad:
                    continue
                # Check if i can be added
                ok = True
                for x in s:
                    if x in conflicts[i]:
                        ok = False
                        break
                if ok:
                    s.add(i)
                    improved = True
                    break  # restart scan after modification
            
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Try 1-for-2: remove one element, see if we can add two
            for rem in list(s):
                s2 = s - {rem}
                # Find elements that can now be added
                candidates = []
                for i in range(1, N + 1):
                    if i in s2 or i in diagonal_bad:
                        continue
                    ok = True
                    for x in s2:
                        if x in conflicts[i]:
                            ok = False
                            break
                    if ok:
                        candidates.append(i)
                
                # Try to add two from candidates
                best_gain = 0
                best_pair = None
                for idx_a in range(len(candidates)):
                    ca = candidates[idx_a]
                    # Check ca doesn't conflict with s2
                    # Already checked above
                    for idx_b in range(idx_a + 1, len(candidates)):
                        cb = candidates[idx_b]
                        # Check ca and cb don't conflict with each other
                        if cb in conflicts[ca]:
                            continue
                        # Check both don't conflict with s2
                        ok_a = all(x not in conflicts[ca] for x in s2)
                        ok_b = all(x not in conflicts[cb] for x in s2)
                        if ok_a and ok_b:
                            best_gain = 2
                            best_pair = (ca, cb)
                            break
                    if best_gain == 2:
                        break
                if best_gain == 2:
                    s2.add(best_pair[0])
                    s2.add(best_pair[1])
                    s = s2
                    improved = True
                    break
        
        return s
    
    best_set = local_search(best_set)
    
    # Try local search from different starting points
    for order in [order2, sorted(all_numbers, key=lambda x: x % 7)]:
        if (time.time() - start_time) > deadline - 10:
            break
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s
    
    return sorted(best_set)
