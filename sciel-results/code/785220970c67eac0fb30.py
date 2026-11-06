import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 88.0  # Safety margin for 90s kill limit

    n = 256
    best_A = []
    best_B = []
    best_size = 0

    # Precompute pair-sum conflict: sum_conflict[x][y] = True if there exists a pair (a,b) with a<=b
    # such that a+b == x+y and {a,b} != {x,y} (as unordered pairs)
    # Actually for Sidon: the multiset of pair sums must be unique. So x,y conflict if
    # there's another pair (a,b) (a<=b, {a,b} != {x,y}) with a+b = x+y.
    
    # Precompute: for each sum s, which pairs produce it
    from collections import defaultdict
    pair_by_sum = defaultdict(list)
    for x in range(1, n + 1):
        for y in range(x, n + 1):
            pair_by_sum[x + y].append((x, y))
    
    # sum_conflict[x][y]: True if x+y is produced by another pair
    sum_conflict = [[False] * (n + 1) for _ in range(n + 1)]
    for s, pairs in pair_by_sum.items():
        if len(pairs) > 1:
            for i, (a, b) in enumerate(pairs):
                for j, (c, d) in enumerate(pairs):
                    if i != j:
                        sum_conflict[a][b] = True
                        sum_conflict[b][a] = True
    
    # Precompute: for each difference d, which ordered pairs (x,y) produce x-y = d
    # diff_conflict for disjointness: if x-y = d and a-b = d with (x,y) != (a,b),
    # then x in A and y in A conflicts with a in B and b in B? No...
    # Actually the constraint is: diffs_A and diffs_B must be disjoint.
    # diffs_A = {a - b : a, b in A, a != b} (nonzero differences)
    # diffs_B = {c - d : c, d in B, c != d}
    # So if x-y = d (nonzero), then having {x,y} in A means d in diffs_A,
    # and having {u,v} in B with u-v = d means d in diffs_B, which conflicts.
    
    # For incremental checking, I'll use sets directly but optimize the inner loop.

    def get_sums(S):
        S = list(S)
        return {a + b for i, a in enumerate(S) for b in S[i:]}

    def get_diffs(S):
        S = list(S)
        return {b - a for i, a in enumerate(S) for b in S[i+1:]}

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        Al = list(A)
        Bl = list(B)
        # Check Sidon for A
        sums_A = set()
        for i in range(len(Al)):
            for j in range(i, len(Al)):
                s = Al[i] + Al[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        # Check Sidon for B
        sums_B = set()
        for i in range(len(Bl)):
            for j in range(i, len(Bl)):
                s = Bl[i] + Bl[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        # Check diff disjointness
        diffs_A = set()
        for i in range(len(Al)):
            for j in range(len(Al)):
                if i != j:
                    d = Al[i] - Al[j]
                    if d != 0:
                        diffs_A.add(d)
        diffs_B = set()
        for i in range(len(Bl)):
            for j in range(len(Bl)):
                if i != j:
                    d = Bl[i] - Bl[j]
                    if d != 0:
                        diffs_B.add(d)
        if diffs_A & diffs_B:
            return False
        return True

    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        """Check if x can be added to A."""
        if x in A:
            return False
        # Check Sidon: new sums are {x+s for s in A} and {2x}
        for s in A:
            if (x + s) in sums_A:
                return False
        if (2 * x) in sums_A:
            return False
        # Check diff disjointness: new diffs from x with elements in A
        for s in A:
            d1 = x - s
            d2 = s - x
            if d1 != 0 and d1 in diffs_B:
                return False
            if d2 != 0 and d2 in diffs_B:
                return False
        return True

    def add_to_A(x, A, sums_A, diffs_A):
        """Add x to A and update sums_A, diffs_A."""
        A.add(x)
        sums_A.add(2 * x)
        for s in A - {x}:
            sums_A.add(x + s)
            diffs_A.add(x - s)
            diffs_A.add(s - x)

    def can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
        """Check if x can be added to B."""
        if x in B:
            return False
        for s in B:
            if (x + s) in sums_B:
                return False
        if (2 * x) in sums_B:
            return False
        for s in B:
            d1 = x - s
            d2 = s - x
            if d1 != 0 and d1 in diffs_A:
                return False
            if d2 != 0 and d2 in diffs_A:
                return False
        return True

    def add_to_B(x, B, sums_B, diffs_B):
        """Add x to B and update sums_B, diffs_B."""
        B.add(x)
        sums_B.add(2 * x)
        for s in B - {x}:
            sums_B.add(x + s)
            diffs_B.add(x - s)
            diffs_B.add(s - x)

    def recompute_all(A, B):
        """Recompute sums and diffs from scratch."""
        Al = list(A)
        Bl = list(B)
        sums_A = set()
        for i in range(len(Al)):
            for j in range(i, len(Al)):
                sums_A.add(Al[i] + Al[j])
        diffs_A = set()
        for i in range(len(Al)):
            for j in range(len(Al)):
                if i != j:
                    d = Al[i] - Al[j]
                    if d != 0:
                        diffs_A.add(d)
        sums_B = set()
        for i in range(len(Bl)):
            for j in range(i, len(Bl)):
                sums_B.add(Bl[i] + Bl[j])
        diffs_B = set()
        for i in range(len(Bl)):
            for j in range(len(Bl)):
                if i != j:
                    d = Bl[i] - Bl[j]
                    if d != 0:
                        diffs_B.add(d)
        return sums_A, diffs_A, sums_B, diffs_B

    def fast_greedy(seed_val):
        """Greedy construction with randomization."""
        rng = random.Random(seed_val)
        
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Try both sets, pick the one that works
            added = False
            order = ['A', 'B'] if rng.random() < 0.5 else ['B', 'A']
            
            for target in order:
                if target == 'A':
                    if len(A) >= 64:
                        continue
                    if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                        add_to_A(x, A, sums_A, diffs_A)
                        added = True
                        break
                else:
                    if len(B) >= 64:
                        continue
                    if can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
                        add_to_B(x, B, sums_B, diffs_B)
                        added = True
                        break
            
            if not added:
                # Try the other order
                for target in order[::-1]:
                    if target == 'A':
                        if len(A) < 64 and can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                            add_to_A(x, A, sums_A, diffs_A)
                            break
                    else:
                        if len(B) < 64 and can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
                            add_to_B(x, B, sums_B, diffs_B)
                            break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def greedy_fill(A, B, sums_A, diffs_A, sums_B, diffs_B, rng=None):
        """Try to add more elements greedily - add all possible in each pass."""
        if rng is None:
            rng = random.Random()
        
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if time.time() - start_time > time_limit:
                    return A, B, sums_A, diffs_A, sums_B, diffs_B
                if x in A or x in B:
                    continue
                if len(A) < 64 and can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    add_to_A(x, A, sums_A, diffs_A)
                    improved = True
                    continue
                if len(B) < 64 and can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
                    add_to_B(x, B, sums_B, diffs_B)
                    improved = True
                    continue
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B, max_iters=200):
        """Try moving elements between A and B to enable more additions."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > time_limit:
                break
            
            # Try moving x from A to B
            for x in list(A):
                if x in B:
                    continue
                if len(B) >= 64:
                    continue
                
                A_temp = A - {x}
                sums_A_temp, diffs_A_temp, _, _ = recompute_all(A_temp, B)
                
                if can_add_to_B(x, B, sums_B, diffs_B, diffs_A_temp):
                    # Move x from A to B
                    A = A_temp
                    sums_A = sums_A_temp
                    diffs_A = diffs_A_temp
                    add_to_B(x, B, sums_B, diffs_B)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try moving x from B to A
            for x in list(B):
                if x in A:
                    continue
                if len(A) >= 64:
                    continue
                
                B_temp = B - {x}
                _, _, sums_B_temp, diffs_B_temp = recompute_all(A, B_temp)
                
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B_temp):
                    # Move x from B to A
                    B = B_temp
                    sums_B = sums_B_temp
                    diffs_B = diffs_B_temp
                    add_to_A(x, A, sums_A, diffs_A)
                    improved = True
                    break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try removing one element and re-filling to get a better configuration.
        Use random subset of candidates to stay within time budget."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        current_size = len(A) + len(B)
        best_size_local = current_size
        best_state = (set(A), set(B))
        
        rng_local = random.Random(int(time.time() * 1000) % 100000)
        
        # Try removing a random subset of elements from A
        candidates_A = list(A)
        rng_local.shuffle(candidates_A)
        # Limit to at most 20 candidates to control time
        candidates_A = candidates_A[:20]
        
        for x in candidates_A:
            if time.time() - start_time > time_limit:
                break
            if len(A) <= 1:
                continue
            
            A_temp = A - {x}
            sums_A_temp, diffs_A_temp, sums_B_temp, diffs_B_temp = recompute_all(A_temp, B)
            
            # Greedily fill
            A_new = set(A_temp)
            B_new = set(B)
            sA = set(sums_A_temp)
            dA = set(diffs_A_temp)
            sB = set(sums_B_temp)
            dB = set(diffs_B_temp)
            
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                if len(A_new) < 64 and can_add_to_A(y, A_new, sA, dA, dB):
                    add_to_A(y, A_new, sA, dA)
                    continue
                if len(B_new) < 64 and can_add_to_B(y, B_new, sB, dB, dA):
                    add_to_B(y, B_new, sB, dB)
                    continue
            
            if len(A_new) + len(B_new) > best_size_local:
                best_size_local = len(A_new) + len(B_new)
                best_state = (set(A_new), set(B_new))
        
        # Try removing a random subset of elements from B
        candidates_B = list(B)
        rng_local.shuffle(candidates_B)
        candidates_B = candidates_B[:20]
        
        for x in candidates_B:
            if time.time() - start_time > time_limit:
                break
            if len(B) <= 1:
                continue
            
            B_temp = B - {x}
            sums_A_temp, diffs_A_temp, sums_B_temp, diffs_B_temp = recompute_all(A, B_temp)
            
            A_new = set(A)
            B_new = set(B_temp)
            sA = set(sums_A_temp)
            dA = set(diffs_A_temp)
            sB = set(sums_B_temp)
            dB = set(diffs_B_temp)
            
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                if len(A_new) < 64 and can_add_to_A(y, A_new, sA, dA, dB):
                    add_to_A(y, A_new, sA, dA)
                    continue
                if len(B_new) < 64 and can_add_to_B(y, B_new, sB, dB, dA):
                    add_to_B(y, B_new, sB, dB)
                    continue
            
            if len(A_new) + len(B_new) > best_size_local:
                best_size_local = len(A_new) + len(B_new)
                best_state = (set(A_new), set(B_new))
        
        if best_size_local > current_size:
            A, B = best_state
            sums_A, diffs_A, sums_B, diffs_B = recompute_all(A, B)
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def two_swap_local_search(A, B, sums_A, diffs_A, sums_B, diffs_B, max_iters=100):
        """Try swapping two elements (one from A, one from B) to improve.
        Use random sampling to stay within budget."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        iters = 0
        rng_local = random.Random(int(time.time() * 1000) % 100000 + 7)
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() - start_time > time_limit:
                break
            
            # Sample random pairs to swap
            list_A = [x for x in A if x not in B]
            list_B = [y for y in B if y not in A]
            
            if not list_A or not list_B:
                break
            
            # Try up to 30 random swap pairs per iteration
            tried = set()
            for _ in range(min(30, len(list_A) * len(list_B))):
                if time.time() - start_time > time_limit:
                    break
                x = rng_local.choice(list_A)
                y = rng_local.choice(list_B)
                if (x, y) in tried:
                    continue
                tried.add((x, y))
                
                A_temp = (A - {x}) | {y}
                B_temp = (B - {y}) | {x}
                
                if len(A_temp) > 64 or len(B_temp) > 64:
                    continue
                
                sA, dA, sB, dB = recompute_all(A_temp, B_temp)
                
                # Check if this enables more additions
                A_new = set(A_temp)
                B_new = set(B_temp)
                for z in range(1, n + 1):
                    if z in A_new or z in B_new:
                        continue
                    if len(A_new) < 64 and can_add_to_A(z, A_new, sA, dA, dB):
                        add_to_A(z, A_new, sA, dA)
                    elif len(B_new) < 64 and can_add_to_B(z, B_new, sB, dB, dA):
                        add_to_B(z, B_new, sB, dB)
                
                if len(A_new) + len(B_new) > len(A) + len(B):
                    A, B = A_new, B_new
                    sums_A, diffs_A, sums_B, diffs_B = recompute_all(A, B)
                    improved = True
                    break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    # Main search loop - more restarts with lighter per-restart work
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Greedy construction
        A, B, sums_A, diffs_A, sums_B, diffs_B = fast_greedy(restart * 1000 + 42)
        
        # Greedy fill
        A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_fill(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: swap (lighter)
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B, max_iters=50)
        
        # Greedy fill again
        A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_fill(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: replace (only if time permits)
        if time.time() - start_time < time_limit - 5:
            A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B)
            A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_fill(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = set(A)
                best_B = set(B)
        
        # Also try with different greedy ordering (second variant)
        if time.time() - start_time > time_limit:
            break
        
        A2, B2, sA2, dA2, sB2, dB2 = fast_greedy(restart * 1000 + 999)
        A2, B2, sA2, dA2, sB2, dB2 = greedy_fill(A2, B2, sA2, dA2, sB2, dB2)
        A2, B2, sA2, dA2, sB2, dB2 = local_search_swap(A2, B2, sA2, dA2, sB2, dB2, max_iters=30)
        A2, B2, sA2, dA2, sB2, dB2 = greedy_fill(A2, B2, sA2, dA2, sB2, dB2)
        
        if is_valid_state(A2, B2):
            size2 = len(A2) + len(B2)
            if size2 > best_size:
                best_size = size2
                best_A = set(A2)
                best_B = set(B2)

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]
    
    # Final polish: one more greedy fill on the best solution
    if time.time() - start_time < time_limit - 2:
        sA, dA, sB, dB = recompute_all(best_A, best_B)
        best_A, best_B, sA, dA, sB, dB = greedy_fill(best_A, best_B, sA, dA, sB, dB)
        if is_valid_state(best_A, best_B):
            pass  # already valid, just got bigger
    
    return [list(best_A), list(best_B)]
