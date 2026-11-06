import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    N = 192
    MAX_SIZE = 48
    K = 3  # number of other summands

    # Precompute all triples of distinct indices (i, j, k) from 0..N-1
    # We need to check: for selected set S, no element a in S divides sum of 3 distinct others in S.
    # Equivalently: for each a in S, and for each triple (b,c,d) of distinct elements in S\{a},
    #   (b+c+d) % a != 0

    # Strategy: Greedy with conflict graph + local search
    # 
    # Key insight: The condition is "for all a in S, for all distinct b,c,d in S\{a}, (b+c+d) % a != 0"
    # 
    # We can model conflicts: element a conflicts with a triple (b,c,d) if (b+c+d) % a == 0.
    # But this is a higher-order conflict, not pairwise.
    #
    # Alternative approach: Build set greedily, checking compatibility efficiently.
    #
    # For a candidate x to be added to set S:
    #   1. For x: no 3 distinct elements in S sum to 0 mod x
    #   2. For each a in S: no 3 distinct elements in (S\{a}) ∪ {x} sum to 0 mod a
    #      Since S was valid, we only need to check triples involving x.
    #      Triples involving x: {x, b, c} where b,c are distinct elements in S\{a}
    #      So: (x + b + c) % a != 0 for all distinct b,c in S\{a}

    def can_add(S, x):
        """Check if x can be added to set S."""
        S_list = list(S)
        n = len(S_list)
        
        # Condition 1: for x, no 3 distinct in S sum to 0 mod x
        if n >= 3:
            if x == 1:
                return False
            # Check all triples in S
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if (S_list[i] + S_list[j] + S_list[k]) % x == 0:
                            return False
        
        # Condition 2: for each a in S, check triples involving x
        for a in S_list:
            if a == 1:
                # Any sum is divisible by 1, so we need at least 3 elements in S\{a} ∪ {x}
                # S\{a} has n-1 elements, plus x makes n elements. Need 3 distinct.
                if n >= 3:  # n-1 + 1 = n >= 3 means n >= 3
                    return False
                continue
            # Check pairs (b, c) in S\{a} such that (x + b + c) % a == 0
            for i in range(n):
                if S_list[i] == a:
                    continue
                for j in range(i + 1, n):
                    if S_list[j] == a:
                        continue
                    if (x + S_list[i] + S_list[j]) % a == 0:
                        return False
        
        return True

    def is_valid(S):
        """Full validation of set S."""
        S_list = list(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            if a == 1:
                if n >= 4:  # need 3 others
                    return False
                continue
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for b, c, d in itertools.combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def greedy(order):
        """Greedy build using given order of candidates."""
        S = set()
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S):
        """Local search: try to add elements, then try swaps."""
        S = set(S)
        unselected = set(range(1, N + 1)) - S
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            for x in list(unselected):
                if len(S) >= MAX_SIZE:
                    break
                if can_add(S, x):
                    S.add(x)
                    unselected.discard(x)
                    improved = True
            
            # Try swaps: remove one, try to add one or more
            if len(S) < MAX_SIZE:
                for a in list(S):
                    if time.time() > deadline:
                        break
                    S.remove(a)
                    unselected.add(a)
                    added = []
                    for x in list(unselected):
                        if len(S) >= MAX_SIZE:
                            break
                        if can_add(S, x):
                            S.add(x)
                            unselected.discard(x)
                            added.append(x)
                    if len(added) >= 2:
                        improved = True
                    elif len(added) == 0:
                        S.add(a)
                        unselected.discard(a)
                    else:
                        # Swapped a for added[0], net zero. Check if we can add more.
                        can_add_more = False
                        for x in list(unselected):
                            if len(S) >= MAX_SIZE:
                                break
                            if can_add(S, x):
                                can_add_more = True
                                break
                        if not can_add_more:
                            S.discard(added[0])
                            unselected.add(added[0])
                            S.add(a)
                            unselected.discard(a)
        
        return S

    # Multiple strategies to find good initial sets
    random.seed(42)
    best = set()

    # Strategy 1: Descending order (larger numbers first, less likely to divide sums)
    S = greedy(range(N, 0, -1))
    if len(S) > len(best):
        best = S

    if time.time() > deadline:
        return sorted(best)

    # Strategy 2: Ascending order
    S = greedy(range(1, N + 1))
    if len(S) > len(best):
        best = S

    if time.time() > deadline:
        return sorted(best)

    # Strategy 3: Prioritize numbers that are unlikely to divide sums
    # Numbers near N/2 or larger are less likely to divide small sums
    # Actually, large numbers are less likely to divide sums of 3 smaller numbers
    # Let's try a weighted random order

    # Strategy 4: Multiple random restarts
    for _ in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy(order)
        if len(S) > len(best):
            best = S

    if time.time() > deadline:
        return sorted(best)

    # Local search on best
    S = local_search(best)
    if len(S) > len(best):
        best = S

    if time.time() > deadline:
        return sorted(best)

    # More random restarts with local search
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy(order)
        S = local_search(S)
        if len(S) > len(best):
            best = S

    if time.time() > deadline:
        return sorted(best)

    # Try to improve with more aggressive local search on best
    if len(best) < MAX_SIZE:
        S = local_search(best)
        if len(S) > len(best):
            best = S

    # Final validation
    if not is_valid(best):
        # Fallback to a known valid small set
        best = set(range(100, 148))  # 48 numbers from 100 to 147
        if not is_valid(best):
            # Even simpler fallback
            best = set()
            for x in range(N, 0, -1):
                if len(best) >= MAX_SIZE:
                    break
                if can_add(best, x):
                    best.add(x)

    return sorted(best)
