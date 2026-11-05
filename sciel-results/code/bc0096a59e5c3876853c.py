import time
import math
import random

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # Pre-filter: remove x where 2*x is a perfect square
    filtered = [x for x in nums if not _is_square(2 * x)]

    # Build conflict graph: edge (a, b) if a <= b and a+b is a perfect square
    # We need to find a maximum independent set (no conflicts)
    # Actually, we need a set where NO pair (a, b) with a<=b has a+b being a perfect square.
    # This includes a=b (diagonal), so we already filtered those.
    # For a < b: conflict if a+b is a perfect square.

    # Modular partitioning approach:
    # Numbers mod 4:
    # If a ≡ r1 (mod 4) and b ≡ r2 (mod 4), then a+b ≡ r1+r2 (mod 4)
    # Perfect squares mod 4 are 0 or 1.
    # So conflicts can only occur when r1+r2 ≡ 0 or 1 (mod 4).
    # 
    # Pairs of residues that CAN conflict:
    # 0+0=0, 0+1=1, 1+0=1, 1+3=0, 3+1=0, 3+3=2(no), 2+2=0, 2+? 
    # Wait, let me redo:
    # Squares mod 4: 0, 1
    # a+b mod 4 must be 0 or 1 for conflict.
    # (0,0)->0 YES, (0,1)->1 YES, (0,3)->3 NO, (0,2)->2 NO
    # (1,1)->2 NO, (1,2)->3 NO, (1,3)->0 YES
    # (2,2)->0 YES, (2,3)->1 YES
    # (3,3)->2 NO
    #
    # So conflict-free residue pairs (where a+b mod 4 is 2 or 3):
    # (1,1), (1,2), (0,3), (3,3), (2,3) wait no...
    #
    # Actually I want NO conflicts, so I want a+b mod 4 NOT in {0,1}.
    # So a+b mod 4 must be 2 or 3.
    #
    # Residue classes that are internally safe (a+b for a,b in same class has mod 4 not in {0,1}):
    # Class 1: 1+1=2 mod 4. SAFE internally.
    # Class 3: 3+3=6=2 mod 4. SAFE internally.
    # Class 0: 0+0=0 mod 4. NOT safe.
    # Class 2: 2+2=4=0 mod 4. NOT safe.
    #
    # Cross-class safe pairs (a+b mod 4 in {2,3}):
    # (0,2): 0+2=2 SAFE
    # (1,3): 1+3=4=0 NOT safe
    # (0,3): 0+3=3 SAFE
    # (1,2): 1+2=3 SAFE
    # (2,3): 2+3=5=1 NOT safe
    # (0,1): 0+1=1 NOT safe
    #
    # So safe combinations:
    # - All of class 1 + all of class 2 (cross safe, both internally safe? Class 2 is NOT internally safe)
    # Hmm, class 2 is not internally safe. So I can't take all of class 2.
    #
    # Let me think about this differently. Take large conflict-free subsets from each class.
    #
    # Strategy: 
    # 1. Take all numbers in residue classes 1 and 3 (mod 4). These are internally safe.
    #    Cross-check: 1+3 = 0 mod 4 which IS a conflict. So I can't take ALL of both.
    #
    # Better approach: Use the modular structure to get a good initial solution, then do local search.
    #
    # Let's just do a greedy + local search approach with the modular insight for initialization.

    # Precompute conflicts
    # conflict[a][b] = True if a+b is a perfect square
    conflict = {}
    squares = set()
    max_sum = 2 * N
    for s in range(4, max_sum + 1):
        r = math.isqrt(s)
        if r * r == s:
            squares.add(s)

    # Build adjacency for greedy
    adj = {x: set() for x in filtered}
    for i, a in enumerate(filtered):
        for b in filtered[i+1:]:
            if (a + b) in squares:
                adj[a].add(b)
                adj[b].add(a)

    # Greedy: min-degree ordering
    def greedy_construct(candidates, adj_map):
        if not candidates:
            return []
        # Sort by degree (number of conflicts in remaining set)
        remaining = set(candidates)
        result = []
        while remaining:
            # Find vertex with minimum degree in remaining
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(adj_map[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
                    if deg == 0:
                        break
            result.append(chosen)
            remaining.discard(chosen)
        return result

    # Initial solution using modular hint: start with residues 1 and 3 mod 4
    # But check cross-conflicts between class 1 and class 3
    class1 = [x for x in filtered if x % 4 == 1]
    class3 = [x for x in filtered if x % 4 == 3]
    class0 = [x for x in filtered if x % 4 == 0]
    class2 = [x for x in filtered if x % 4 == 2]

    # Try different initial seeds
    best_set = set()
    best_size = 0

    seeds = [
        class1 + class3,
        class1 + class0,
        class1 + class2,
        class3 + class0,
        class3 + class2,
        class0 + class2,
        class1 + class2 + [x for x in class0 if x not in (set(class1)|set(class2))],
        filtered,
    ]

    for seed in seeds:
        if not seed:
            continue
        sol = greedy_construct(seed, adj)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = set(sol)

    # Local search: try to improve
    # 1. Try adding elements
    # 2. Try swapping: remove one, add two
    improved = True
    while improved and (time.time() < deadline - 5.0):
        improved = False
        # Try adding
        for x in filtered:
            if x in best_set:
                continue
            # Check if x conflicts with any in best_set
            conflicts = adj[x] & best_set
            if not conflicts:
                best_set.add(x)
                best_size += 1
                improved = True
        # Try swaps: remove one element, try to add two
        for x in list(best_set):
            if (time.time() > deadline - 5.0):
                break
            best_set.discard(x)
            original_size = len(best_set)
            # Try to add elements that were conflicting with x
            added = []
            for y in filtered:
                if y in best_set or y == x:
                    continue
                if adj[y] & best_set:
                    continue
                best_set.add(y)
                added.append(y)
            if len(added) >= 2:
                improved = True
            else:
                for y in added:
                    best_set.discard(y)
                best_set.add(x)

    # More aggressive local search: 2-opt style
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            # Try removing one and adding multiple
            for x in list(best_set):
                if time.time() > deadline - 5.0:
                    break
                best_set.discard(x)
                cur_size = len(best_set)
                added = []
                for y in filtered:
                    if y in best_set or y == x:
                        continue
                    if adj[y] & best_set:
                        continue
                    best_set.add(y)
                    added.append(y)
                if len(added) > 1:
                    improved = True
                else:
                    for y in added:
                        best_set.discard(y)
                    best_set.add(x)
            if not improved:
                break

    # Final polish: try random restarts if time permits
    if time.time() < deadline - 15.0:
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Random subset initialization
            shuffled = filtered[:]
            random.shuffle(shuffled)
            # Take a random large subset and greedily construct
            subset_size = len(shuffled) // 2 + random.randint(0, len(shuffled) // 4)
            seed = shuffled[:subset_size]
            sol = greedy_construct(seed, adj)
            # Local improvement
            cur_set = set(sol)
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                for x in filtered:
                    if x in cur_set:
                        continue
                    if not (adj[x] & cur_set):
                        cur_set.add(x)
                        changed = True
            if len(cur_set) > best_size:
                best_size = len(cur_set)
                best_set = cur_set

    result = sorted(best_set)
    if not result:
        return [1]
    return result

