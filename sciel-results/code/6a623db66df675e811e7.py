"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Precompute groups: for each x, groups[x][L] = list of y such that lcm(x,y) == L
    groups = [{} for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            if x == y:
                continue
            L = lcm_table[x][y]
            if L not in groups[x]:
                groups[x][L] = []
            groups[x][L].append(y)
    
    # Precompute witness pairs: witness[x][L] = set of (a,b) such that
    # lcm(x,a)==L, lcm(x,b)==L, and lcm(a,b)==L
    witness = [{} for _ in range(n + 1)]
    for x in range(1, n + 1):
        for L, y_list in groups[x].items():
            if len(y_list) >= 2:
                pairs = set()
                for i in range(len(y_list)):
                    for j in range(i + 1, len(y_list)):
                        a, b = y_list[i], y_list[j]
                        if lcm_table[a][b] == L:
                            pairs.add((min(a, b), max(a, b)))
                if pairs:
                    witness[x][L] = pairs
    
    def is_compatible(x, selected_set):
        """Fast compatibility check using precomputed witness pairs."""
        for L, pairs in witness[x].items():
            for a, b in pairs:
                if a in selected_set and b in selected_set:
                    return False
        return True
    
    # Enumerate all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Build element-to-triples mapping
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    def greedy_hitting_set(random_seed=None):
        """Greedy: repeatedly remove the element in the most uncovered triples."""
        if random_seed is not None:
            random.seed(random_seed)
        uncovered = set(bad_triples)
        removed = set()
        
        while uncovered:
            if time.time() > deadline - 15:
                break
            best_x = None
            best_size = 0
            candidates = []
            for x in elem_triples:
                if x in removed:
                    continue
                size = len(elem_triples[x] & uncovered)
                if size > best_size:
                    best_size = size
                    candidates = [x]
                elif size == best_size and size > 0:
                    candidates.append(x)
            if not candidates or best_size == 0:
                break
            best_x = random.choice(candidates) if random_seed is not None else candidates[0]
            removed.add(best_x)
            uncovered -= elem_triples[best_x]
        
        return set(range(1, n + 1)) - removed
    
    def greedy_solve(order):
        """Greedy: add elements in given order if compatible."""
        selected_set = set()
        for x in order:
            if is_compatible(x, selected_set):
                selected_set.add(x)
        return selected_set
    
    def local_improve(selected_set, do_kicks=False):
        """Try to improve by adding elements, swaps, and optionally kicks."""
        selected_set = set(selected_set)
        improved = True
        while improved:
            if time.time() > deadline - 8:
                break
            improved = False
            # Try adding any missing element
            for x in range(1, n + 1):
                if time.time() > deadline - 8:
                    break
                if x not in selected_set and is_compatible(x, selected_set):
                    selected_set.add(x)
                    improved = True
            # Try 1-for-1 swaps
            if time.time() > deadline - 8:
                break
            for a in list(selected_set):
                if time.time() > deadline - 8:
                    break
                temp_set = selected_set - {a}
                for x in range(1, n + 1):
                    if x not in temp_set and is_compatible(x, temp_set):
                        if len(temp_set) + 1 > len(selected_set):
                            selected_set = temp_set | {x}
                            improved = True
                            break
            # Try 1-for-2 swaps
            if time.time() > deadline - 8:
                break
            for a in list(selected_set):
                if time.time() > deadline - 8:
                    break
                temp_set = selected_set - {a}
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 8:
                        break
                    if x not in temp_set and is_compatible(x, temp_set):
                        temp_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    selected_set = temp_set
                    improved = True
                    break
            # Kicks: 2-swap moves to escape local optima
            if do_kicks and time.time() < deadline - 8:
                kick_found = False
                for a in list(selected_set):
                    if time.time() > deadline - 8:
                        break
                    for b in list(selected_set):
                        if a >= b:
                            continue
                        if time.time() > deadline - 8:
                            break
                        temp_set = selected_set - {a, b}
                        added = 0
                        for x in range(1, n + 1):
                            if time.time() > deadline - 8:
                                break
                            if x not in temp_set and is_compatible(x, temp_set):
                                temp_set.add(x)
                                added += 1
                                if added >= 3:
                                    break
                        if added >= 3:
                            selected_set = temp_set
                            kick_found = True
                            improved = True
                            break
                    if kick_found:
                        break
        return selected_set
    
    def backtracking_polish(selected_set, time_limit=5.0):
        """Backtracking branch-and-bound to try to add more elements."""
        selected_set = set(selected_set)
        candidates = [x for x in range(1, n + 1) if x not in selected_set]
        # Sort candidates by conflict count (fewer conflicts first)
        def conflict_count(x):
            total = 0
            for L, pairs in witness[x].items():
                total += len(pairs)
            return total
        candidates.sort(key=conflict_count)
        
        best_size = len(selected_set)
        best_set = set(selected_set)
        end_time = time.time() + time_limit
        
        def bt(idx, current_set, current_size):
            nonlocal best_size, best_set
            if time.time() > end_time:
                return
            if current_size >= best_size:
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                return
            if idx >= len(candidates):
                return
            # Prune: even if we add all remaining, can we beat best?
            if current_size + (len(candidates) - idx) <= best_size:
                return
            x = candidates[idx]
            # Try adding x
            if is_compatible(x, current_set):
                current_set.add(x)
                bt(idx + 1, current_set, current_size + 1)
                current_set.discard(x)
            # Try skipping x
            bt(idx + 1, current_set, current_size)
        
        bt(0, selected_set, len(selected_set))
        return best_set
    
    # === Phase 1: Multiple greedy hitting set attempts ===
    best_set = set()
    
    # Deterministic hitting set
    result = greedy_hitting_set()
    best_set = result
    
    # Randomized hitting set with different tie-breaking
    for trial in range(15):
        if time.time() > deadline - 30:
            break
        result = greedy_hitting_set(random_seed=trial * 7 + 3)
        if len(result) > len(best_set):
            best_set = result
    
    # === Phase 2: Local improvement on best hitting set result ===
    best_set = local_improve(best_set)
    
    # === Phase 3: Multiple greedy builds with different orderings ===
    def num_divisors(x):
        count = 0
        for d in range(1, int(x ** 0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        temp = x
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    def conflict_count(x):
        total = 0
        for L, pairs in witness[x].items():
            total += len(pairs)
        return total
    
    orders = [
        numbers,
        list(reversed(numbers)),
        sorted(numbers, key=num_divisors),
        sorted(numbers, key=conflict_count),
        sorted(numbers, key=conflict_count, reverse=True),
        sorted(numbers, key=num_distinct_primes),
        sorted(numbers, key=lambda x: -num_distinct_primes(x)),
    ]
    
    for order in orders:
        if time.time() > deadline - 25:
            break
        result = greedy_solve(order)
        result = local_improve(result)
        if len(result) > len(best_set):
            best_set = result
    
    # === Phase 4: Seeding with large range and primes+1 ===
    # Large range seed
    if time.time() < deadline - 25:
        large_seed = set(range(65, 129))
        # Filter to only compatible elements
        filtered = set()
        for x in sorted(large_seed, key=conflict_count):
            if is_compatible(x, filtered):
                filtered.add(x)
        filtered = local_improve(filtered)
        if len(filtered) > len(best_set):
            best_set = filtered
    
    # Primes+1 seed
    if time.time() < deadline - 25:
        def is_prime(x):
            if x < 2:
                return False
            if x < 4:
                return True
            if x % 2 == 0 or x % 3 == 0:
                return False
            d = 5
            while d * d <= x:
                if x % d == 0 or x % (d + 2) == 0:
                    return False
                d += 6
            return True
        
        primes_plus_1 = [p + 1 for p in range(2, 129) if is_prime(p) and p + 1 <= 128]
        seed_set = set(primes_plus_1)
        # Add more elements
        for x in sorted(numbers, key=conflict_count):
            if x not in seed_set and is_compatible(x, seed_set):
                seed_set.add(x)
        seed_set = local_improve(seed_set)
        if len(seed_set) > len(best_set):
            best_set = seed_set
    
    # === Phase 5: Random restarts ===
    if time.time() < deadline - 20:
        random.seed(12345)
        for trial in range(40):
            if time.time() > deadline - 15:
                break
            order = list(numbers)
            random.shuffle(order)
            result = greedy_solve(order)
            result = local_improve(result, do_kicks=(trial % 5 == 0))
            if len(result) > len(best_set):
                best_set = result
    
    # === Phase 6: Kicks on the best set ===
    if time.time() < deadline - 10:
        kicked = local_improve(best_set, do_kicks=True)
        if len(kicked) > len(best_set):
            best_set = kicked
    
    # === Phase 7: Backtracking polish ===
    if time.time() < deadline - 6:
        time_remaining = deadline - 6 - time.time()
        if time_remaining > 1:
            polished = backtracking_polish(best_set, time_limit=min(time_remaining, 5.0))
            if len(polished) > len(best_set):
                best_set = polished
            # Final local improve after polish
            if time.time() < deadline - 3:
                final = local_improve(best_set)
                if len(final) > len(best_set):
                    best_set = final
    
    # Final polish
    if time.time() < deadline - 2:
        best_set = local_improve(best_set)
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
