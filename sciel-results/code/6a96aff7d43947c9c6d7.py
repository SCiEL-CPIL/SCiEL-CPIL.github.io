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
    # Then x is incompatible with S iff there exists (a,b) in witness[x][L] for some L with a,b in S
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
    
    def is_compatible_witness(x, selected_set):
        """Fast compatibility check using precomputed witness pairs."""
        for L, pairs in witness[x].items():
            for a, b in pairs:
                if a in selected_set and b in selected_set:
                    return False
        return True
    
    # === Phase 1: Greedy hitting set (deletion approach) ===
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
    
    def greedy_hitting_set():
        """Greedy: repeatedly remove the element in the most uncovered triples."""
        uncovered = set(bad_triples)
        removed = set()
        
        while uncovered:
            if time.time() > deadline - 10:
                break
            best_x = None
            best_size = 0
            for x in elem_triples:
                if x in removed:
                    continue
                size = len(elem_triples[x] & uncovered)
                if size > best_size:
                    best_size = size
                    best_x = x
            if best_x is None or best_size == 0:
                break
            removed.add(best_x)
            uncovered -= elem_triples[best_x]
        
        return set(range(1, n + 1)) - removed
    
    # Try multiple hitting set orderings (randomized tie-breaking)
    best_set = set()
    
    # Deterministic hitting set
    result = greedy_hitting_set()
    best_set = result
    
    # Randomized hitting set with different tie-breaking
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        random.seed(trial * 7 + 3)
        uncovered = set(bad_triples)
        removed = set()
        
        while uncovered:
            if time.time() > deadline - 20:
                break
            best_x = None
            best_size = 0
            # Collect all candidates with max size, pick randomly
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
            best_x = random.choice(candidates)
            removed.add(best_x)
            uncovered -= elem_triples[best_x]
        
        result = set(range(1, n + 1)) - removed
        if len(result) > len(best_set):
            best_set = result
    
    # === Phase 2: Local improvement (addition + swaps) ===
    def local_improve(selected_set):
        """Try to improve by adding elements or doing swaps."""
        selected_set = set(selected_set)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding any missing element
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if x not in selected_set and is_compatible_witness(x, selected_set):
                    selected_set.add(x)
                    improved = True
            # Try 1-for-2 swaps: remove one element, try to add two
            if time.time() > deadline - 5:
                break
            for a in list(selected_set):
                if time.time() > deadline - 5:
                    break
                temp_set = selected_set - {a}
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if x not in temp_set and is_compatible_witness(x, temp_set):
                        temp_set.add(x)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    selected_set = temp_set
                    improved = True
                    break
        return selected_set
    
    best_set = local_improve(best_set)
    
    # === Phase 3: Additional greedy builds + local improvement ===
    def greedy_solve(order):
        """Greedy: add elements in given order if compatible."""
        selected = []
        selected_set = set()
        for x in order:
            if is_compatible_witness(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected_set
    
    def num_divisors(x):
        count = 0
        for d in range(1, int(x ** 0.5) + 1):
            if x % d == 0:
                count += 1
                if d != x // d:
                    count += 1
        return count
    
    def conflict_count(x):
        """Number of witness pairs involving x."""
        total = 0
        for L, pairs in witness[x].items():
            total += len(pairs)
        return total
    
    # Try various orderings
    orders = [
        numbers,
        list(reversed(numbers)),
        sorted(numbers, key=num_divisors),
        sorted(numbers, key=conflict_count),
        sorted(numbers, key=conflict_count, reverse=True),
    ]
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        result = greedy_solve(order)
        result = local_improve(result)
        if len(result) > len(best_set):
            best_set = result
    
    # Random restarts
    if time.time() < deadline - 15:
        random.seed(12345)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            order = list(numbers)
            random.shuffle(order)
            result = greedy_solve(order)
            result = local_improve(result)
            if len(result) > len(best_set):
                best_set = result
    
    # Final polish
    if time.time() < deadline - 3:
        best_set = local_improve(best_set)
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
