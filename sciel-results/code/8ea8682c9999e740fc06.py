import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    N = 128

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # For each element v, precompute set of pairs (a,b) with a<b such that (a,b,v) is bad
    # We'll store as: conflict_pairs[v] = list of (a, b) where a < b and (a,b,v) is a bad triple
    conflict_pairs = [[] for _ in range(N + 1)]
    # Also: for each pair (a,b), the set of v that conflict
    pair_forbidden = {}  # (a,b) -> set of v
    for (a, b, c) in bad_triples:
        # v=a conflicts with pair (b,c)
        conflict_pairs[a].append((b, c))
        pair_forbidden.setdefault((b, c), set()).add(a)
        # v=b conflicts with pair (a,c)
        conflict_pairs[b].append((a, c))
        pair_forbidden.setdefault((a, c), set()).add(b)
        # v=c conflicts with pair (a,b)
        conflict_pairs[c].append((a, b))
        pair_forbidden.setdefault((a, b), set()).add(c)

    # Convert to sets for fast lookup
    conflict_pairs_set = [set(p) for p in conflict_pairs]

    if time.time() > deadline:
        return [1]

    def greedy_from_set(initial_set, order=None):
        """
        Greedy add elements from 'order' (or 1..N) to 'initial_set'.
        Returns the final list.
        """
        if order is None:
            order = list(range(1, N + 1))
        
        selected_list = list(initial_set)
        selected_set = set(selected_list)
        pair_block_count = [0] * (N + 1)
        
        # Rebuild pair_block_count for initial_set
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                key = (a, b) if a < b else (b, a)
                if key in pair_forbidden:
                    for w in pair_forbidden[key]:
                        pair_block_count[w] += 1

        for x in order:
            if x in selected_set:
                continue
            if pair_block_count[x] == 0:
                # Add x
                selected_list.append(x)
                selected_set.add(x)
                for y in selected_list[:-1]:
                    key = (x, y) if x < y else (y, x)
                    if key in pair_forbidden:
                        for w in pair_forbidden[key]:
                            pair_block_count[w] += 1
        return selected_list

    # Multiple greedy strategies
    candidates = []

    # Strategy 1: Natural order
    candidates.append(greedy_from_set([]))

    # Strategy 2: Reverse order
    candidates.append(greedy_from_set([], list(range(N, 0, -1))))

    # Strategy 3: Sorted by conflict count (ascending) - fewer conflicts first
    conflict_count = [0] * (N + 1)
    for v in range(1, N + 1):
        conflict_count[v] = len(conflict_pairs_set[v])
    order3 = sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x))
    candidates.append(greedy_from_set([], order3))

    # Strategy 4: Sorted by conflict count (descending)
    order4 = sorted(range(1, N + 1), key=lambda x: (-conflict_count[x], x))
    candidates.append(greedy_from_set([], order4))

    # Strategy 5: Primes and 1
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    primes = [1] + [p for p in range(2, N + 1) if is_prime(p)]
    candidates.append(greedy_from_set(primes)) # Start with primes, add others

    # Strategy 6: Powers of 2
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128]
    candidates.append(greedy_from_set(powers2))

    # Strategy 7: Odd numbers
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    candidates.append(greedy_from_set(odds))

    # Strategy 8: Numbers with few prime factors
    def total_omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    order8 = sorted(range(1, N + 1), key=lambda x: (total_omega(x), x))
    candidates.append(greedy_from_set([], order8))

    # Strategy 9: Sorted by distinct prime factors
    def distinct_omega(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    order9 = sorted(range(1, N + 1), key=lambda x: (distinct_omega(x), x))
    candidates.append(greedy_from_set([], order9))

    # Strategy 10: Even numbers
    evens = [x for x in range(2, N + 1, 2)]
    candidates.append(greedy_from_set(evens))

    # Random restarts for initial greedy
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidates.append(greedy_from_set([], order))

    # Pick best candidate
    best = max(candidates, key=len)
    best_size = len(best)
    
    # Local Search: Randomized Kick
    # We take the current best set, remove k random elements, and re-greedy.
    # This helps escape local optima.
    
    kick_iterations = 0
    while time.time() < deadline - 5:
        kick_iterations += 1
        # Decide kick size: start small, maybe increase?
        # For N=128, kicking 2-4 elements is reasonable.
        k = random.randint(2, 4)
        
        current_set = list(best)
        if len(current_set) < k:
            break
            
        # Remove k random elements
        to_remove = random.sample(current_set, k)
        remove_set = set(to_remove)
        initial_for_kick = [x for x in current_set if x not in remove_set]
        
        # Re-greedy from this initial set
        # Use a random order to add back elements? Or standard order?
        # Standard order is deterministic and fast. Random order might help.
        # Let's use a shuffled order for the "add" phase to diversify.
        add_order = list(range(1, N + 1))
        random.shuffle(add_order)
        
        new_set = greedy_from_set(initial_for_kick, add_order)
        
        if len(new_set) > best_size:
            best = new_set
            best_size = len(best)
            # Reset kick counter or reduce kick size?
            # Just continue.
        else:
            # Optional: Accept with probability to allow wandering?
            # For now, greedy hill climbing (only accept improvements) is safer.
            # If we want SA, we'd accept worse solutions with prob exp(-delta/T).
            # But maintaining validity is key.
            pass

    # Final validation
    def is_valid(sel):
        s = set(sel)
        if len(s) != len(sel) or len(s) == 0:
            return False
        for a, b, c in combinations(sel, 3):
            if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                return False
        return True

    if not is_valid(best):
        # Fallback: rebuild from scratch with validation
        return [1]

    if not best:
        return [1]

    return best
