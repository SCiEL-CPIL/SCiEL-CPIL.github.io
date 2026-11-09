"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    start_time = time.time()
    
    # Precompute gcd table for all pairs in [1, 128]
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(b,c) == gcd(a,c)
    bad_triples = set()
    for a in range(1, 129):
        for b in range(a+1, 129):
            for c in range(b+1, 129):
                g1 = gcd_table[a][b]
                g2 = gcd_table[b][c]
                g3 = gcd_table[a][c]
                if g1 == g2 == g3:
                    bad_triples.add((a, b, c))
    
    # For each pair, store the set of third elements that form a bad triple
    # conflict[a][b] = set of c such that (a,b,c) is bad
    conflict = {}
    for (a, b, c) in bad_triples:
        if a not in conflict:
            conflict[a] = {}
        if b not in conflict[a]:
            conflict[a][b] = set()
        conflict[a][b].add(c)
        
        if b not in conflict:
            conflict[b] = {}
        if a not in conflict[b]:
            conflict[b][a] = set()
        conflict[b][a].add(c)
        
        if c not in conflict:
            conflict[c] = {}
        if a not in conflict[c]:
            conflict[c][a] = set()
        conflict[c][a].add(b)
        
        if c not in conflict:
            conflict[c] = {}
        if b not in conflict[c]:
            conflict[c][b] = set()
        conflict[c][b].add(a)
    
    def is_valid(selected, x):
        """Check if adding x to selected creates any bad triple."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            ga = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                gb = gcd_table[b][x]
                gab = gcd_table[a][b]
                if ga == gb == gab:
                    return False
        return True
    
    def count_conflicts(selected, x):
        """Count how many bad triples would be formed if x were added."""
        count = 0
        n = len(selected)
        for i in range(n):
            a = selected[i]
            ga = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                gb = gcd_table[b][x]
                gab = gcd_table[a][b]
                if ga == gb == gab:
                    count += 1
        return count
    
    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute prime factor sets
    pf_sets = [set() for _ in range(129)]
    for i in range(1, 129):
        pf_sets[i] = prime_factors(i)
    
    def greedy(order, selected_init=None):
        selected = list(selected_init) if selected_init else []
        selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            if is_valid(selected, x):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_search(selected, max_iterations=1000):
        """Improve the set by trying swaps and additions."""
        best = list(selected)
        best_size = len(best)
        best_set = set(best)
        no_improve_count = 0
        max_no_improve = 20
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try adding a new element
            for x in range(1, 129):
                if x in best_set:
                    continue
                if is_valid(best, x):
                    best.append(x)
                    best_set.add(x)
                    best_size += 1
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for i, a in enumerate(best):
                    # Try removing a and adding some b
                    new_set = [x for x in best if x != a]
                    new_set_set = set(new_set)
                    for b in range(1, 129):
                        if b in new_set_set:
                            continue
                        if is_valid(new_set, b):
                            new_set.append(b)
                            if len(new_set) > best_size:
                                best = new_set
                                best_size = len(best)
                                best_set = set(best)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0
        
        return best
    
    nums = list(range(1, 129))
    best = []
    
    # Strategy 1: Greedy with various orderings
    orderings = []
    
    # Ordering by number of distinct prime factors (fewer first = safer)
    orderings.append(sorted(nums, key=lambda n: (len(pf_sets[n]), n)))
    
    # Ordering by largest prime factor (larger first = more unique)
    orderings.append(sorted(nums, key=lambda n: (-max(pf_sets[n], default=1), n)))
    
    # Ordering by value
    orderings.append(sorted(nums))
    orderings.append(sorted(nums, reverse=True))
    
    # Ordering: primes first, then composites with few factors
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    composites_sorted = sorted(composites, key=lambda n: (len(pf_sets[n]), n))
    orderings.append(primes + composites_sorted)
    
    # Ordering: numbers with unique large prime factors first
    orderings.append(sorted(nums, key=lambda n: (-max(pf_sets[n], default=1), len(pf_sets[n]), n)))
    
    # Random orderings with fixed seeds
    for seed in [42, 123, 7, 999, 555]:
        rng = random.Random(seed)
        shuffled = nums[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = greedy(order)
        result = local_search(result, max_iterations=500)
        if len(result) > len(best):
            best = result
            if time.time() > deadline - 10:
                break
    
    # Strategy 2: Start from a good base and do extensive local search
    if time.time() < deadline - 15:
        # Try starting with just primes
        primes_only = [n for n in nums if is_prime(n)]
        result = greedy(primes_only)
        result = local_search(result, max_iterations=2000)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Iterative improvement with multiple restarts
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            rng = random.Random(1000 + restart)
            shuffled = nums[:]
            rng.shuffle(shuffled)
            result = greedy(shuffled)
            result = local_search(result, max_iterations=1000)
            if len(result) > len(best):
                best = result
    
    # Final polish: try to add any remaining elements
    best_set = set(best)
    changed = True
    while changed:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if is_valid(best, x):
                best.append(x)
                best_set.add(x)
                changed = True
    
    # Ensure nonempty
    if not best:
        best = [1, 2]
    
    return best
# EVOLVE-BLOCK-END
