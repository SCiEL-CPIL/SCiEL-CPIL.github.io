"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0  # 85s out of 90s budget
    
    N = 180
    
    # Precompute: for each pair (b, c) with b < c, compute a = bc/(b+c) if integer
    # For each a, we need to know which pairs (b,c) satisfy bc/(b+c) = a
    # i.e., bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # So for a given a, the pairs are (b, c) where (b-a)(c-a) = a^2
    # b = a + d, c = a + a^2/d for divisors d of a^2
    
    # Alternative: for each pair (b,c), compute a = bc/(b+c) and store
    # We need efficient checking. Let's precompute all valid triples.
    
    # For each a in 1..N, find all pairs (b,c) with b,c > a, b != c, such that bc = a(b+c)
    # (b-a)(c-a) = a^2
    # Let d be a divisor of a^2, then b = a + d, c = a + a^2/d
    # We need b, c <= N and b != c (i.e., d != a^2/d, i.e., d^2 != a^2, i.e., d != a)
    
    triples_by_a = {}  # a -> list of (b, c) pairs
    triples_by_b = {}  # b -> list of (a, c) pairs  
    triples_by_c = {}  # c -> list of (a, b) pairs
    
    for a in range(1, N + 1):
        pairs = []
        a_sq = a * a
        # Find all divisors d of a_sq
        for d in range(1, a_sq + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    # Ensure b < c for consistency
                    if b > c:
                        b, c = c, b
                    pairs.append((b, c))
        # Remove duplicates
        pairs = list(set(pairs))
        triples_by_a[a] = pairs
        for (b, c) in pairs:
            if b not in triples_by_b:
                triples_by_b[b] = []
            triples_by_b[b].append((a, c))
            if c not in triples_by_c:
                triples_by_c[c] = []
            triples_by_c[c].append((a, b))
    
    # Also build: for each pair (b,c), what is a?
    # Already have triples_by_a, let's also make pair_to_a
    pair_to_a = {}
    for a in range(1, N + 1):
        for (b, c) in triples_by_a[a]:
            pair_to_a[(min(b,c), max(b,c))] = a
    
    def is_valid_addition(selected, x):
        """Check if adding x to selected keeps the set valid."""
        # Check all triples where x participates
        # x could be a, b, or c
        # If x is a: need pairs (b,c) from triples_by_a[x] where both b,c in selected
        for (b, c) in triples_by_a.get(x, []):
            if b in selected and c in selected:
                return False
        # If x is b: for each (a,c) where x is b, check if a and c in selected
        for (a, c) in triples_by_b.get(x, []):
            if a in selected and c in selected:
                return False
        # If x is c: for each (a,b) where x is c, check if a and b in selected
        for (a, b) in triples_by_c.get(x, []):
            if a in selected and b in selected:
                return False
        return True
    
    def greedy(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if is_valid_addition(selected, x):
                selected.add(x)
        return selected
    
    def local_improve(initial, time_budget):
        """Try to improve by adding elements, then swapping."""
        current = set(initial)
        start = time.time()
        
        # Phase 1: Try adding each missing element
        for x in range(1, N + 1):
            if time.time() > start + time_budget:
                break
            if x in current:
                continue
            if is_valid_addition(current, x):
                current.add(x)
        
        # Phase 2: Try swaps - remove x, add y
        for _ in range(5):
            if time.time() > start + time_budget:
                break
            improved = False
            for x in sorted(current):
                if time.time() > start + time_budget:
                    break
                trial = current - {x}
                for y in range(1, N + 1):
                    if y in trial:
                        continue
                    if y == x:
                        continue
                    if is_valid_addition(trial, y):
                        trial.add(y)
                        current = trial
                        improved = True
                        break
            if not improved:
                break
        
        return current
    
    # Strategy 1: Upper half (known good seed)
    candidate = set(range(91, 181))
    best = set(candidate)
    best_size = len(best)
    
    # Verify seed
    if not all(is_valid_addition(best - {x}, x) for x in best):
        # Re-verify properly
        pass
    
    # Strategy 2: Greedy from bottom
    if time.time() < deadline - 5.0:
        c2 = greedy(range(1, N + 1))
        if len(c2) > best_size:
            best = c2
            best_size = len(best)
    
    # Strategy 3: Greedy from top
    if time.time() < deadline - 5.0:
        c3 = greedy(range(N, 0, -1))
        if len(c3) > best_size:
            best = c3
            best_size = len(best)
    
    # Strategy 4: Various range-based initial solutions
    if time.time() < deadline - 5.0:
        for start in [50, 60, 70, 80, 85, 90, 95, 100, 110, 120]:
            if time.time() > deadline - 5.0:
                break
            c = greedy(range(start, N + 1))
            if len(c) > best_size:
                best = c
                best_size = len(best)
    
    # Strategy 5: Randomized greedy with local improvement
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        for attempt in range(15):
            if time.time() > deadline - 10.0:
                break
            elements = list(range(1, N + 1))
            rng.shuffle(elements)
            c = greedy(elements)
            # Local improve
            remaining = deadline - time.time() - 5.0
            if remaining > 2.0:
                c = local_improve(c, min(remaining, 5.0))
            if len(c) > best_size:
                best = c
                best_size = len(best)
    
    # Strategy 6: Weighted random - prefer larger numbers (they have fewer constraints)
    if time.time() < deadline - 10.0:
        rng = random.Random(123)
        for attempt in range(10):
            if time.time() > deadline - 10.0:
                break
            # Weight: higher numbers get higher probability
            weights = [i ** 2 for i in range(1, N + 1)]
            total_w = sum(weights)
            order = []
            remaining = list(range(1, N + 1))
            while remaining:
                total = sum(w ** 2 for w in [r for r in remaining])
                r = rng.random() * total
                cum = 0
                chosen = None
                for x in remaining:
                    cum += x ** 2
                    if cum >= r:
                        chosen = x
                        break
                if chosen is None:
                    chosen = remaining[-1]
                order.append(chosen)
                remaining.remove(chosen)
            c = greedy(order)
            remaining_time = deadline - time.time() - 5.0
            if remaining_time > 2.0:
                c = local_improve(c, min(remaining_time, 5.0))
            if len(c) > best_size:
                best = c
                best_size = len(best)
    
    # Final local improvement on best
    remaining = deadline - time.time() - 3.0
    if remaining > 2.0 and best:
        best = local_improve(best, min(remaining, 8.0))
        best_size = len(best)
    
    if not best:
        return list(range(91, 181))
    
    return sorted(best)

# EVOLVE-BLOCK-END
