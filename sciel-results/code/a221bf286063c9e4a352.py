"""Rewritten program: local-search optimization for bounded-pairwise-LCM subset."""
# EVOLVE-BLOCK-START
import math
import time

_N = 512
_LIMIT = 512

def _lcm(a, b):
    return a // math.gcd(a, b) * b

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s wall-clock

    # Precompute gcd and lcm for all pairs 1..512
    # Use a 2D approach but only for pairs where both <= 512
    # For speed, precompute lcm_ok[i][j] as boolean
    # But 512x512 = 262144 entries, fine
    # Actually store as a set of allowed pairs or a bitset per element
    # Use: ok[i] = set of j such that lcm(i,j) <= 512
    ok = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            l = i // math.gcd(i, j) * j
            if l <= _LIMIT:
                ok[i].add(j)
                ok[j].add(i)

    # Seed: all numbers 1..22 (isqrt(512)=22), plus try to add more
    # Start with a greedy construction: sort by "difficulty" and add if compatible
    # A good heuristic: numbers with many divisors / high prime powers are harder
    # Let's start with the known seed and then improve

    # Seed: 1..22
    current = set(range(1, 23))

    # Verify seed is valid (it should be)
    # Try to add elements greedily
    # Order to try: smaller numbers first (more compatible)
    for candidate in range(23, _N + 1):
        # Check if candidate is compatible with all in current
        if all(candidate in ok[x] for x in current):
            current.add(candidate)

    # Now local search: try to swap one out for multiple in, or add if possible
    # Phase 1: Try adding any remaining element
    for candidate in range(1, _N + 1):
        if candidate not in current:
            if all(candidate in ok[x] for x in current):
                current.add(candidate)

    # Phase 2: Local search - try removing one element and adding two or more
    # Repeat until no improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and see if we can add more
        for remove in list(current):
            if time.time() > deadline:
                break
            trial = current - {remove}
            # Try to add as many as possible from the complement
            added = 0
            for candidate in range(1, _N + 1):
                if candidate in trial:
                    continue
                if all(candidate in ok[x] for x in trial):
                    trial.add(candidate)
                    added += 1
            # Net gain = added - 1 (we removed one)
            if added > 1:  # net gain >= 1
                current = trial
                improved = True
                break  # restart the outer loop since current changed

    # Phase 3: More aggressive local search - try removing up to 2 elements
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing pairs
            elems = list(current)
            for i in range(len(elems)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(elems)):
                    if time.time() > deadline:
                        break
                    trial = current - {elems[i], elems[j]}
                    added = 0
                    for candidate in range(1, _N + 1):
                        if candidate in trial:
                            continue
                        if all(candidate in ok[x] for x in trial):
                            trial.add(candidate)
                            added += 1
                    if added > 2:  # net gain >= 1
                        current = trial
                        improved = True
                        break
                if improved:
                    break

    # Final verification
    result = sorted(current)
    # Validate
    for idx, a in enumerate(result):
        for b in result[idx + 1:]:
            if _lcm(a, b) > _LIMIT:
                # Fallback to safe seed
                return list(range(1, 23))
    
    if not result:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
