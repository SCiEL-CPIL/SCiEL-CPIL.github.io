"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    points = list(range(N + 1))
    removed = set()
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        k = len(points)
        # Find the smallest prime p <= k where all p residue classes are present
        for p in primes:
            if p > k:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = [v]
                else:
                    classes[r].append(v)
            if len(classes) == p:
                # Choose the residue class to remove using a better heuristic:
                # minimize the maximum class size after removal
                best_class = None
                best_max_size = float('inf')
                for r, elems in classes.items():
                    # Simulate removing this class
                    max_size = 0
                    for r2, elems2 in classes.items():
                        if r2 != r:
                            if len(elems2) > max_size:
                                max_size = len(elems2)
                    # After removal, the max class size among remaining is max_size
                    # We want to minimize this
                    if max_size < best_max_size:
                        best_max_size = max_size
                        best_class = r
                    elif max_size == best_max_size:
                        # Tie-break: remove the smallest class
                        if len(elems) < len(classes[best_class]):
                            best_class = r
                remove_set = set(classes[best_class])
                removed.update(remove_set)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    
    # Local improvement: try to add back removed elements
    if time.time() < deadline - 10:
        for v in sorted(removed):
            if time.time() > deadline:
                break
            # Check if adding v back keeps the tuple admissible
            candidate = set(points) | {v}
            k = len(candidate)
            admissible = True
            for p in primes:
                if p > k:
                    break
                classes = set()
                for v2 in candidate:
                    classes.add(v2 % p)
                    if len(classes) == p:
                        break
                if len(classes) == p:
                    admissible = False
                    break
            if admissible:
                points.append(v)
                removed.discard(v)
    
    return points
# EVOLVE-BLOCK-END
