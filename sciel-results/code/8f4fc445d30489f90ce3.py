"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time as _time

_DEADLINE = _time.time() + 290  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _residue_classes(points, p):
    classes = {}
    for v in points:
        classes.setdefault(v % p, []).append(v)
    return classes


def _check_admissible(points, primes_le_k):
    """Return True if for every prime p <= len(points), not all p residue classes are used."""
    k = len(points)
    for p in primes_le_k:
        if p > k:
            break
        # count distinct residues
        seen = set()
        for v in points:
            seen.add(v % p)
            if len(seen) == p:
                return False
    return True


def construct():
    # Start from 1 and the primes above sqrt(N); repair any prime whose residue classes are all covered
    # by deleting the least-populated class, until the tuple is admissible.
    points = [1] + [p for p in _primes(N) if p * p > N]
    changed = True
    while changed:
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break

    # Local search: try to add back excluded elements one at a time if admissibility is preserved.
    # The greedy repair removed elements; we can try re-inserting them.
    # First, build the set of excluded elements (those in [0, N] not in points).
    point_set = set(points)
    # We only consider a reasonable candidate pool: elements that were removed or nearby.
    # For efficiency, consider all integers in [0, N] not in point_set, but limit work.
    candidates = [v for v in range(N + 1) if v not in point_set]

    # Precompute primes up to N (max possible k)
    max_k = N + 1
    all_primes = _primes(max_k)

    improved = True
    while improved and _time.time() < _DEADLINE:
        improved = False
        k = len(points)
        # For each candidate, check if adding it keeps admissibility.
        # A candidate v is addable iff for every prime p <= k+1, v does not complete all p residue classes.
        # i.e., for every p <= k+1, there exists some residue class mod p not covered by points ∪ {v}.
        # Equivalently: for every p <= k+1, either (a) p is not fully covered by points (so adding v is safe),
        # or (b) v's residue mod p is in a class already present (so it doesn't add a new class).
        # The only dangerous case: p is fully covered by points AND v's residue is the missing one — but if
        # points already cover all p classes, that can't happen. So danger is: p is NOT fully covered by points,
        # and v's residue is the unique missing class.
        # Actually: points may or may not cover all p classes. If points already cover all p classes, that's
        # already a violation — but we maintain admissibility, so points never cover all p classes for p <= k.
        # After adding v, k becomes k+1. We need: for all p <= k+1, points∪{v} does NOT cover all p classes.
        # For p <= k: points already don't cover all p classes. Adding v could complete them.
        #   v completes p iff v % p is the missing residue class for p.
        # For p = k+1 (if prime): points may or may not cover all (k+1) classes. Adding v could complete.

        # Efficient check: precompute for each prime p <= k+1 which residues are missing in points.
        # Then v is safe iff for every prime p <= k+1, v % p is NOT the missing residue (or p has multiple missing).
        # If p has 0 missing (fully covered), that's already invalid — shouldn't happen.
        # If p has 1 missing residue r, v is unsafe iff v % p == r.
        # If p has >= 2 missing residues, v is always safe for that p.

        # Build missing-residue info for primes <= k+1
        primes_to_check = [p for p in all_primes if p <= k + 1]
        unsafe_for = {}  # p -> set of residues that would complete p (i.e., the missing ones)
        for p in primes_to_check:
            classes = _residue_classes(points, p)
            all_residues = set(range(p))
            missing = all_residues - set(classes.keys())
            if len(missing) == 1:
                r = next(iter(missing))
                unsafe_for[p] = r

        # Now try candidates in order; add the first one that's safe.
        # Sort candidates by some heuristic: prefer smaller values? Or just iterate.
        # For efficiency, iterate and add as many as possible in this pass.
        added_any = False
        for v in candidates:
            if v in point_set:
                continue
            # Check safety
            safe = True
            for p, r in unsafe_for.items():
                if v % p == r:
                    safe = False
                    break
            if safe:
                points.append(v)
                point_set.add(v)
                candidates.remove(v)
                added_any = True
                improved = True
                # Recompute unsafe_for after adding v (k increased by 1)
                k = len(points)
                primes_to_check = [p for p in all_primes if p <= k + 1]
                unsafe_for = {}
                for p in primes_to_check:
                    classes = _residue_classes(points, p)
                    all_residues = set(range(p))
                    missing = all_residues - set(classes.keys())
                    if len(missing) == 1:
                        r = next(iter(missing))
                        unsafe_for[p] = r
            if _time.time() >= _DEADLINE:
                break

    return points
# EVOLVE-BLOCK-END
