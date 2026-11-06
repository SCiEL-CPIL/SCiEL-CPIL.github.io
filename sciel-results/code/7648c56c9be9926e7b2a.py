"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes.
    For each prime p in primes_to_check, some residue class mod p must be missed."""
    point_set = set(points)
    for p in primes_to_check:
        seen = set()
        for v in points:
            seen.add(v % p)
        if len(seen) == p:
            return False
    return True


def _greedy_construct(deadline):
    """Greedy construction: start with all integers in [0, N], remove points to make admissible."""
    points = list(range(N + 1))
    # We need to check primes up to len(points). Initially len = N+1 = 5001.
    # Primes up to 5001.
    primes = _primes(N + 1)
    
    # We'll process primes in increasing order.
    # For each prime p, if all p residue classes are present, we must remove points.
    # Strategy: remove the least-populated residue class.
    
    # To be efficient, we maintain a set of points.
    point_set = set(points)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(point_set):
            # If p > len(points), it's impossible to cover all p classes, so we're fine.
            # Actually, the condition is "for every prime p <= len(tuple) some residue class mod p is missed".
            # So we only care about p <= len(point_set).
            continue
        
        # Count occurrences of each residue class mod p
        counts = [0] * p
        for v in point_set:
            counts[v % p] += 1
        
        if len([c for c in counts if c > 0]) == p:
            # All classes are present. We need to remove all points in the smallest class.
            min_class = counts.index(min(counts))
            to_remove = [v for v in point_set if v % p == min_class]
            for v in to_remove:
                point_set.remove(v)
    
    return list(point_set)


def _randomized_greedy_with_recovery(seed, deadline):
    """
    Randomized greedy: 
    1. Start with a random subset or all points.
    2. Use a smarter removal strategy or add-back.
    
    The knowledge says: "Randomized greedy with local add-back search allows recovery of points 
    incorrectly removed by deterministic greedy, improving density."
    
    Let's try a different approach:
    Start with a random permutation of [0, N].
    Greedily add points if they don't violate the admissibility condition?
    No, admissibility is a global property.
    
    Alternative:
    1. Generate a candidate set S.
    2. Check if S is admissible.
    3. If not, find a prime p where all classes are covered.
    4. Remove a point from S that is in a "critical" class for p.
    
    Better:
    Start with S = [0, ..., N].
    While S is not admissible:
        Find a prime p <= |S| such that all p classes are covered.
        Choose a random point x in S such that x % p is in a class that, if removed, 
        doesn't immediately break other constraints?
        Actually, just remove a random point from one of the covered classes.
        
    To improve on deterministic greedy, we can use randomization in which class to remove.
    """
    random.seed(seed)
    point_set = set(range(N + 1))
    primes = _primes(N + 1)
    
    # We'll iterate through primes, but we might need to revisit.
    # A simple iterative removal:
    # While not admissible:
    #   Find a bad prime p.
    #   Remove a random point from a random covered class mod p.
    
    # This can be slow if we check admissibility from scratch each time.
    # Let's maintain the state.
    
    # For now, let's implement a simpler randomized removal.
    # We'll keep a list of points and a set for O(1) removal.
    
    # Optimization: Only check primes that are currently "bad".
    # A prime p is bad if all p classes are represented.
    
    # Let's try a local search:
    # 1. Get a greedy solution.
    # 2. Try to add back points that were removed, if possible.
    
    # This function will just do randomized greedy removal.
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(point_set):
            continue
            
        counts = [0] * p
        classes = [[] for _ in range(p)]
        for v in point_set:
            r = v % p
            counts[r] += 1
            classes[r].append(v)
            
        if all(c > 0 for c in counts):
            # Pick a random non-empty class to remove
            # To be smarter, pick the class with the smallest count, but with some randomness
            non_empty = [i for i in range(p) if counts[i] > 0]
            # Sort by count, pick from the bottom few
            non_empty.sort(key=lambda i: counts[i])
            # Pick randomly from the bottom 20% or so
            k = max(1, len(non_empty) // 5)
            chosen_class = random.choice(non_empty[:k])
            to_remove = classes[chosen_class]
            # Remove all points in this class
            for v in to_remove:
                point_set.remove(v)
                
    return list(point_set)


def _local_add_back_search(initial_points, primes, deadline):
    """
    Try to add back points that were removed.
    initial_points is the set from greedy.
    We try to add points from [0, N] \ initial_points.
    """
    point_set = set(initial_points)
    available = set(range(N + 1)) - point_set
    # Shuffle available points
    avail_list = list(available)
    random.shuffle(avail_list)
    
    # We need to check if adding a point keeps the set admissible.
    # Adding a point can only make it *less* admissible (cover more classes).
    # So we check if the new set is admissible.
    
    # To speed up, we can precompute the "bad" primes.
    # A prime p is bad if all classes are covered.
    # Adding a point x covers class x % p.
    # If p was not bad, adding x might make it bad if x % p was the only missing class.
    
    # Let's just try adding points one by one.
    for x in avail_list:
        if time.time() > deadline:
            break
        
        # Check if adding x keeps the set admissible
        # We only need to check primes p <= len(point_set) + 1
        # And specifically, primes where x % p might complete the set of classes.
        
        # For efficiency, let's just do a full check for now, but it might be slow.
        # Let's optimize:
        # Precompute for each prime p, the set of classes currently covered.
        # This is expensive to maintain.
        
        # Alternative: Just check the specific primes that could be affected.
        # A prime p is affected if x % p is a class that was previously missing.
        
        # Let's maintain a dictionary: p -> set of covered classes.
        # But we don't have that here. Let's build it once.
        
        # Actually, let's just try a few points and do a full admissibility check.
        # If it's too slow, we'll optimize.
        
        # Full check is O(|S| * pi(|S|)). |S| ~ 3000, pi(3000) ~ 430.
        # 3000 * 430 = 1.2M operations.
        # We have ~1700 points to try. 1700 * 1.2M = 2B operations. Too slow in Python.
        
        # We need a smarter check.
        # When adding x, only primes p where x % p was the *last* missing class are at risk.
        # So we need to know, for each p, which classes are missing.
        
        # Let's build the missing classes structure once.
        # This is O(|S| * pi(|S|)).
        
        # Let's refactor: build the structure, then try adding.
        
        pass # Placeholder, will implement below


def _smart_local_search(seed, deadline):
    """
    1. Start with a randomized greedy solution.
    2. Build a structure to quickly check admissibility after adding a point.
    3. Try to add back points.
    """
    random.seed(seed)
    
    # Step 1: Randomized greedy
    point_set = set(range(N + 1))
    primes = _primes(N + 1)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(point_set):
            continue
        counts = [0] * p
        classes = [[] for _ in range(p)]
        for v in point_set:
            r = v % p
            counts[r] += 1
            classes[r].append(v)
        if all(c > 0 for c in counts):
            non_empty = [i for i in range(p) if counts[i] > 0]
            non_empty.sort(key=lambda i: counts[i])
            k = max(1, len(non_empty) // 5)
            chosen_class = random.choice(non_empty[:k])
            for v in classes[chosen_class]:
                point_set.remove(v)
    
    # Step 2: Local add-back search
    # Build structure: for each prime p <= N+1, track missing classes.
    # missing[p] = set of classes mod p not covered by point_set.
    # If missing[p] is empty, p is "bad".
    
    # We only care about p <= len(point_set) + 1 (since we might add one point).
    max_p = len(point_set) + 1
    primes_to_check = [p for p in primes if p <= max_p]
    
    missing = {}
    for p in primes_to_check:
        covered = set()
        for v in point_set:
            covered.add(v % p)
        missing[p] = set(range(p)) - covered
    
    # Now try to add points from [0, N] \ point_set
    available = list(set(range(N + 1)) - point_set)
    random.shuffle(available)
    
    for x in available:
        if time.time() > deadline:
            break
        
        # Check if adding x is safe
        safe = True
        for p in primes_to_check:
            # If p is currently bad (missing[p] is empty), adding x will keep it bad.
            # If p is not bad, adding x covers class x % p.
            # This is only dangerous if x % p was the ONLY missing class for p.
            # i.e., len(missing[p]) == 1 and x % p in missing[p].
            if len(missing[p]) == 1 and (x % p) in missing[p]:
                safe = False
                break
        
        if safe:
            # Add x
            point_set.add(x)
            # Update missing structure
            for p in primes_to_check:
                r = x % p
                if r in missing[p]:
                    missing[p].remove(r)
            # Also, new primes might become relevant if len(point_set) increased?
            # The condition is p <= len(point_set).
            # If we add a point, len increases by 1.
            # A prime p where p == len(point_set) was not checked before, but now is.
            # We need to check if this new p is bad.
            new_p = len(point_set)
            if new_p in missing:
                if len(missing[new_p]) == 0:
                    # It's bad! We should not have added x.
                    # Remove x and revert.
                    point_set.remove(x)
                    for p in primes_to_check:
                        r = x % p
                        missing[p].add(r)
                    safe = False
                    break
            else:
                # We didn't track this prime before. Let's check it.
                covered = set()
                for v in point_set:
                    covered.add(v % new_p)
                if len(covered) == new_p:
                    # It's bad!
                    point_set.remove(x)
                    for p in primes_to_check:
                        r = x % p
                        missing[p].add(r)
                    safe = False
                    break
    
    return list(point_set)


def construct():
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    best = []
    
    # Run multiple randomized trials
    for seed in range(1000):
        if time.time() > deadline:
            break
        try:
            points = _smart_local_search(seed, deadline)
            if len(points) > len(best):
                best = points
        except Exception:
            continue
    
    # If we haven't found anything, fall back to deterministic greedy
    if not best:
        best = _greedy_construct(deadline)
        
    return best
# EVOLVE-BLOCK-END
