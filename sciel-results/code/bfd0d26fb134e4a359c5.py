import time

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def _count_residues(points, p):
    """Count how many points fall in each residue class mod p."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    return counts


def _score_removal(points, p, target_class, primes_up_to_N, current_len):
    """
    Heuristic score for removing residue class `target_class` mod p.
    We want to minimize future damage: prefer removing classes that
    are less 'important' for other primes.
    
    Score = number of points removed + penalty for how many other primes
    would still be satisfied after removal (lower penalty = better).
    """
    removed_count = sum(1 for v in points if v % p == target_class)
    
    # Estimate future damage: for each other prime q <= current_len,
    # check if removing this class would make q's coverage worse.
    # A simpler heuristic: prefer removing the class that has the most
    # uniform distribution across other primes (less concentrated).
    
    # Actually, let's use a simpler but effective heuristic:
    # Remove the class with fewest elements. If tied, prefer the one
    # whose elements are more evenly spread across small primes.
    
    return removed_count


def construct():
    # Start from the full range [0, N] and repair admissibility.
    # For each prime p <= len(points), if all p residue classes are covered,
    # remove the elements in the least-populated class. Repeat until stable.
    # This directly maximizes the tuple size by removing the minimum necessary.
    
    points = list(range(N + 1))
    
    # Cache primes up to N+1
    primes_up_to_N = _primes(N + 1)
    
    # Iterate repair
    # We'll loop until no change or deadline
    while time.time() < _DEADLINE:
        changed = False
        current_len = len(points)
        if current_len < 2:
            break
        
        # Get primes up to current_len, sorted by how "urgent" they are
        # Urgency: primes where all classes are covered AND the min class size is small
        # (removing a small class costs less)
        primes_to_check = [p for p in primes_up_to_N if p <= current_len]
        
        # Score each prime that needs fixing
        candidates = []
        for p in primes_to_check:
            counts = _count_residues(points, p)
            if min(counts) == 0:
                continue
            min_count = min(counts)
            # Cost of fixing this prime = min_count (elements to remove)
            # Prefer fixing primes with smaller cost first? Or larger?
            # Actually, we should fix the one that's most "constrained" first.
            # A prime with min_count=1 is more urgent than one with min_count=100.
            candidates.append((min_count, p, counts))
        
        if not candidates:
            break
        
        # Sort by min_count (ascending) - fix the most constrained first
        candidates.sort(key=lambda x: x[0])
        
        # Try the most constrained prime first
        min_count, p, counts = candidates[0]
        
        # Find all classes with min_count
        min_classes = [c for c in range(p) if counts[c] == min_count]
        
        # For tie-breaking, use a heuristic: pick the class whose removal
        # causes least damage to other primes' admissibility.
        # Simple approach: pick the class with fewest elements that are "shared"
        # with other critical residue classes.
        
        # For now, just pick the first one (smallest index) for simplicity
        target_class = min_classes[0]
        
        # Remove all elements congruent to target_class mod p
        points = [v for v in points if v % p != target_class]
        changed = True
    
    return points
# EVOLVE-BLOCK-END
