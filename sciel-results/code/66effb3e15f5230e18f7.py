"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import math


def _sieve_primes(limit):
    """Return list of primes up to limit using a standard sieve."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i, v in enumerate(is_prime) if v]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple for the given primes."""
    for p in primes:
        if len(points) < p:
            continue
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def construct():
    """
    Construct the largest admissible tuple within [0, N].
    
    Strategy: Greedy construction with smart point selection.
    We build the set incrementally, adding points that don't cause
    any prime p <= current_size to have all p residue classes covered.
    
    We use a priority-based approach: try to add points from [0, N]
    in an order that maximizes final size.
    """
    primes = _sieve_primes(N)
    
    # We'll try a different approach: start with all integers and remove
    # to make admissible. This is more direct.
    
    points = set(range(N + 1))
    
    # For each prime, we need at least one residue class to be missing.
    # We process primes in increasing order.
    # For each prime p, if all p classes are present, remove the class
    # with the fewest points.
    
    # Sort primes to process smaller ones first (they have tighter constraints)
    sorted_primes = sorted(primes)
    
    for p in sorted_primes:
        # Only need to check if len(points) >= p
        if len(points) < p:
            continue
            
        # Count points in each residue class mod p
        class_counts = [0] * p
        class_members = [[] for _ in range(p)]
        
        for v in points:
            r = v % p
            class_counts[r] += 1
            class_members[r].append(v)
        
        # If all classes are non-empty, we have a problem
        if min(class_counts) > 0:
            # Remove the class with the fewest members
            min_count = min(class_counts)
            # Find the first class with this minimum count
            for r in range(p):
                if class_counts[r] == min_count:
                    # Remove all points in this class
                    for v in class_members[r]:
                        points.discard(v)
                    break
    
    # After removal, we may need to re-check larger primes since removal
    # of points can only help (fewer residues covered), so admissibility
    # is preserved or improved.
    
    return sorted(points)
# EVOLVE-BLOCK-END
