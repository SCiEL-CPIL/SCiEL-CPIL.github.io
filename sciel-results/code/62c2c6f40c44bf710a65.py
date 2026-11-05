"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def is_squarefree(n):
        """Check if n is squarefree."""
        if n <= 1:
            return True
        tmp = n
        p = 2
        while p * p <= tmp:
            if tmp % (p * p) == 0:
                return False
            tmp //= p
            while tmp % p == 0:
                tmp //= p
            p += 1 if p == 2 else 2
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit using a^2 * b^3 form."""
        powerful = set()
        max_b = int(round(limit ** (1.0 / 3.0))) + 2

        # Precompute squarefree numbers up to max_b
        sqfree_list = []
        for b in range(1, max_b + 1):
            if is_squarefree(b):
                b3 = b * b * b
                if b3 <= limit:
                    sqfree_list.append((b, b3))

        for b, b3 in sqfree_list:
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)

        return sorted(powerful)

    def count_representations(powerful):
        """Count number of representations N = a + b with a <= b, a, b in powerful."""
        n_pows = len(powerful)
        counts = defaultdict(int)
        pair_lists = defaultdict(list)
        
        # Use a two-pointer approach or nested loops
        # For each a, iterate over b >= a
        for i in range(n_pows):
            a = powerful[i]
            for j in range(i, n_pows):
                b = powerful[j]
                N = a + b
                if N > 2 * powerful[-1]:
                    break
                counts[N] += 1
                pair_lists[N].append((a, b))
        
        return counts, pair_lists

    # Strategy: Generate powerful numbers up to increasing limits and find N with max representations
    # Start with a moderate limit that's fast, then increase if time permits
    
    best_count = 0
    best_N = 2
    best_pairs = []

    # Try different limits
    limits_to_try = [100000, 500000, 2000000, 10000000, 50000000]

    for L in limits_to_try:
        if time.time() > deadline - 15:
            break

        powerful = generate_powerful(L)
        n_pows = len(powerful)
        
        if n_pows < 2:
            continue

        # Count representations
        counts = defaultdict(int)
        pair_lists = defaultdict(list)
        
        for i in range(n_pows):
            a = powerful[i]
            if 2 * a > 2 * L:
                break
            for j in range(i, n_pows):
                b = powerful[j]
                N = a + b
                if N > 2 * L:
                    break
                counts[N] += 1
                pair_lists[N].append((a, b))
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break

        # Find best N in this batch
        for N, cnt in counts.items():
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pair_lists[N]

        # If we found a good result, we can stop
        if best_count >= 50:
            break

    # Verify and return
    pairs = [[a, b] for a, b in best_pairs if a <= b and a + b == best_N]
    # Remove duplicates and sort
    pairs = sorted(set(pairs))
    
    # Verify that N is within bounds
    if best_N < 2 or best_N > 10**18:
        # Fallback
        best_N = 2
        best_pairs = []
        pairs = []

    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
