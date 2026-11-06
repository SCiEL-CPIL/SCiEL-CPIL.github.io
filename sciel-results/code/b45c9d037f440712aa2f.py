"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random

# EVOLVE-BLOCK-START

def _build_sorted_masks(n):
    """Build frozenset of all sorted masks (0^a * 1^b) for n bits."""
    masks = set()
    for k in range(n + 1):
        # k ones in the lowest k bits: bits 0..k-1 are 1, bits k..n-1 are 0
        # Wait: sorted ascending means 0,0,...,0,1,1,...,1
        # i.e., bits 0..(n-k-1) are 0, bits (n-k)..n-1 are 1
        # mask = ((1 << k) - 1) << (n - k)
        mask = ((1 << k) - 1) << (n - k)
        masks.add(mask)
    return frozenset(masks)

_SORTED_MASKS = _build_sorted_masks(N)

def _fast_verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using precomputed sorted mask set."""
    sorted_set = _SORTED_MASKS
    for mask in range(1 << n):
        state = mask
        for i, j in comparators:
            if (state >> i) & 1 == 0 and (state >> j) & 1 == 1:
                state ^= (1 << i) | (1 << j)
        if state not in sorted_set:
            return False
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def construct():
    deadline = time.time() + 85  # 5 second safety margin before 90s kill
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    random.seed(42)
    current = list(best)
    
    # Verify initial
    if not _fast_verify_network(current):
        return best
    
    # Local search
    max_iterations = 200000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        
        # Try random modification
        mod_type = random.random()
        
        if mod_type < 0.5 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                # Greedy cascade: try to remove more
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    if len(current) <= 70:
                        break
                    idx2 = random.randint(0, len(current) - 1)
                    candidate2 = current[:idx2] + current[idx2+1:]
                    if _fast_verify_network(candidate2):
                        current = candidate2
                        if len(current) < best_len:
                            best = list(current)
                            best_len = len(current)
        
        elif mod_type < 0.75:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        else:
            # Try replacing a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            # Generate random valid comparator, biased toward non-adjacent
            while True:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a != b:
                    if a > b:
                        a, b = b, a
                    break
            candidate = list(current)
            candidate[idx] = (a, b)
            if _fast_verify_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
    
    return best

# EVOLVE-BLOCK-END
