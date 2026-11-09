"""Fast bitset-based search for reciprocal weight of 3-progression-free sets on [1,256].

Combines:
- Bitset representation + precomputed AP-pairs (fast can_add)
- Base-3 Cantor set inspired initializations
- Multi-stage local search with 1-for-2 and 2-for-3 swaps
- Multiple randomized greedy initializations for diverse starting points
"""
import time
import random
from fractions import Fraction

N = 256
TIME_LIMIT = 55.0  # safety margin before 60s kill

# Precompute reciprocal Fractions
recip_frac = [Fraction(0, 1)] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} forms a 3-AP with a,b != x
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        # Case 1: x is first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        # Case 2: x is middle term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        # Case 3: x is last term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

def can_add(bitset, x):
    """Check if adding x to bitset creates a 3-AP. Returns True if safe."""
    for a, b in ap_pairs[x]:
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def bitset_to_list(bitset):
    """Convert bitset to sorted list of elements."""
    return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]

def list_to_bitset(lst):
    b = 0
    for x in lst:
        b |= (1 << (x - 1))
    return b

def bitset_score(bitset):
    """Exact reciprocal sum of a bitset."""
    total = Fraction(0, 1)
    for i in range(1, N + 1):
        if bitset & (1 << (i - 1)):
            total += recip_frac[i]
    return total

def greedy_construct(order):
    """Greedy: add elements in given order if they don't create a 3-AP."""
    bitset = 0
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
    return bitset

def greedy_by_reciprocal():
    """Greedy in order of decreasing 1/x (smallest x first)."""
    return greedy_construct(list(range(1, N + 1)))

def cantor_set_elements():
    """Generate elements whose base-3 representation contains only 0s and 1s (Cantor set).
    These are guaranteed to be 3-AP-free."""
    elements = set()
    for x in range(1, N + 1):
        # Check if base-3 representation has only 0s and 1s
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            elements.add(x)
    return sorted(elements)

def cantor_based_greedy():
    """Start with Cantor set elements, then greedily add more."""
    cantor = cantor_set_elements()
    bitset = 0
    for x in cantor:
        bitset |= (1 << (x - 1))
    # Now greedily add remaining elements in order of decreasing 1/x
    for x in range(1, N + 1):
        if not (bitset & (1 << (x - 1))):
            if can_add(bitset, x):
                bitset |= (1 << (x - 1))
    return bitset

def greedy_randomized(seed, trials=10):
    """Multiple randomized greedy runs, return best bitset."""
    rng = random.Random(seed)
    best_bitset = 0
    best_score = Fraction(0, 1)
    for _ in range(trials):
        candidates = list(range(1, N + 1))
        weights = [1.0 / x for x in candidates]
        noisy = [w * (0.5 + rng.random()) for w in weights]
        order = sorted(candidates, key=lambda x: noisy[x - 1], reverse=True)
        b = greedy_construct(order)
        s = bitset_score(b)
        if s > best_score:
            best_score = s
            best_bitset = b
    return best_bitset

def local_search_single_swap(bitset, deadline):
    """
    Local search: try all single-element swaps (remove one, add another)
    to find the best improvement. Repeat until no improvement.
    """
    current = bitset
    current_score = bitset_score(current)
    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        best_delta = Fraction(0, 1)
        best_remove = None
        best_add = None

        # Get current elements
        elements = bitset_to_list(current)

        for removed in elements:
            temp = current & ~(1 << (removed - 1))
            # Try adding each candidate not in temp
            for add_candidate in range(1, N + 1):
                if temp & (1 << (add_candidate - 1)):
                    continue
                if can_add(temp, add_candidate):
                    delta = recip_frac[add_candidate] - recip_frac[removed]
                    if delta > best_delta:
                        best_delta = delta
                        best_remove = removed
                        best_add = add_candidate

        if best_remove is not None and best_delta > 0:
            current = (current & ~(1 << (best_remove - 1))) | (1 << (best_add - 1))
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            break

    return best, best_score

def local_search_1for2_swap(bitset, deadline):
    """
    Local search with 1-for-2 swaps: try removing one element and adding two
    smaller elements (net gain in reciprocal sum). This can increase the
    reciprocal sum significantly.
    """
    current = bitset
    current_score = bitset_score(current)
    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        best_delta = Fraction(0, 1)
        best_remove = None
        best_add1 = None
        best_add2 = None

        elements = bitset_to_list(current)

        for removed in elements:
            temp = current & ~(1 << (removed - 1))
            # Try all pairs (a, b) with a < b, both not in temp, both can be added
            # To limit search, only consider a, b < removed (smaller = better reciprocal)
            for a in range(1, removed):
                if temp & (1 << (a - 1)):
                    continue
                if not can_add(temp, a):
                    continue
                temp2 = temp | (1 << (a - 1))
                for b in range(a + 1, removed):
                    if temp2 & (1 << (b - 1)):
                        continue
                    if not can_add(temp2, b):
                        continue
                    delta = recip_frac[a] + recip_frac[b] - recip_frac[removed]
                    if delta > best_delta:
                        best_delta = delta
                        best_remove = removed
                        best_add1 = a
                        best_add2 = b

        if best_remove is not None and best_delta > 0:
            current = (current & ~(1 << (best_remove - 1))) | (1 << (best_add1 - 1)) | (1 << (best_add2 - 1))
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            break

    return best, best_score

def local_search_2for3_swap(bitset, deadline):
    """
    Local search with 2-for-3 swaps: try removing two elements and adding three
    smaller elements. This can further increase reciprocal sum.
    """
    current = bitset
    current_score = bitset_score(current)
    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        best_delta = Fraction(0, 1)
        best_remove1 = None
        best_remove2 = None
        best_add1 = None
        best_add2 = None
        best_add3 = None

        elements = bitset_to_list(current)
        n_elem = len(elements)
        
        # Limit combinations for performance
        max_pairs = min(n_elem, 15)
        
        for i in range(max_pairs):
            for j in range(i + 1, min(i + 10, n_elem)):
                r1 = elements[i]
                r2 = elements[j]
                temp = current & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                
                # Try adding three elements, all smaller than max(r1,r2)
                max_add = max(r1, r2)
                for a in range(1, max_add):
                    if temp & (1 << (a - 1)):
                        continue
                    if not can_add(temp, a):
                        continue
                    temp2 = temp | (1 << (a - 1))
                    for b in range(a + 1, max_add):
                        if temp2 & (1 << (b - 1)):
                            continue
                        if not can_add(temp2, b):
                            continue
                        temp3 = temp2 | (1 << (b - 1))
                        for c in range(b + 1, max_add):
                            if temp3 & (1 << (c - 1)):
                                continue
                            if not can_add(temp3, c):
                                continue
                            delta = recip_frac[a] + recip_frac[b] + recip_frac[c] - recip_frac[r1] - recip_frac[r2]
                            if delta > best_delta:
                                best_delta = delta
                                best_remove1 = r1
                                best_remove2 = r2
                                best_add1 = a
                                best_add2 = b
                                best_add3 = c

        if best_remove1 is not None and best_delta > 0:
            current = (current & ~(1 << (best_remove1 - 1)) & ~(1 << (best_remove2 - 1)))
            current |= (1 << (best_add1 - 1)) | (1 << (best_add2 - 1)) | (1 << (best_add3 - 1))
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            break

    return best, best_score

def try_add_all(bitset, deadline):
    """Try to add any element that can be added (greedy completion)."""
    current = bitset
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (current & (1 << (x - 1))):
                if can_add(current, x):
                    current |= (1 << (x - 1))
                    improved = True
    return current

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Phase 1: Get diverse starting points
    best_bitset = greedy_by_reciprocal()
    best_score = bitset_score(best_bitset)
    
    # Try Cantor-based initialization
    cantor_bitset = cantor_based_greedy()
    cantor_score = bitset_score(cantor_bitset)
    if cantor_score > best_score:
        best_score = cantor_score
        best_bitset = cantor_bitset

    seed = 0
    while time.time() < deadline - 30:
        candidate = greedy_randomized(seed, trials=15)
        if candidate:
            val = bitset_score(candidate)
            if val > best_score:
                best_score = val
                best_bitset = candidate
        seed += 1

    # Phase 2: Try adding all possible elements
    best_bitset = try_add_all(best_bitset, deadline - 25)
    best_score = bitset_score(best_bitset)

    # Phase 3: Single-swap local search
    best_bitset, best_score = local_search_single_swap(best_bitset, deadline - 15)

    # Phase 4: 1-for-2 swap local search (the key refinement)
    best_bitset, best_score = local_search_1for2_swap(best_bitset, deadline - 5)

    # Phase 5: 2-for-3 swap local search
    best_bitset, best_score = local_search_2for3_swap(best_bitset, deadline - 2)

    # Phase 6: Final single-swap polish
    best_bitset, best_score = local_search_single_swap(best_bitset, deadline)

    # Phase 7: Try adding any remaining elements
    best_bitset = try_add_all(best_bitset, deadline)

    return bitset_to_list(best_bitset)
