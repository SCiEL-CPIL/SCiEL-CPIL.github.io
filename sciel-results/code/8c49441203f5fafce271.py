"""Combined program: Cantor set seed + many randomized greedy restarts + fast swap local search with bitset verification."""
import time
import random

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

# Precompute all 4-AP triples that include each position x
# For each x, store (a, b, c) where {x, a, b, c} forms a 4-AP and x is one of the terms
# This lets us quickly find blockers
_ap_triples = [None] * (N + 1)
for x in range(1, N + 1):
    triples = []
    for d in _pos_last[x]:
        a, b, c = x - 3*d, x - 2*d, x - d
        if a >= 1:
            triples.append((a, b, c))
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2*d, x + 3*d
        if c <= N:
            triples.append((a, b, c))
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2*d
        if a >= 1 and c <= N:
            triples.append((a, b, c))
    for d in _pos_third[x]:
        a, b, c = x - 2*d, x - d, x + d
        if a >= 1 and c <= N:
            triples.append((a, b, c))
    _ap_triples[x] = triples

# Precompute constraint counts
_constraint_counts = [0] * (N + 1)
for x in range(1, N + 1):
    _constraint_counts[x] = len(_ap_triples[x])

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for a, b, c in _ap_triples[x]:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            return False
    return True

def _find_blockers(x, bits):
    """Find all elements that block x from being added. Returns a set."""
    blockers = set()
    for a, b, c in _ap_triples[x]:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            blockers.update([a, b, c])
    return blockers


def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for a, b, c in _ap_triples[x]:
            if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
                return False
    return True


def _greedy_forward_bitset():
    """Fast forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_reverse_order_bitset():
    """Fast reverse-order greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random_order_bitset(rng):
    """Random-order greedy using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random_reverse_bitset(rng):
    """Random-order greedy with reverse bias using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    # Bias towards larger numbers first
    order.sort(key=lambda x: -x if rng.random() < 0.7 else x)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_density_weighted_bitset(rng):
    """
    Greedy that weights positions by their constraint density.
    Lower numbers have fewer potential APs (can't be last term of long APs),
    so they're "easier" to include. We bias towards including them.
    """
    # Use precomputed constraint counts
    alpha = rng.uniform(0.3, 0.7)
    weighted_order = []
    for x in range(1, N + 1):
        count = _constraint_counts[x]
        # Lower constraint count -> higher priority
        weight = 1.0 / (1.0 + count * alpha)
        weighted_order.append((x, weight))
    
    # Sample from weighted order with some randomness
    order = []
    remaining = list(weighted_order)
    while remaining:
        if not remaining:
            break
        # Pick a random element with probability proportional to weight
        total_weight = sum(w for _, w in remaining)
        r = rng.random() * total_weight
        cum = 0
        for i, (x, w) in enumerate(remaining):
            cum += w
            if cum >= r:
                order.append(x)
                remaining.pop(i)
                break
    
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_hybrid_bitset(rng):
    """
    Hybrid greedy: start with low-constraint positions, then fill in randomly.
    """
    # Phase 1: Add low-constraint positions first
    constraint_counts = [( _constraint_counts[x], x) for x in range(1, N + 1)]
    constraint_counts.sort()
    
    bits = 0
    added = set()
    
    # Add lowest-constraint positions first (with some randomization)
    n_phase1 = rng.randint(100, 200)
    for count, x in constraint_counts[:n_phase1]:
        if x not in added and _can_add_bitset(x, bits):
            bits |= (1 << x)
            added.add(x)
    
    # Phase 2: Fill in remaining positions randomly
    remaining = [x for x in range(1, N + 1) if x not in added]
    rng.shuffle(remaining)
    for x in remaining:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
            added.add(x)
    
    return bits

def _greedy_generalized_cantor(rng):
    """
    Generalized Cantor set: in base 3, allow digits {0,1} with some probability of allowing digit 2
    at each position. This creates diverse structured seeds.
    """
    # Each base-3 digit position has a probability of allowing '2'
    # Higher positions (more significant) have lower probability of allowing '2'
    max_digits = 6  # 3^6 = 729 > 512
    allow_prob = [0.0] * max_digits  # Start with standard Cantor
    for d in range(max_digits):
        # Allow '2' with increasing probability for less significant digits
        allow_prob[d] = rng.uniform(0.0, 0.3 * (d + 1) / max_digits)
    
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        for d in range(max_digits):
            digit = t % 3
            if digit == 2 and rng.random() > allow_prob[d]:
                valid = False
                break
            t //= 3
            if t == 0:
                break
        if valid and _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}


def _local_improve_bitset(bits, deadline):
    """Fast local search using bitset with improved swap strategy."""
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current_bits >> x) & 1) and _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
        
        # Pass 2: Try swaps (remove b, add x) - use precomputed triples
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers using precomputed triples
            blockers = _find_blockers(x, current_bits)
            
            if not blockers:
                continue
                
            # Try up to 5 unique blockers
            if len(blockers) > 5:
                continue
                
            # Sort blockers by constraint count (try removing low-constraint ones first)
            sorted_blockers = sorted(blockers, key=lambda b: _constraint_counts[b])
            
            for b in sorted_blockers[:5]:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Greedy re-fill on candidate
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, N + 1):
                            if time.time() > deadline - 2:
                                break
                            if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                candidate_bits |= (1 << y)
                                changed = True
                                
                    cand_size = bin(candidate_bits).count('1')
                    if cand_size > current_size:
                        current_bits = candidate_bits
                        current_size = cand_size
                        improved = True
                        break
    
    # Pass 3: Try 2-for-1 swaps (remove b1, b2, add x)
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    break
                if (current_bits >> x) & 1:
                    continue
                
                # Find blockers using precomputed triples
                blockers = _find_blockers(x, current_bits)
                
                if not blockers or len(blockers) > 8:
                    continue
                
                # Try pairs of blockers, sorted by constraint count
                blist = sorted(blockers, key=lambda b: _constraint_counts[b])
                for i in range(min(len(blist), 6)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, min(len(blist), 6)):
                        if time.time() > deadline - 2:
                            break
                        b1, b2 = blist[i], blist[j]
                        candidate_bits = current_bits & ~(1 << b1) & ~(1 << b2)
                        if _can_add_bitset(x, candidate_bits):
                            candidate_bits |= (1 << x)
                            
                            # Greedy re-fill
                            changed = True
                            while changed and time.time() < deadline - 2:
                                changed = False
                                for y in range(1, N + 1):
                                    if time.time() > deadline - 2:
                                        break
                                    if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                        candidate_bits |= (1 << y)
                                        changed = True
                                        
                            cand_size = bin(candidate_bits).count('1')
                            if cand_size > current_size:
                                current_bits = candidate_bits
                                current_size = cand_size
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    return current_bits


def _cantor_set_bits():
    """
    Generate a 4-AP-free set using the base-3 Cantor set construction.
    Numbers in [1, N] whose base-3 representation contains only 0s and 1s
    (no digit 2) form a 4-AP-free set.
    """
    bits = 0
    for x in range(1, N + 1):
        # Check if base-3 representation of x contains only 0s and 1s
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits


def _cantor_local_improve(bits, deadline):
    """
    Local improvement starting from a Cantor-set seed.
    Uses a more aggressive add-then-swap strategy.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current_bits >> x) & 1) and _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
        
        # Pass 2: Try swaps (remove b, add x) with re-fill - use precomputed triples
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers using precomputed triples
            blockers = _find_blockers(x, current_bits)
            
            if not blockers:
                continue
                
            # Only try if number of unique blockers is small
            if len(blockers) > 5:
                continue
                
            # Sort by constraint count
            sorted_blockers = sorted(blockers, key=lambda b: _constraint_counts[b])
            
            for b in sorted_blockers[:5]:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Greedy re-fill on candidate
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, N + 1):
                            if time.time() > deadline - 2:
                                break
                            if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                candidate_bits |= (1 << y)
                                changed = True
                                
                    cand_size = bin(candidate_bits).count('1')
                    if cand_size > current_size:
                        current_bits = candidate_bits
                        current_size = cand_size
                        improved = True
                        break
    
    return current_bits


def construct():
    deadline = time.time() + 55  # Safety margin within 60s limit

    # 1) Start with Cantor set (strong structured seed ~200 elements)
    cantor_bits = _cantor_set_bits()
    cantor_size = bin(cantor_bits).count('1')
    best_bits = cantor_bits
    best_size = cantor_size

    # 1b) Also try deterministic forward greedy
    fwd_bits = _greedy_forward_bitset()
    fwd_size = bin(fwd_bits).count('1')
    if fwd_size > best_size:
        best_bits = fwd_bits
        best_size = fwd_size

    # 1c) Deterministic reverse greedy
    rev_bits = _greedy_reverse_order_bitset()
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_bits = rev_bits
        best_size = rev_size

    # 1d) Try generalized Cantor variants
    rng_init = random.Random(0)
    for _ in range(3):
        gen_bits = _greedy_generalized_cantor(rng_init)
        gen_size = bin(gen_bits).count('1')
        if gen_size > best_size:
            best_bits = gen_bits
            best_size = gen_size

    # 2) Local improvement on Cantor seed (this is the key phase)
    if time.time() < deadline - 15:
        cantor_improved = _cantor_local_improve(cantor_bits, deadline)
        cantor_imp_size = bin(cantor_improved).count('1')
        if cantor_imp_size > best_size:
            best_bits = cantor_improved
            best_size = cantor_imp_size

    # 3) Many randomized restarts with diverse strategies
    rng = random.Random(42)
    while time.time() < deadline - 10:
        r = rng.random()
        if r < 0.2:
            bits = _greedy_random_order_bitset(rng)
        elif r < 0.4:
            bits = _greedy_random_reverse_bitset(rng)
        elif r < 0.6:
            bits = _greedy_density_weighted_bitset(rng)
        elif r < 0.8:
            bits = _greedy_hybrid_bitset(rng)
        else:
            bits = _greedy_generalized_cantor(rng)
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size
            # Also try local improvement on this promising set
            if time.time() < deadline - 8:
                improved = _local_improve_bitset(bits, deadline - 5)
                imp_size = bin(improved).count('1')
                if imp_size > best_size:
                    best_bits = improved
                    best_size = imp_size

    # 4) Final local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size
    
    # 5) Try a few more local improvements with different random seeds
    if time.time() < deadline - 3:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 3:
                break
            rng2 = random.Random(seed)
            bits = _greedy_density_weighted_bitset(rng2)
            size = bin(bits).count('1')
            if size >= best_size - 5:  # Close to best, worth improving
                improved = _local_improve_bitset(bits, deadline - 2)
                imp_size = bin(improved).count('1')
                if imp_size > best_size:
                    best_bits = improved
                    best_size = imp_size

    # 5) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 6) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        # Fallback to Cantor set (always valid)
        best_set = _bits_to_set(_cantor_set_bits())

    return sorted(best_set)
