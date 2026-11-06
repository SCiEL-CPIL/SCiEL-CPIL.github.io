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

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for d in _pos_last[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x - 2 * d)) & 1 and (bits >> (x - 3 * d)) & 1:
            return False
    for d in _pos_first[x]:
        if (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1 and (bits >> (x + 3 * d)) & 1:
            return False
    for d in _pos_second[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1:
            return False
    for d in _pos_third[x]:
        if (bits >> (x - 2 * d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1:
            return False
    return True


def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for d in _pos_last[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - 3 * d)) & 1):
                return False
        for d in _pos_first[x]:
            if ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1) and ((bits >> (x + 3 * d)) & 1):
                return False
        for d in _pos_second[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1):
                return False
        for d in _pos_third[x]:
            if ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1):
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
    """Random reverse-order greedy using bitset."""
    order = list(range(N, 0, -1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_density_weighted_bitset(rng):
    """Density-weighted greedy: prioritize positions with fewer future constraints."""
    # Precompute a static "density score" for each position:
    # score[x] = number of 4-APs that x participates in (as any position).
    # Lower score = less constrained = better to pick early? Or higher?
    # Actually, we want to pick elements that block fewer future options.
    # Let's define weight[x] = number of d values across all positions.
    weights = [0] * (N + 1)
    for x in range(1, N + 1):
        weights[x] = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
    
    order = list(range(1, N + 1))
    # Sort by weight descending (more constrained first) or ascending?
    # Let's try ascending (less constrained first) to leave room.
    # But randomization is key. Let's shuffle with a bias.
    # Simple approach: sort by weight, then shuffle within buckets? 
    # Or just use weight as a key in a weighted shuffle.
    # For simplicity, let's just sort by weight and then do a random greedy on that sorted list?
    # No, let's just shuffle normally but this function is a placeholder for a specific strategy.
    # Let's try: sort by weight ascending, then take from that order.
    order.sort(key=lambda x: weights[x])
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_hybrid_bitset(rng):
    """Hybrid: start with forward greedy, then try to add random elements."""
    bits = _greedy_forward_bitset()
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not ((bits >> x) & 1) and _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}


def _local_improve_bitset(bits, deadline):
    """Fast local search using bitset."""
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
        
        # Pass 2: Try swaps (remove b, add x)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers
            blockers = []
            for d in _pos_last[x]:
                a, b, c = x - d, x - 2 * d, x - 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_first[x]:
                a, b, c = x + d, x + 2 * d, x + 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_second[x]:
                a, b, c = x - d, x + d, x + 2 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            for d in _pos_third[x]:
                a, b, c = x - 2 * d, x - d, x + d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.extend([a, b, c])
            
            if not blockers:
                continue
                
            unique_blockers = set(blockers)
            if len(unique_blockers) > 3:
                continue
                
            for b in unique_blockers:
                if time.time() > deadline - 2:
                    break
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
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


def _cantor_set_bits():
    """Generate the Cantor set (base-3 digits with no '2') within [1, N]."""
    bits = 0
    for x in range(1, N + 1):
        # Check if x in base 3 has no digit 2
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
    """Local improvement starting from Cantor set."""
    return _local_improve_bitset(bits, deadline)


def construct():
    deadline = time.time() + 80  # Safety margin within 90s limit

    # 1) Deterministic forward greedy (fast)
    best_bits = _greedy_forward_bitset()
    best_size = bin(best_bits).count('1')

    # 2) Cantor set seed (strong structured start)
    cantor_bits = _cantor_set_bits()
    cantor_size = bin(cantor_bits).count('1')
    if cantor_size > best_size:
        best_bits = cantor_bits
        best_size = cantor_size

    # 3) Many randomized restarts
    rng = random.Random(42)
    while time.time() < deadline - 15:
        # Mix different greedy strategies
        r = rng.random()
        if r < 0.3:
            bits = _greedy_random_order_bitset(rng)
        elif r < 0.5:
            bits = _greedy_random_reverse_bitset(rng)
        elif r < 0.7:
            bits = _greedy_density_weighted_bitset(rng)
        elif r < 0.85:
            bits = _greedy_hybrid_bitset(rng)
        else:
            bits = _greedy_forward_bitset()
        
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size

    # 4) Local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size

    # 5) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 6) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        best_bits = _greedy_forward_bitset()
        best_set = _bits_to_set(best_bits)

    return sorted(best_set)
