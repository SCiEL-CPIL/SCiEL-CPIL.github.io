"""Combined program: Bitset-based greedy restarts with density-weighted strategies and enhanced local search."""
import time
import random

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
# x is the 4th term: x-3d, x-2d, x-d, x
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
# x is the 1st term: x, x+d, x+2d, x+3d
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
# x is the 2nd term: x-d, x, x+d, x+2d
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
# x is the 3rd term: x-2d, x-d, x, x+d
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

def _greedy_density_weighted_bitset(rng):
    """
    Greedy that weights positions by their constraint density.
    Lower numbers have fewer potential APs (can't be last term of long APs),
    so they're "easier" to include. We bias towards including them.
    """
    # Compute constraint count for each position
    # Pre-computed counts for speed
    constraint_counts = []
    for x in range(1, N + 1):
        count = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
        constraint_counts.append((x, count))
    
    # Sort by constraint count (ascending), with some randomness
    alpha = rng.uniform(0.3, 0.7)
    weighted_order = []
    for x, count in constraint_counts:
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
    constraint_counts = []
    for x in range(1, N + 1):
        count = len(_pos_last[x]) + len(_pos_first[x]) + len(_pos_second[x]) + len(_pos_third[x])
        constraint_counts.append((count, x))
    
    constraint_counts.sort()
    
    bits = 0
    added = set()
    
    # Add lowest-constraint positions first (with some randomization)
    # Take a random subset of the low constraint ones to add diversity
    low_constraint_indices = rng.sample(range(len(constraint_counts)), min(150, len(constraint_counts)))
    for idx in low_constraint_indices:
        count, x = constraint_counts[idx]
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
        
        # Pass 2: Try swaps (remove b, add x)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers
            blockers = set()
            for d in _pos_last[x]:
                a, b, c = x - d, x - 2 * d, x - 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_first[x]:
                a, b, c = x + d, x + 2 * d, x + 3 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_second[x]:
                a, b, c = x - d, x + d, x + 2 * d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            for d in _pos_third[x]:
                a, b, c = x - 2 * d, x - d, x + d
                if ((current_bits >> a) & 1) and ((current_bits >> b) & 1) and ((current_bits >> c) & 1):
                    blockers.update([a, b, c])
            
            if not blockers:
                continue # Should have been caught in Pass 1, but just in case
                
            # Only try if number of unique blockers is small (e.g., <= 3)
            # If many blockers, swap is unlikely to help or too expensive.
            if len(blockers) > 3:
                continue
                
            for b in blockers:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Greedy re-fill on candidate to see if we can gain size
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
                        break # Restart pass 2 for this x? Or just continue. 
                        # Breaking inner loop, continue outer.
    
    return current_bits

def construct():
    deadline = time.time() + 80  # Safety margin within 90s limit

    # 1) Deterministic forward greedy (fast)
    best_bits = _greedy_forward_bitset()
    best_size = bin(best_bits).count('1')

    # 1b) Deterministic reverse greedy
    rev_bits = _greedy_reverse_order_bitset()
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_bits = rev_bits
        best_size = rev_size

    # 2) Many randomized restarts with diverse strategies
    rng = random.Random(42)
    while time.time() < deadline - 15:
        r = rng.random()
        if r < 0.3:
            bits = _greedy_random_order_bitset(rng)
        elif r < 0.6:
            bits = _greedy_density_weighted_bitset(rng)
        elif r < 0.8:
            bits = _greedy_hybrid_bitset(rng)
        else:
            # Random reverse bias
            order = list(range(1, N + 1))
            rng.shuffle(order)
            order.sort(key=lambda x: -x if rng.random() < 0.7 else x)
            bits = 0
            for x in order:
                if _can_add_bitset(x, bits):
                    bits |= (1 << x)
        
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size

    # 3) Local improvement on the best set found
    if time.time() < deadline - 5:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size

    # 4) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 5) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        best_set = _bits_to_set(_greedy_forward_bitset())

    return sorted(best_set)
