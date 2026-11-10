"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log2

def _build_conflict_bitsets(n):
    """Return list of Python ints as bitsets for conflicts of each node 1..n."""
    conflicts = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)
    return conflicts

def _greedy_with_bitsets(conflicts, n, order=None):
    """Greedy max independent set using bitset conflict checks."""
    if order is None:
        order = list(range(1, n + 1))
    selected = 0  # bitset of selected
    for v in order:
        bit = 1 << v
        if (selected & conflicts[v]) == 0:
            selected |= bit
    return selected

def _count_bits(x):
    """Popcount for Python int."""
    return bin(x).count('1')

def _local_search_bitsets(conflicts, n, selected, deadline, max_iters=50000):
    """Local search with bitset operations for speed."""
    selected_mask = selected
    best = selected_mask
    best_size = _count_bits(selected_mask)
    all_mask = (1 << (n + 1)) - 2  # bits 1..n set
    unselected = all_mask & ~selected_mask
    
    def can_add_bitset(v_bit, sel):
        v = v_bit.bit_length() - 1
        return (sel & conflicts[v]) == 0
    
    # Phase 1: Greedy additions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add any unselected vertex
        candidates = unselected
        while candidates:
            # Pick lowest set bit
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            if (selected_mask & conflicts[v]) == 0:
                selected_mask |= lsb
                unselected ^= lsb
                candidates ^= lsb
                changed = True
                if _count_bits(selected_mask) > best_size:
                    best = selected_mask
                    best_size = _count_bits(selected_mask)
            else:
                candidates ^= lsb
    
    # Phase 2: 1-for-1 swaps
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        if not unselected:
            break
        # Pick a random unselected vertex
        bits = []
        temp = unselected
        while temp:
            lsb = temp & (-temp)
            bits.append(lsb)
            temp ^= lsb
        if not bits:
            break
        v_bit = random.choice(bits)
        v = v_bit.bit_length() - 1
        
        blockers = selected_mask & conflicts[v]
        if blockers == 0:
            selected_mask |= v_bit
            unselected ^= v_bit
            if _count_bits(selected_mask) > best_size:
                best = selected_mask
                best_size = _count_bits(selected_mask)
            continue
        
        # Try removing one blocker
        b_bits = []
        temp = blockers
        while temp:
            lsb = temp & (-temp)
            b_bits.append(lsb)
            temp ^= lsb
        
        for b_bit in b_bits:
            new_sel = selected_mask & ~b_bit
            if (new_sel & conflicts[v]) == 0:
                # Check if we gain or break tie with potential further additions
                new_size = _count_bits(new_sel)
                if new_size >= best_size:
                    selected_mask = new_sel | v_bit
                    unselected = (unselected | b_bit) & ~v_bit
                    if _count_bits(selected_mask) > best_size:
                        best = selected_mask
                        best_size = _count_bits(selected_mask)
                    break
    
    # Phase 3: 1-for-2 swaps
    for _ in range(max_iters // 2):
        if time.time() >= deadline:
            break
        if not unselected:
            break
        bits = []
        temp = unselected
        while temp:
            lsb = temp & (-temp)
            bits.append(lsb)
            temp ^= lsb
        if not bits:
            break
        v_bit = random.choice(bits)
        v = v_bit.bit_length() - 1
        
        blockers = selected_mask & conflicts[v]
        b_bits = []
        temp = blockers
        while temp:
            lsb = temp & (-temp)
            b_bits.append(lsb)
            temp ^= lsb
        
        if len(b_bits) == 2:
            b1, b2 = b_bits
            new_sel = selected_mask & ~b1 & ~b2
            if (new_sel & conflicts[v]) == 0:
                selected_mask = new_sel | v_bit
                unselected = (unselected | b1 | b2) & ~v_bit
                if _count_bits(selected_mask) > best_size:
                    best = selected_mask
                    best_size = _count_bits(selected_mask)
    
    # Final greedy fill
    candidates = unselected
    while candidates and time.time() < deadline:
        lsb = candidates & (-candidates)
        v = lsb.bit_length() - 1
        if (selected_mask & conflicts[v]) == 0:
            selected_mask |= lsb
            unselected ^= lsb
            if _count_bits(selected_mask) > best_size:
                best = selected_mask
                best_size = _count_bits(selected_mask)
        candidates ^= lsb
    
    return best

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill
    
    n = 256
    conflicts = _build_conflict_bitsets(n)
    
    # Start with all odds (known valid, size 128)
    odds_mask = 0
    for i in range(1, 257, 2):
        odds_mask |= (1 << i)
    
    # Greedy with different orderings
    best = _greedy_with_bitsets(conflicts, n, list(range(1, n + 1)))
    if _count_bits(best) < _count_bits(odds_mask):
        best = odds_mask
    
    # Try greedy with random orderings
    for _ in range(100):
        if time.time() >= deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = _greedy_with_bitsets(conflicts, n, order)
        if _count_bits(s) > _count_bits(best):
            best = s
    
    # Local search from best
    best = _local_search_bitsets(conflicts, n, best, deadline)
    
    # Convert bitset to sorted list
    result = []
    for i in range(1, n + 1):
        if best & (1 << i):
            result.append(i)
    
    return result
# EVOLVE-BLOCK-END
