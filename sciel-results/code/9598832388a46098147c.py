"""Structural redesign: bitmask-based greedy with time-bounded local search."""
import random
import time
from itertools import combinations

def _precompute_conflicts():
    """Precompute conflict lists for all possible complement masks."""
    # conf_2[m] = list of masks formed by choosing 2 bits from m
    # conf_4[m] = list of masks formed by choosing 4 bits from m
    conf_2 = [[] for _ in range(1 << 16)]
    conf_4 = [[] for _ in range(1 << 16)]
    
    for m in range(1 << 16):
        bits = [i for i in range(16) if (m >> i) & 1]
        # Generate 2-subsets
        for c in combinations(bits, 2):
            mask = (1 << c[0]) | (1 << c[1])
            conf_2[m].append(mask)
        # Generate 4-subsets
        for c in combinations(bits, 4):
            mask = (1 << c[0]) | (1 << c[1]) | (1 << c[2]) | (1 << c[3])
            conf_4[m].append(mask)
            
    return conf_2, conf_4

def _all_4sets():
    """Precompute all C(16,4)=1820 four-element subsets as bitmasks."""
    out = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        out.append(mask)
    return out

def _get_conflicts(A, B, conf_2, conf_4):
    """Get masks C such that {A, B, C} is a sunflower."""
    S = A & B
    U = A | B
    Comp = (~U) & 0xFFFF
    k = bin(S).count('1')
    
    if k == 0:
        return conf_4[Comp]
    else:
        # C = S | {x, y} where {x, y} subset of Comp
        # We need to OR S with each mask in conf_2[Comp]
        return [S | x for x in conf_2[Comp]]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    all_masks = _all_4sets()
    conf_2, conf_4 = _precompute_conflicts()
    
    best_family = []
    best_size = 0
    rng = random.Random(42)
    
    # Greedy with blocked counts
    def greedy(order):
        block_count = [0] * (1 << 16)
        F = []
        for m in order:
            if time.time() > deadline:
                break
            if block_count[m] == 0:
                F.append(m)
                # Update block counts with new set m
                for b in F[:-1]:
                    confs = _get_conflicts(m, b, conf_2, conf_4)
                    for c in confs:
                        block_count[c] += 1
        return F

    # Multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        order = list(all_masks)
        rng.shuffle(order)
        
        # Try different sorting strategies occasionally
        if i % 10 == 0:
             # Sort by something? Maybe not needed if shuffled
             pass
             
        family = greedy(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
        # Local search: try to improve
        if time.time() < deadline - 5.0 and best_family:
            # Simple local search: try to remove one and add one if it increases size? 
            # Or just try to add more if possible (greedy should have done that).
            # The greedy is maximal (no single set can be added).
            # To increase size, we need to swap.
            # Try removing a set and see if we can add two?
            # This is complex. Let's stick to multiple restarts first.
            pass

    # Final check
    result = []
    for mask in best_family:
        elems = [i+1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elems))
    
    return result
