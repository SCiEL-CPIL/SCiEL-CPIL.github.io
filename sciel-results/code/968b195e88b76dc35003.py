"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _compute_reset_length_bitmask(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    # Precompute transition masks for each state and letter
    # trans_a[state] is the bit representing where state goes under 'a'
    # Actually, we need to map a subset (bitmask) to a new subset.
    # Let's precompute for each state i, the destination bit for a and b.
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    # BFS
    visited = {full_mask}
    # Queue stores (mask, distance)
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
        
        # Transition with 'a'
        new_mask_a = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
            
        # Transition with 'b'
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
            
    return -1

def _is_synchronizing_bitmask(a, b):
    """Check if automaton synchronizes using bitmask BFS."""
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    visited = {full_mask}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        if mask & (mask - 1) == 0:
            return True
        
        new_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append(new_mask_a)
            
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append(new_mask_b)
            
    return False

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0
    best_a = None
    best_b = None
    best_len = -1

    def evaluate(a, b):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return
        if _is_synchronizing_bitmask(a, b):
            length = _compute_reset_length_bitmask(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length

    # Strategy 1: Pinwheel construction
    # States 0..N-1. 'a' shifts: i -> (i+1) % N for i < N-1, and N-1 -> 0
    # 'b' maps: 0 -> 0, i -> i-1 for i >= 1
    # This is the classic pinwheel that achieves (N-1)^2 for odd N, close for even N
    for shift in range(N):
        a = [(i + 1 + shift) % N for i in range(N)]
        b = [0] + list(range(N - 1))
        evaluate(a, b)
    
    # Strategy 2: Rank-2 cycle construction
    # 'a': cycle 0->1->2->...->N-1->0
    # 'b': 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    a = list(range(1, N)) + [0]
    b = [0] + list(range(N - 1))
    evaluate(a, b)
    
    # Strategy 3: Modified pinwheel with different 'b' mappings
    # Try various 'b' functions that create rank-2 subsets
    for b_type in range(5):
        a = list(range(1, N)) + [0]  # cycle
        if b_type == 0:
            b = [0] * N  # all to 0 (trivial)
        elif b_type == 1:
            b = [0] + list(range(N - 1))  # standard
        elif b_type == 2:
            b = list(range(N - 1)) + [0]  # shift left, wrap
        elif b_type == 3:
            b = [(i + N // 2) % N for i in range(N)]  # half-shift
        elif b_type == 4:
            b = [N - 1 - i for i in range(N)]  # reverse
        evaluate(a, b)
    
    # Strategy 4: Systematic search over structured families
    # Family: 'a' is a cycle, 'b' has specific structure
    for a_shift in range(N):
        if time.time() > deadline:
            return [best_a, best_b] if best_a else [list(range(1, N)) + [0], [0] + list(range(N - 1))]
        a = [(i + 1 + a_shift) % N for i in range(N)]
        
        # b: maps most states to i-1, but with variations
        for b_var in range(20):
            if time.time() > deadline:
                return [best_a, best_b] if best_a else [list(range(1, N)) + [0], [0] + list(range(N - 1))]
            b = [0] + list(range(N - 1))
            # Perturb a few positions
            if b_var > 0:
                pos = b_var % N
                b[pos] = (b[pos] + 1) % N
            if b_var > 10:
                pos2 = (b_var + 3) % N
                b[pos2] = (b[pos2] + 2) % N
            evaluate(a, b)
    
    # Strategy 5: Known optimal constructions from literature
    # For N=12, the best known is around 66-70 (not full 110)
    # Try the "double cycle" construction
    for offset in range(N):
        if time.time() > deadline:
            break
        a = list(range(1, N)) + [0]
        b = [(i + offset) % N for i in range(N)]
        evaluate(a, b)
    
    # Strategy 6: Exhaustive search over small parameter space
    # 'a' is always a cycle, 'b' is determined by a few parameters
    for p1 in range(N):
        for p2 in range(N):
            if time.time() > deadline:
                return [best_a, best_b] if best_a else [list(range(1, N)) + [0], [0] + list(range(N - 1))]
            a = list(range(1, N)) + [0]
            b = [0] * N
            for i in range(N):
                if i == 0:
                    b[i] = p1
                elif i == 1:
                    b[i] = p2
                else:
                    b[i] = i - 1
            evaluate(a, b)
    
    # Strategy 7: Random restarts with structured perturbations
    rng = random.Random(999)
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        # Start from best or random
        if best_a and rng.random() < 0.7:
            a = list(best_a)
            b = list(best_b)
        else:
            a = list(range(1, N)) + [0]
            b = [0] + list(range(N - 1))
        
        # Apply 1-3 random perturbations
        n_perturb = rng.randint(1, 3)
        for _ in range(n_perturb):
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                b[idx] = rng.randrange(N)
        
        evaluate(a, b)
    
    if best_a is None:
        # Fallback
        best_a = list(range(1, N)) + [0]
        best_b = [0] + list(range(N - 1))
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
