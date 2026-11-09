"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 12

# EVOLVE-BLOCK-START

def _compute_threshold_bitmask(a_map, b_map, n):
    """
    Compute the exact shortest reset word length using bitmask-based BFS.
    Returns threshold length, or None if not synchronizing.
    """
    size = 1 << n
    full_mask = size - 1
    
    # Precompute transitions for all masks
    # img_a[mask] is the bitmask of states reachable from 'mask' via 'a'
    img_a = [0] * size
    img_b = [0] * size
    
    for mask in range(1, size):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        img_a[mask] = img_a[prev_mask] | (1 << a_map[idx])
        img_b[mask] = img_b[prev_mask] | (1 << b_map[idx])
        
    # BFS
    from collections import deque
    visited = [False] * size
    queue = deque()
    queue.append(full_mask)
    visited[full_mask] = True
    depth = 0
    
    while queue:
        size_q = len(queue)
        for _ in range(size_q):
            mask = queue.popleft()
            if mask & (mask - 1) == 0: # Power of 2, i.e., singleton
                return depth
            
            next_a = img_a[mask]
            next_b = img_b[mask]
            
            if not visited[next_a]:
                visited[next_a] = True
                queue.append(next_a)
            
            if not visited[next_b]:
                visited[next_b] = True
                queue.append(next_b)
                
        depth += 1
        
    return None

def _pawlik_automaton():
    # Standard Černý/Pawlik automaton
    # a: i -> i+1 mod N
    # b: 0 -> 0, i -> i-1 for i>0
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return [a, b]

def _anti_pawlik_automaton():
    """
    Construct an 'anti-Pawlik' automaton by inverting the structural roles.
    Instead of a being a cycle and b being a reset-to-0, we make:
    - a: a permutation that creates long chains (e.g., a single N-cycle but with different orientation)
    - b: a map that expands subsets rather than contracting them, forcing longer reset sequences
    
    The idea: In Pawlik, 'a' shifts everything, 'b' resets to 0. 
    In anti-Pawlik, we try to make both letters 'expand' subsets in a controlled way,
    or make one letter create a structure that the other must carefully dismantle.
    
    One approach: 
    - a: i -> (i + 1) mod N (cycle)
    - b: i -> (i * 2) mod N (or some other non-injective map that doesn't immediately collapse)
    
    But for N=12, let's try:
    - a: i -> (i + 1) % 12
    - b: i -> (i + 7) % 12 if i < 6, else (i - 7) % 12  (creates two cycles of length 6)
    
    Actually, let's try a known good construction:
    - a: i -> (i + 1) % N
    - b: i -> (i + k) % N where gcd(k, N) > 1, so b is not a single cycle
    """
    a = [(i + 1) % N for i in range(N)]
    # Use a step that creates multiple cycles
    # For N=12, step 4 gives cycles: 0->4->8->0, 1->5->9->1, 2->6->10->2, 3->7->11->3
    # This forces 'a' to move between cycles to synchronize
    b = [(i + 4) % N for i in range(N)]
    return [a, b]

def _pawlik_variant_2():
    """
    Another variant: 
    - a: i -> (i + 1) % N
    - b: i -> (i + 2) % N if i % 2 == 0, else (i - 2) % N
    """
    a = [(i + 1) % N for i in range(N)]
    b = []
    for i in range(N):
        if i % 2 == 0:
            b.append((i + 2) % N)
        else:
            b.append((i - 2) % N)
    return [a, b]

def _random_synchronizing(rng):
    """
    Generate a random automaton that is likely to synchronize.
    Strategy: Start with Pawlik and make small random modifications.
    """
    base = _pawlik_automaton()
    trans = [row[:] for row in base]
    
    # Make 1-3 random modifications
    for _ in range(rng.randint(1, 3)):
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        trans[letter][state] = rng.randint(0, N - 1)
    
    return trans

def _local_search(trans, rng, deadline):
    """
    Perform local search to improve the threshold.
    """
    current = [row[:] for row in trans]
    current_thresh = _compute_threshold_bitmask(current[0], current[1], N)
    
    if current_thresh is None:
        # If not synchronizing, try to fix it
        for _ in range(100):
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            current[letter][state] = rng.randint(0, N - 1)
            current_thresh = _compute_threshold_bitmask(current[0], current[1], N)
            if current_thresh is not None:
                break
    
    if current_thresh is None:
        return None, -1
    
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    T = 5.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Propose mutation
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        
        old_dest = current[letter][state]
        new_dest = rng.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
            
        current[letter][state] = new_dest
        
        new_thresh = _compute_threshold_bitmask(current[0], current[1], N)
        
        if new_thresh is None:
            current[letter][state] = old_dest
            continue
            
        if new_thresh >= current_thresh:
            current_thresh = new_thresh
            if new_thresh > best_thresh:
                best_thresh = new_thresh
                best = [row[:] for row in current]
        else:
            delta = new_thresh - current_thresh
            prob = math.exp(delta / T)
            if rng.random() < prob:
                current_thresh = new_thresh
            else:
                current[letter][state] = old_dest
        
        T *= cooling
    
    return best, best_thresh

def construct():
    deadline = time.time() + 50  # Safety margin
    rng = random.Random(42)
    
    # Try multiple different structural families
    candidates = []
    
    # 1. Standard Pawlik
    candidates.append(_pawlik_automaton())
    
    # 2. Anti-Pawlik (inverted structure)
    candidates.append(_anti_pawlik_automaton())
    
    # 3. Pawlik variant 2
    candidates.append(_pawlik_variant_2())
    
    # 4-10. Random synchronizing candidates
    for _ in range(7):
        candidates.append(_random_synchronizing(rng))
    
    # Evaluate all candidates
    best_trans = None
    best_thresh = -1
    
    for trans in candidates:
        thresh = _compute_threshold_bitmask(trans[0], trans[1], N)
        if thresh is not None and thresh > best_thresh:
            best_thresh = thresh
            best_trans = trans
    
    if best_trans is None:
        # Fallback
        best_trans = _pawlik_automaton()
        best_thresh = _compute_threshold_bitmask(best_trans[0], best_trans[1], N)
        if best_thresh is None:
            best_thresh = 0
    
    # Local search from the best candidate
    improved_trans, improved_thresh = _local_search(best_trans, rng, deadline)
    
    if improved_trans is not None and improved_thresh > best_thresh:
        best_trans = improved_trans
        best_thresh = improved_thresh
    
    return best_trans

# EVOLVE-BLOCK-END
