"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def _bfs_reset_length_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses integer bitmasks for subsets to improve performance.
    """
    N = len(a_trans)
    start_mask = (1 << N) - 1  # All states present
    
    # Quick check: if a or b is constant, it's not a cycle, but we need to check if it resets
    # Actually, if a is constant, applying 'a' once makes the set size 1.
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
        
    visited = {start_mask}
    # Queue stores (mask, distance)
    queue = deque([(start_mask, 0)])
    
    # Precompute destination masks for each state for a and b?
    # dest_mask_a[i] = 1 << a_trans[i]
    dest_a = [1 << a_trans[i] for i in range(N)]
    dest_b = [1 << b_trans[i] for i in range(N)]
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Generate next states
        # Apply 'a'
        next_mask_a = 0
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_a |= dest_a[idx]
            m ^= lsb
            
        if next_mask_a not in visited:
            visited.add(next_mask_a)
            queue.append((next_mask_a, dist + 1))
            
        # Apply 'b'
        next_mask_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = (lsb.bit_length() - 1)
            next_mask_b |= dest_b[idx]
            m ^= lsb
            
        if next_mask_b not in visited:
            visited.add(next_mask_b)
            queue.append((next_mask_b, dist + 1))
            
    return -1  # Not synchronizing

def _build_subset_transition_table(a_trans, b_trans):
    """
    Precompute subset transitions for all 2^N subsets.
    Returns two lists: trans_a[mask] and trans_b[mask] for mask in 0..2^N-1.
    """
    N = len(a_trans)
    num_masks = 1 << N
    
    # Precompute single-state transitions
    dest_a = [1 << a_trans[i] for i in range(N)]
    dest_b = [1 << b_trans[i] for i in range(N)]
    
    trans_a = [0] * num_masks
    trans_b = [0] * num_masks
    
    # Compute transitions for all masks using DP
    # For a mask, trans[mask] = OR of dests for each state in mask
    # We can compute this incrementally
    for mask in range(1, num_masks):
        # Find the least significant set bit
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        if prev_mask == 0:
            trans_a[mask] = dest_a[idx]
            trans_b[mask] = dest_b[idx]
        else:
            trans_a[mask] = trans_a[prev_mask] | dest_a[idx]
            trans_b[mask] = trans_b[prev_mask] | dest_b[idx]
    
    return trans_a, trans_b

def _bfs_with_table(trans_a, trans_b, N):
    """
    BFS on subset automaton using precomputed transition table.
    """
    start_mask = (1 << N) - 1
    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return dist
        
        next_a = trans_a[mask]
        next_b = trans_b[mask]
        
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return False
        score = _bfs_reset_length_fast(a_trans, b_trans)
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]
            return True
        return False

    # Strategy 1: Pinwheel constructions with different parameters
    for step in range(1, N):
        a_trans = [(i + step) % N for i in range(N)]
        for target in range(N):
            b_trans = [0 if i == 0 else i - 1 for i in range(N)]
            b_trans[0] = target
            try_automaton(a_trans, b_trans)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: Systematic enumeration
    a_trans_base = [(i + 1) % N for i in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b_trans = [(k + 1) % N for k in range(N)]
            b_trans[i] = j
            try_automaton(a_trans_base, b_trans)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 3: More diverse structures
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        for special_state in range(N):
            for special_target in range(N):
                b_trans = [(i + 1) % N for i in range(N)]
                b_trans[special_state] = special_target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Strategy 4: Simulated Annealing with precomputed transition table
    if time.time() < deadline - 5.0:
        random.seed(42)
        
        # Initialize from best known or default
        if best_a is None:
            current_a = [(i + 1) % N for i in range(N)]
            current_b = [0 if i == 0 else i - 1 for i in range(N)]
        else:
            current_a = best_a[:]
            current_b = best_b[:]
        
        current_score = best_score
        best_sa_score = best_score
        best_sa_a = best_a[:] if best_a else current_a[:]
        best_sa_b = best_b[:] if best_b else current_b[:]
        
        # Build initial transition table
        trans_a, trans_b = _build_subset_transition_table(current_a, current_b)
        current_score = _bfs_with_table(trans_a, trans_b, N)
        
        # Simulated annealing parameters
        T_start = 10.0
        T_end = 0.01
        cooling_rate = 0.999
        max_iters = 50000
        
        T = T_start
        iter_count = 0
        
        while time.time() < deadline - 2.0 and T > T_end and iter_count < max_iters:
            # Generate a mutation
            # Choose which transition to modify: a or b, and which state
            if random.random() < 0.5:
                # Modify a_trans
                i = random.randint(0, N - 1)
                old_dest = current_a[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_a[i] = new_dest
                
                # Rebuild transition table (expensive but necessary)
                trans_a, trans_b = _build_subset_transition_table(current_a, current_b)
                new_score = _bfs_with_table(trans_a, trans_b, N)
                
                # Accept or reject
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_sa_score:
                        best_sa_score = new_score
                        best_sa_a = current_a[:]
                        best_sa_b = current_b[:]
                else:
                    # Accept with probability exp((new_score - current_score) / T)
                    # Since scores are integers, delta is at most -1
                    delta = new_score - current_score
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        current_a[i] = old_dest
                        trans_a, trans_b = _build_subset_transition_table(current_a, current_b)
                        current_score = _bfs_with_table(trans_a, trans_b, N)
            else:
                # Modify b_trans
                i = random.randint(0, N - 1)
                old_dest = current_b[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old_dest:
                    continue
                current_b[i] = new_dest
                
                trans_a, trans_b = _build_subset_transition_table(current_a, current_b)
                new_score = _bfs_with_table(trans_a, trans_b, N)
                
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_sa_score:
                        best_sa_score = new_score
                        best_sa_a = current_a[:]
                        best_sa_b = current_b[:]
                else:
                    delta = new_score - current_score
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_b[i] = old_dest
                        trans_a, trans_b = _build_subset_transition_table(current_a, current_b)
                        current_score = _bfs_with_table(trans_a, trans_b, N)
            
            # Cool down
            T *= cooling_rate
            iter_count += 1
        
        # Update global best if SA found something better
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_a = best_sa_a
            best_b = best_sa_b
    
    # Fallback
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
