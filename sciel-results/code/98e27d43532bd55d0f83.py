"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _bfs_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns (threshold, word) where word is a string of 'a'/'b'.
    If not synchronizing, returns (None, None).
    """
    # Precompute transition for all subsets for both letters
    # Represent subsets as bitmasks
    # trans_a_mask[s] = bitmask of delta_a(s)
    # We'll compute on-the-fly but cache
    
    # Actually, let's precompute for all 2^N subsets
    # But 2^12 = 4096, so we can precompute
    # delta(mask, letter) = OR of (1 << trans[state]) for each state in mask
    
    # Precompute single-state transitions
    # For each state i, trans_a[i] and trans_b[i]
    # For a subset mask, delta_a(mask) = OR over i in mask of (1 << trans_a[i])
    
    # We can precompute delta for all masks using DP
    # delta(mask, a) = delta(mask without lowest bit, a) | (1 << trans_a[lowest_bit])
    
    # Let's precompute for both letters
    delta_a = [0] * (1 << N)
    delta_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        # Find lowest set bit
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        delta_a[mask] = delta_a[prev] | (1 << trans_a[idx])
        delta_b[mask] = delta_b[prev] | (1 << trans_b[idx])
    
    # BFS from FULL_MASK
    # visited: set of masks
    # parent: dict from mask to (parent_mask, letter)
    visited = {FULL_MASK}
    parent = {}  # mask -> (parent_mask, letter)
    queue = deque([FULL_MASK])
    
    while queue:
        mask = queue.popleft()
        if mask != FULL_MASK:
            # Reconstruct word
            word = []
            m = mask
            while m != FULL_MASK:
                pm, letter = parent[m]
                word.append(letter)
                m = pm
            word.reverse()
            return len(word), ''.join(word)
        
        for letter, delta in [('a', delta_a), ('b', delta_b)]:
            next_mask = delta[mask]
            if next_mask not in visited:
                visited.add(next_mask)
                parent[next_mask] = (mask, letter)
                queue.append(next_mask)
    
    return None, None

def construct():
    """
    Construct a 12-state synchronizing automaton maximizing the reset threshold.
    Uses random restart local search with bitmask BFS for fast evaluation.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_threshold = -1
    best_a = None
    best_b = None
    
    # Initialize with the known good construction: cycle + sink
    # a: cycle (i+1)%N, b: all to 0
    # But the insight says cycle-shift variant: b maps N-1 to 1 instead of 0
    # Let's try that first
    init_a = [(i + 1) % N for i in range(N)]
    init_b = [0] * N
    init_b[N-1] = 1  # cycle-shift variant
    
    thresh, _ = _bfs_threshold(init_a, init_b)
    if thresh is not None and thresh > best_threshold:
        best_threshold = thresh
        best_a = list(init_a)
        best_b = list(init_b)
    
    # Also try the simple one
    simple_b = [0] * N
    thresh, _ = _bfs_threshold(init_a, simple_b)
    if thresh is not None and thresh > best_threshold:
        best_threshold = thresh
        best_a = list(init_a)
        best_b = list(simple_b)
    
    # Random restart local search
    random.seed(42)
    
    def random_automaton():
        """Generate a random complete deterministic automaton."""
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        return a, b
    
    def local_search(a, b, max_iterations=1000):
        """Local search: mutate one transition at a time, accept if threshold improves or stays same."""
        current_a = list(a)
        current_b = list(b)
        current_thresh, _ = _bfs_threshold(current_a, current_b)
        if current_thresh is None:
            return None, None, -1
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Choose a random mutation: pick a random letter, random state, random new target
            letter = random.choice(['a', 'b'])
            state = random.randint(0, N-1)
            old_target = (current_a if letter == 'a' else current_b)[state]
            new_target = random.randint(0, N-1)
            if new_target == old_target:
                continue
            
            if letter == 'a':
                current_a[state] = new_target
                new_thresh, _ = _bfs_threshold(current_a, current_b)
                if new_thresh is None or new_thresh < current_thresh:
                    current_a[state] = old_target
                else:
                    current_thresh = new_thresh
            else:
                current_b[state] = new_target
                new_thresh, _ = _bfs_threshold(current_a, current_b)
                if new_thresh is None or new_thresh < current_thresh:
                    current_b[state] = old_target
                else:
                    current_thresh = new_thresh
        
        return current_a, current_b, current_thresh
    
    # Do random restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Generate a random starting point
        a, b = random_automaton()
        
        # Run local search
        new_a, new_b, new_thresh = local_search(a, b, max_iterations=500)
        
        if new_thresh is not None and new_thresh > best_threshold:
            best_threshold = new_thresh
            best_a = new_a
            best_b = new_b
        
        # Also try local search from the best found so far with more iterations
        if restart_count % 5 == 0 and time.time() < deadline:
            new_a, new_b, new_thresh = local_search(best_a, best_b, max_iterations=200)
            if new_thresh is not None and new_thresh > best_threshold:
                best_threshold = new_thresh
                best_a = new_a
                best_b = new_b
    
    # Final check
    if best_a is None:
        # Fallback to simple construction
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_a, best_b

# EVOLVE-BLOCK-END
