"""
Constructs a 12-state binary deterministic automaton with a long shortest reset word.
Uses bitmask-based BFS for fast threshold evaluation and random restart local search.
"""

import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def reset_threshold(a, b):
    """
    Compute the exact shortest reset word length using BFS on subset automaton.
    Returns -1 if not synchronizing, 0 if already a singleton, else the length.
    Uses bitmask representation for speed.
    """
    # Precompute transition of any subset under a and b
    # trans_a[mask] = image of subset 'mask' under letter 'a'
    # We can compute this on the fly or precompute. For N=12, 4096 subsets.
    
    # Precompute images for all subsets under a and b
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << a[i])
                mb |= (1 << b[i])
        trans_a[mask] = ma
        trans_b[mask] = mb
    
    # BFS from FULL_MASK, looking for any singleton
    # Distance 0: FULL_MASK
    # We want shortest word that maps FULL_MASK to a singleton
    
    # Use BFS: state is the subset (mask), we want to reach a singleton
    # Start: FULL_MASK
    # Goal: any mask with popcount == 1
    
    # visited: set of masks we've seen
    # We can use a boolean array of size 2^N
    visited = [False] * (1 << N)
    visited[FULL_MASK] = True
    
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    # Singletons are masks with exactly one bit set
    # We can check: mask & (mask - 1) == 0 and mask != 0
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if this is a singleton (goal)
        if mask != 0 and (mask & (mask - 1)) == 0:
            return dist
        
        # Expand: apply 'a' and 'b'
        for t in (trans_a, trans_b):
            new_mask = t[mask]
            if not visited[new_mask]:
                visited[new_mask] = True
                queue.append((new_mask, dist + 1))
    
    return -1  # Not synchronizing


def is_synchronizing(a, b):
    """Quick check if automaton synchronizes."""
    return reset_threshold(a, b) >= 0


# EVOLVE-BLOCK-START
def construct():
    """
    Find a 12-state binary automaton with a long shortest reset word.
    Uses random restart local search with fast BFS-based threshold evaluation.
    Returns two lists of length N: transition functions for letters 'a' and 'b'.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_a = None
    best_b = None
    best_threshold = -1
    
    # Strategy: random restart local search
    # 1. Generate random automaton
    # 2. Check if it synchronizes
    # 3. If yes, try to improve by mutating transitions
    # 4. Keep best found
    
    rng = random.Random(42)  # Deterministic seed for reproducibility
    
    def random_automaton():
        a = [rng.randint(0, N-1) for _ in range(N)]
        b = [rng.randint(0, N-1) for _ in range(N)]
        return a, b
    
    def mutate(a, b, num_mutations=1):
        """Randomly mutate some transitions."""
        a_new = a[:]
        b_new = b[:]
        for _ in range(num_mutations):
            if rng.random() < 0.5:
                i = rng.randint(0, N-1)
                a_new[i] = rng.randint(0, N-1)
            else:
                i = rng.randint(0, N-1)
                b_new[i] = rng.randint(0, N-1)
        return a_new, b_new
    
    def local_search(a, b, max_iterations=200):
        """
        Local search: try mutations, keep if threshold improves or stays same.
        Returns best (a, b, threshold) found.
        """
        current_a = a[:]
        current_b = b[:]
        current_thresh = reset_threshold(current_a, current_b)
        
        if current_thresh < 0:
            return current_a, current_b, -1
        
        best_a_local = current_a[:]
        best_b_local = current_b[:]
        best_thresh_local = current_thresh
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            a_new, b_new = mutate(current_a, current_b, num_mutations=1)
            thresh_new = reset_threshold(a_new, b_new)
            
            if thresh_new > best_thresh_local:
                best_a_local = a_new[:]
                best_b_local = b_new[:]
                best_thresh_local = thresh_new
                current_a = a_new[:]
                current_b = b_new[:]
                current_thresh = thresh_new
            elif thresh_new >= 0 and thresh_new == current_thresh:
                # Accept sideways moves with some probability to escape local optima
                if rng.random() < 0.3:
                    current_a = a_new[:]
                    current_b = b_new[:]
                    current_thresh = thresh_new
        
        return best_a_local, best_b_local, best_thresh_local
    
    # Phase 1: Try some structured initial candidates
    # The Pin construction: a is a cycle, b maps 0->0, i->i+1 for i>0
    a_pin = [(i + 1) % N for i in range(N)]
    b_pin = [0] * N
    for i in range(1, N):
        b_pin[i] = (i + 1) % N
    thresh_pin = reset_threshold(a_pin, b_pin)
    if thresh_pin > best_threshold:
        best_a = a_pin[:]
        best_b = b_pin[:]
        best_threshold = thresh_pin
    
    # Another structured candidate: a is cycle, b maps all to 0
    a_cycle = [(i + 1) % N for i in range(N)]
    b_const = [0] * N
    thresh_const = reset_threshold(a_cycle, b_const)
    if thresh_const > best_threshold:
        best_a = a_cycle[:]
        best_b = b_const[:]
        best_threshold = thresh_const
    
    # Phase 2: Random restart local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Generate random automaton
        a_rand, b_rand = random_automaton()
        thresh_rand = reset_threshold(a_rand, b_rand)
        
        if thresh_rand < 0:
            continue
        
        if thresh_rand > best_threshold:
            best_a = a_rand[:]
            best_b = b_rand[:]
            best_threshold = thresh_rand
        
        # Do local search from this starting point
        a_ls, b_ls, thresh_ls = local_search(a_rand, b_rand, max_iterations=100)
        
        if thresh_ls > best_threshold:
            best_a = a_ls[:]
            best_b = b_ls[:]
            best_threshold = thresh_ls
    
    # Ensure we return a valid synchronizing automaton
    if best_a is None or best_b is None or best_threshold < 0:
        # Fallback: use the Pin construction which is guaranteed to synchronize
        a_pin = [(i + 1) % N for i in range(N)]
        b_pin = [0] * N
        for i in range(1, N):
            b_pin[i] = (i + 1) % N
        return [a_pin, b_pin]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
