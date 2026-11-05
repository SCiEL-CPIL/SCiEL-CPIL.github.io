"""
Constructs a 12-state binary deterministic automaton that synchronizes.
Uses bitmask-based BFS for fast threshold computation and random restart
local search to maximize the shortest reset word length.
"""

import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1


def reset_threshold(a, b):
    """
    Compute the exact shortest reset word length for a 12-state binary automaton.
    Uses BFS on the subset automaton represented as bitmasks.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    # Precompute transition of each state under 'a' and 'b'
    # trans_a[s] = destination of state s under letter 'a'
    # trans_b[s] = destination of state s under letter 'b'
    
    # For a subset (bitmask), applying letter 'a' means:
    # new_mask = OR of (1 << a[s]) for each s in mask
    # We can precompute the image of each single state, then build subset images.
    
    # Actually, for efficiency, let's precompute the image of each possible subset.
    # There are 2^12 = 4096 subsets. We can precompute img_a[mask] and img_b[mask].
    
    img_a = [0] * (1 << N)
    img_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        for s in range(N):
            if mask & (1 << s):
                ma |= (1 << a[s])
                mb |= (1 << b[s])
        img_a[mask] = ma
        img_b[mask] = mb
    
    # BFS from FULL_MASK
    # State: subset bitmask
    # Goal: reach any singleton subset (exactly one bit set)
    # A singleton mask m satisfies m & (m - 1) == 0 and m != 0
    
    visited = [False] * (1 << N)
    visited[FULL_MASK] = True
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if this is a singleton
        if mask != 0 and (mask & (mask - 1)) == 0:
            return dist
        
        # Try applying 'a'
        new_mask_a = img_a[mask]
        if not visited[new_mask_a]:
            visited[new_mask_a] = True
            queue.append((new_mask_a, dist + 1))
        
        # Try applying 'b'
        new_mask_b = img_b[mask]
        if not visited[new_mask_b]:
            visited[new_mask_b] = True
            queue.append((new_mask_b, dist + 1))
    
    return -1  # Not synchronizing


def is_singleton(mask):
    return mask != 0 and (mask & (mask - 1)) == 0


def random_automaton():
    """Generate a random 12-state binary automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b


def mutate(a, b):
    """
    Mutate an automaton by changing a few random transitions.
    Returns new (a, b) lists.
    """
    new_a = a[:]
    new_b = b[:]
    
    # Change 1-3 random transitions
    num_changes = random.randint(1, 3)
    for _ in range(num_changes):
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            new_a[idx] = random.randint(0, N - 1)
        else:
            idx = random.randint(0, N - 1)
            new_b[idx] = random.randint(0, N - 1)
    
    return new_a, new_b


def local_search(start_a, start_b, deadline):
    """
    Perform local search starting from (start_a, start_b).
    Returns the best (a, b, threshold) found before deadline.
    """
    best_a = start_a[:]
    best_b = start_b[:]
    best_thresh = reset_threshold(best_a, best_b)
    
    if best_thresh < 0:
        return best_a, best_b, best_thresh
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_thresh = best_thresh
    
    while time.time() < deadline:
        # Mutate
        new_a, new_b = mutate(current_a, current_b)
        new_thresh = reset_threshold(new_a, new_b)
        
        if new_thresh < 0:
            continue
        
        # Accept if better or equal (with some probability for equal to explore)
        if new_thresh > current_thresh or (new_thresh == current_thresh and random.random() < 0.3):
            current_a = new_a
            current_b = new_b
            current_thresh = new_thresh
            
            if current_thresh > best_thresh:
                best_a = current_a[:]
                best_b = current_b[:]
                best_thresh = current_thresh
    
    return best_a, best_b, best_thresh


# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 12-state binary automaton with maximum shortest reset word length.
    Uses random restart local search with bitmask-based BFS for threshold computation.
    """
    start_time = time.time()
    # Leave 10 seconds margin before the 90-second kill
    deadline = start_time + 80.0
    
    # Start with a known good synchronizing automaton
    # The Černý construction: a is a cycle, b maps 0 to 0 and i to (i+1)%N for i>0
    a0 = [(i + 1) % N for i in range(N)]
    b0 = [0] * N
    for i in range(1, N):
        b0[i] = (i + 1) % N
    
    best_a = a0[:]
    best_b = b0[:]
    best_thresh = reset_threshold(best_a, best_b)
    
    # Random restart local search
    num_restarts = 0
    while time.time() < deadline:
        # Generate a random starting point
        ra, rb = random_automaton()
        
        # Only proceed if it synchronizes
        thresh = reset_threshold(ra, rb)
        if thresh < 0:
            continue
        
        # Run local search from this point
        la, lb, lt = local_search(ra, rb, deadline)
        
        if lt > best_thresh:
            best_a = la
            best_b = lb
            best_thresh = lt
        
        num_restarts += 1
    
    return best_a, best_b
# EVOLVE-BLOCK-END
