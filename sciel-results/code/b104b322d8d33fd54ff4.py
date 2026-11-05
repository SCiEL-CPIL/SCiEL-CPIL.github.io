"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _calculate_threshold(a_trans, b_trans, max_steps=100000):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton. Returns None if not synchronizing within max_steps.
    """
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    steps = 0
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        steps += 1
        
        if steps > max_steps:
            return None
        
        if len(curr) == 1:
            return curr_dist
        
        # Apply letter 'a'
        next_a = frozenset(a_trans[i] for i in curr)
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = frozenset(b_trans[i] for i in curr)
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
    
    return None

def _is_cyclic_perm(a_trans):
    """Check if a_trans is a cyclic permutation (single cycle of length N)."""
    visited = [False] * N
    for i in range(N):
        if not visited[i]:
            # Start a new cycle
            curr = i
            count = 0
            while not visited[curr]:
                visited[curr] = True
                curr = a_trans[curr]
                count += 1
            if count != N:
                return False
    return True

def construct():
    """
    Construct a 12-state binary deterministic automaton that synchronizes,
    maximizing the length of its shortest reset word.
    
    Strategy: Search over automata where 'a' is a cyclic permutation and 'b' 
    is a carefully chosen non-permutation map. This is based on Pinsker-like
    constructions which are known to achieve linear reset thresholds.
    """
    deadline = time.time() + 75  # Safety margin before 90s limit
    best_threshold = 0
    best_a = list(range(1, N)) + [0]  # Default cyclic permutation
    best_b = [0] * N
    
    # Precompute all cyclic permutations? No, there are (N-1)! which is too many.
    # Instead, fix 'a' to be a specific cyclic permutation and vary 'b'.
    # The specific cycle structure matters less than 'b' breaking the symmetry.
    
    # Use a standard cyclic permutation for 'a': i -> (i+1) % N
    a_trans = [(i + 1) % N for i in range(N)]
    
    # We will search over possible 'b' maps.
    # A good heuristic is to make 'b' map most states to a small set of states,
    # but not all to the same one (to avoid trivial synchronization).
    # Specifically, Pinsker-type automata often have 'b' as a "reset" map with some exceptions.
    
    # Let's try a structured search first, then random hill-climbing.
    
    # Phase 1: Structured search over "almost reset" maps for 'b'
    # b maps all states to 0, except one state k maps to some other state j.
    # Or b maps most to 0, but a few are different.
    
    # Actually, a known family for high threshold is:
    # a: i -> i+1 mod N
    # b: i -> 0 for i != k, b[k] = 1 (or similar)
    # Let's test variations of this.
    
    for k in range(N):
        for j in range(N):
            if k == j:
                continue
            b_trans = [0] * N
            b_trans[k] = j
            thresh = _calculate_threshold(a_trans, b_trans)
            if thresh is not None and thresh > best_threshold:
                best_threshold = thresh
                best_a = a_trans[:]
                best_b = b_trans[:]
                if time.time() > deadline:
                    return [best_a, best_b]
    
    # Phase 2: More complex 'b' maps
    # b maps all to 0, except two states k1, k2 map to j1, j2 respectively.
    for k1 in range(N):
        for j1 in range(N):
            if k1 == j1:
                continue
            for k2 in range(N):
                if k2 == k1:
                    continue
                for j2 in range(N):
                    if k2 == j2:
                        continue
                    if k1 > k2:  # Avoid duplicates
                        continue
                    b_trans = [0] * N
                    b_trans[k1] = j1
                    b_trans[k2] = j2
                    thresh = _calculate_threshold(a_trans, b_trans)
                    if thresh is not None and thresh > best_threshold:
                        best_threshold = thresh
                        best_a = a_trans[:]
                        best_b = b_trans[:]
                        if time.time() > deadline:
                            return [best_a, best_b]
    
    # Phase 3: Random hill-climbing
    # Start from the best found so far, or a random valid 'b'
    if best_threshold > 0:
        current_b = best_b[:]
        current_threshold = best_threshold
    else:
        current_b = [0] * N
        current_b[0] = 1
        current_threshold = _calculate_threshold(a_trans, current_b) or 0
    
    # Random perturbation of 'b'
    while time.time() < deadline:
        # Pick a random state and change its destination
        idx = random.randint(0, N - 1)
        old_val = current_b[idx]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            continue
        current_b[idx] = new_val
        
        thresh = _calculate_threshold(a_trans, current_b)
        if thresh is not None and thresh > current_threshold:
            current_threshold = thresh
            if thresh > best_threshold:
                best_threshold = thresh
                best_b = current_b[:]
        else:
            # Revert if it didn't improve (or became non-synchronizing)
            current_b[idx] = old_val
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
