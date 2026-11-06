"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _bfs_reset_length(trans_a, trans_b):
    """Compute exact shortest reset word length via BFS on subset automaton."""
    full = frozenset(range(N))
    if len(full) == 1:
        return 0
    # BFS: state is a frozenset of states
    visited = {full: 0}
    queue = deque([(full, 0)])
    while queue:
        subset, dist = queue.popleft()
        # Apply letter a
        next_a = frozenset(trans_a[s] for s in subset)
        if len(next_a) == 1:
            return dist + 1
        if next_a not in visited:
            visited[next_a] = dist + 1
            queue.append((next_a, dist + 1))
        # Apply letter b
        next_b = frozenset(trans_b[s] for s in subset)
        if len(next_b) == 1:
            return dist + 1
        if next_b not in visited:
            visited[next_b] = dist + 1
            queue.append((next_b, dist + 1))
    return -1  # not synchronizing

def _pinwheel():
    """Classic pinwheel: a is cycle, b collapses toward 0."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i for i in range(1, N)]
    # b: 0->0, 1->1, 2->2, ... no that's identity.
    # Let me use: b maps i -> (i-1) mod N for i>0, b(0)=0
    b = [0] + [(i - 1) % N for i in range(1, N)]
    # So b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    return a, b

def _pinwheel_variant(shift):
    """Pinwheel with a shift parameter."""
    a = [(i + 1) % N for i in range(N)]
    # b: map i to (i - shift) mod N, but fix state 0
    b = [0] * N
    for i in range(1, N):
        b[i] = (i - shift) % N
        if b[i] == 0 and i != 0:
            b[i] = 0  # ensure 0 is fixed
    return a, b

def _two_cycle_path():
    """2-cycle plus path structure."""
    # States 0,1 form a 2-cycle under a; path 2->3->...->N-1
    a = [1, 0] + list(range(2, N))
    # b: collapse everything to 0
    b = [0] * N
    return a, b

def _rank2_construction():
    """Rank-2 style construction."""
    a = list(range(N))  # identity
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    # Make a a permutation that's not identity
    a = [(i + 1) % N for i in range(N)]
    return a, b

def _generate_random_candidate(rng):
    """Generate a random complete binary automaton."""
    a = [rng.randint(0, N - 1) for _ in range(N)]
    b = [rng.randint(0, N - 1) for _ in range(N)]
    return a, b

def _local_search(a, b, rng, max_iters=5000, deadline=None):
    """Local search: swap random transitions to try to increase reset length."""
    best_a = list(a)
    best_b = list(b)
    best_len = _bfs_reset_length(best_a, best_b)
    if best_len < 0:
        return best_a, best_b, 0
    
    current_a = list(a)
    current_b = list(b)
    current_len = best_len
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Pick a random transition to modify
        letter = rng.choice(['a', 'b'])
        state = rng.randint(0, N - 1)
        new_dest = rng.randint(0, N - 1)
        
        if letter == 'a':
            old = current_a[state]
            current_a[state] = new_dest
            new_len = _bfs_reset_length(current_a, current_b)
            if new_len < 0:
                current_a[state] = old
            else:
                if new_len > best_len:
                    best_a = list(current_a)
                    best_b = list(current_b)
                    best_len = new_len
                elif new_len < current_len:
                    current_a[state] = old
        else:
            old = current_b[state]
            current_b[state] = new_dest
            new_len = _bfs_reset_length(current_a, current_b)
            if new_len < 0:
                current_b[state] = old
            else:
                if new_len > best_len:
                    best_a = list(current_a)
                    best_b = list(current_b)
                    best_len = new_len
                elif new_len < current_len:
                    current_b[state] = old
        current_len = _bfs_reset_length(current_a, current_b) if new_len >= 0 else current_len
    
    return best_a, best_b, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s
    
    rng = random.Random(42)
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Try structural families
    candidates = []
    candidates.append(_pinwheel())
    for shift in range(1, N):
        candidates.append(_pinwheel_variant(shift))
    candidates.append(_two_cycle_path())
    candidates.append(_rank2_construction())
    
    # Add more structured candidates
    # Cycle with a "defect" at position k
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[k] = 0  # defect
        candidates.append((a, b))
    
    # Evaluate structural candidates
    for a, b in candidates:
        if time.time() > deadline:
            break
        length = _bfs_reset_length(a, b)
        if length > best_len:
            best_len = length
            best_a = list(a)
            best_b = list(b)
    
    # Random search with local refinement
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            a, b = _generate_random_candidate(rng)
            length = _bfs_reset_length(a, b)
            if length > best_len:
                best_len = length
                best_a = list(a)
                best_b = list(b)
            elif length > 0:
                # Try local search from this candidate
                a2, b2, len2 = _local_search(a, b, rng, max_iters=2000, deadline=deadline)
                if len2 > best_len:
                    best_len = len2
                    best_a = list(a2)
                    best_b = list(b2)
    
    # If we have a good candidate, try to improve it with more local search
    if best_a is not None and time.time() < deadline:
        a2, b2, len2 = _local_search(best_a, best_b, rng, max_iters=10000, deadline=deadline)
        if len2 > best_len:
            best_len = len2
            best_a = list(a2)
            best_b = list(b2)
    
    # Final safety: if no valid candidate found, return a known synchronizing one
    if best_a is None or best_len <= 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        best_a = a
        best_b = b
        best_len = _bfs_reset_length(a, b)
    
    return best_a, best_b

# EVOLVE-BLOCK-END
