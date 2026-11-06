"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _compute_threshold(trans0, trans1, deadline=None):
    """BFS on subset automaton to find exact reset threshold."""
    full = frozenset(range(N))
    # BFS from full set
    visited = {full: 0}
    queue = deque([(full, "")])
    while queue:
        subset, word = queue.popleft()
        if len(subset) == 1:
            return len(word), word
        if deadline is not None and time.time() > deadline:
            return None, None
        # Try appending 0
        new_set = frozenset(trans0[s] for s in subset)
        if new_set not in visited:
            visited[new_set] = len(word) + 1
            queue.append((new_set, word + "0"))
        # Try appending 1
        new_set = frozenset(trans1[s] for s in subset)
        if new_set not in visited:
            visited[new_set] = len(word) + 1
            queue.append((new_set, word + "1"))
    return None, None

def _is_synchronizing(trans0, trans1, deadline=None):
    """Check if automaton synchronizes using BFS."""
    full = frozenset(range(N))
    visited = {full}
    queue = deque([full])
    while queue:
        subset = queue.popleft()
        if len(subset) == 1:
            return True
        if deadline is not None and time.time() > deadline:
            return None  # unknown
        new_set = frozenset(trans0[s] for s in subset)
        if new_set not in visited:
            visited.add(new_set)
            queue.append(new_set)
        new_set = frozenset(trans1[s] for s in subset)
        if new_set not in visited:
            visited.add(new_set)
            queue.append(new_set)
    return False

def _random_automaton():
    """Generate a random complete deterministic automaton."""
    trans0 = [random.randint(0, N - 1) for _ in range(N)]
    trans1 = [random.randint(0, N - 1) for _ in range(N)]
    return trans0, trans1

def _mutate_automaton(trans0, trans1, num_mutations=1):
    """Randomly mutate some transitions."""
    trans0 = list(trans0)
    trans1 = list(trans1)
    for _ in range(num_mutations):
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            trans0[idx] = random.randint(0, N - 1)
        else:
            idx = random.randint(0, N - 1)
            trans1[idx] = random.randint(0, N - 1)
    return trans0, trans1

def _generate_ping_pong_structures():
    """Generate structured automata with ping-pong like properties."""
    structures = []
    
    # Classic Černý: cycle + decrement with fixed point
    t0 = [(i + 1) % N for i in range(N)]
    t1 = [0] * N
    for i in range(1, N):
        t1[i] = i - 1
    structures.append((t0, t1))
    
    # Variant: cycle + decrement with different fixed point
    for fixed in range(N):
        t1 = [0] * N
        for i in range(N):
            if i == fixed:
                t1[i] = fixed
            else:
                t1[i] = (i - 1) % N
        structures.append((t0, t1))
    
    # Reverse cycle + increment
    t0_rev = [(i - 1) % N for i in range(N)]
    t1_inc = [N - 1] * N
    for i in range(N - 1):
        t1_inc[i] = i + 1
    structures.append((t0_rev, t1_inc))
    
    # Various cycle lengths with shift
    for cycle_len in range(3, N + 1):
        t0 = [0] * N
        # First cycle_len states form a cycle
        for i in range(cycle_len):
            t0[i] = (i + 1) % cycle_len
        # Remaining states map into the cycle
        for i in range(cycle_len, N):
            t0[i] = i % cycle_len
        
        # Letter 1: shift down with fixed point at 0
        t1 = [0] * N
        for i in range(1, N):
            t1[i] = i - 1
        structures.append((t0, t1))
    
    return structures

def construct():
    """
    Constructs a synchronizing automaton with 12 states and 2 letters.
    Uses structured generation and random search to maximize reset threshold.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    best_threshold = 0
    best_trans = None
    
    # Phase 1: Evaluate structured constructions
    structures = _generate_ping_pong_structures()
    for t0, t1 in structures:
        if time.time() > deadline:
            break
        thresh, _ = _compute_threshold(t0, t1, deadline)
        if thresh is not None and thresh > best_threshold:
            best_threshold = thresh
            best_trans = (list(t0), list(t1))
    
    # Phase 2: Random search with local refinement
    # Start from best known or random
    if best_trans is None:
        best_trans = _random_automaton()
    
    current_t0, current_t1 = best_trans
    current_thresh = best_threshold
    
    # Local search: mutate and evaluate
    attempts = 0
    max_attempts = 5000
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate candidate via mutation
        mut_strength = 1 if attempts % 3 != 0 else 2
        cand_t0, cand_t1 = _mutate_automaton(current_t0, current_t1, mut_strength)
        
        # Quick check: is it synchronizing?
        synch = _is_synchronizing(cand_t0, cand_t1, deadline)
        if synch is None:
            continue
        if not synch:
            continue
        
        # Compute exact threshold
        thresh, _ = _compute_threshold(cand_t0, cand_t1, deadline)
        if thresh is None:
            continue
        
        if thresh > current_thresh:
            current_thresh = thresh
            current_t0, current_t1 = cand_t0, cand_t1
            best_threshold = current_thresh
            best_trans = (list(current_t0), list(current_t1))
    
    # Phase 3: More aggressive random restarts if time remains
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline:
                break
            rt0, rt1 = _random_automaton()
            synch = _is_synchronizing(rt0, rt1, deadline)
            if synch is None or not synch:
                continue
            thresh, _ = _compute_threshold(rt0, rt1, deadline)
            if thresh is not None and thresh > best_threshold:
                best_threshold = thresh
                best_trans = (list(rt0), list(rt1))
    
    # Return best found
    if best_trans is None:
        # Fallback: classic Černý
        t0 = [(i + 1) % N for i in range(N)]
        t1 = [0] * N
        for i in range(1, N):
            t1[i] = i - 1
        return [t0, t1]
    
    return [best_trans[0], best_trans[1]]
# EVOLVE-BLOCK-END
