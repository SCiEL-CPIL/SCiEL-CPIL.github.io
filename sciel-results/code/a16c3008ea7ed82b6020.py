"""Construct a 12-state binary deterministic automaton maximizing reset threshold."""
import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # safety margin before 90s kill

def _compute_threshold(trans):
    """Compute exact shortest reset word length via BFS on subset automaton.
    Returns (threshold, word) or (None, None) if not synchronizing."""
    a, b = trans
    # State 0 is the target singleton (we can pick any, but 0 is conventional)
    # Actually, we need to find ANY singleton reachable from the full set.
    # BFS from full set {0..N-1}
    start = tuple(range(N))
    visited = {start: ""}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        word = visited[state]
        if len(state) == 1:
            return len(word), word
        # Try 'a'
        new_state_a = tuple(sorted(set(a[i] for i in state)))
        if new_state_a not in visited:
            visited[new_state_a] = word + "a"
            queue.append(new_state_a)
        # Try 'b'
        new_state_b = tuple(sorted(set(b[i] for i in state)))
        if new_state_b not in visited:
            visited[new_state_b] = word + "b"
            queue.append(new_state_b)
    return None, None

def _is_synchronizing(trans):
    """Quick check if automaton synchronizes (threshold is finite)."""
    a, b = trans
    start = tuple(range(N))
    visited = {start}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if len(state) == 1:
            return True
        new_a = tuple(sorted(set(a[i] for i in state)))
        if new_a not in visited:
            visited.add(new_a)
            queue.append(new_a)
        new_b = tuple(sorted(set(b[i] for i in state)))
        if new_b not in visited:
            visited.add(new_b)
            queue.append(new_b)
    return False

def _pinwheel_init(k):
    """Generate a pinwheel/cycle+comb style automaton.
    'a' is a cycle, 'b' maps most states toward a sink but with some structure."""
    a = list(range(1, N)) + [0]  # cycle 0->1->2->...->11->0
    # b maps each state i to some target to create comb structure
    # Classic high-threshold construction: b maps i to (i - k) % N for some, 
    # but that's just another permutation. We need a non-permutation.
    # Pinwheel: b[i] = (i + k) % N would be permutation. 
    # For synchronizing, b must not be permutation.
    # Standard: b[i] = i for most, but b[0] = 0, b[1] = 0, etc.
    # Let's use the "cycle with a spike" structure:
    # a: cycle
    # b: maps state 0 to 0, and for i>0, b[i] = i-1 (toward 0)
    # This gives threshold N-1 = 11. Not great.
    
    # Better pinwheel: 
    # a: i -> (i+1) % N
    # b: i -> (i + k) % N for i in some range, and i -> 0 for others
    # Actually, the known best for N=12 is around 75+ from the knowledge.
    # Let's try: a is cycle, b maps i to (i - k) % N but with one state mapped to 0.
    
    # Simpler: use the "Černý automaton" variant:
    # a: i -> (i+1) % N
    # b: i -> 0 if i == 0, else i - 1
    # This gives threshold N-1.
    
    # For higher thresholds, use "pinwheel" where b doesn't just decrement:
    # b[i] = (i + k) % N for all i except one state that maps to 0.
    # But that's still a permutation except one state.
    
    # Let me just generate random-ish but structured:
    b = [0] * N
    for i in range(N):
        b[i] = (i + k) % N
    # Make it non-permutation by mapping one state to 0
    b[0] = 0
    return [a, b]

def _random_init():
    """Generate a random complete deterministic automaton."""
    a = [random.randint(0, N-1) for _ in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    return [a, b]

def _hill_climb(trans, max_steps=5000, time_limit=20.0):
    """Local search: try single changes to 'b' (and occasionally 'a') to improve threshold."""
    a, b = trans[:]
    best_threshold, _ = _compute_threshold([a, b])
    if best_threshold is None:
        return trans, 0
    
    start_time = time.time()
    step = 0
    while step < max_steps:
        if time.time() - start_time > time_limit:
            break
        step += 1
        # Pick a random position and letter to modify
        letter = random.choice(['a', 'b'])
        pos = random.randint(0, N-1)
        old_val = a[pos] if letter == 'a' else b[pos]
        new_val = random.randint(0, N-1)
        if new_val == old_val:
            continue
        
        # Apply change
        if letter == 'a':
            a[pos] = new_val
        else:
            b[pos] = new_val
        
        new_threshold, _ = _compute_threshold([a, b])
        
        if new_threshold is None:
            # Revert
            if letter == 'a':
                a[pos] = old_val
            else:
                b[pos] = old_val
        elif new_threshold > best_threshold:
            best_threshold = new_threshold
        else:
            # Revert (or accept with small probability for simulated annealing, but keep simple)
            if letter == 'a':
                a[pos] = old_val
            else:
                b[pos] = old_val
    
    return [a, b], best_threshold

def construct():
    start_time = time.time()
    best_threshold = 0
    best_trans = None
    
    # Try pinwheel initializations with different k values
    for k in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        trans = _pinwheel_init(k)
        threshold, _ = _compute_threshold(trans)
        if threshold is not None and threshold > best_threshold:
            best_threshold = threshold
            best_trans = trans[:]
    
    # Try some random initializations
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        trans = _random_init()
        threshold, _ = _compute_threshold(trans)
        if threshold is not None and threshold > best_threshold:
            best_threshold = threshold
            best_trans = trans[:]
    
    # Hill-climb from best found
    if best_trans is not None:
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 5:
            best_trans, best_threshold = _hill_climb(best_trans, max_steps=20000, time_limit=remaining_time - 5)
    
    # If we still have time, try more random starts and hill-climb
    while time.time() - start_time < DEADLINE - 10:
        trans = _random_init()
        threshold, _ = _compute_threshold(trans)
        if threshold is not None:
            if threshold > best_threshold:
                best_threshold = threshold
                best_trans = trans[:]
            # Hill-climb from this random start
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 10:
                trans2, thr2 = _hill_climb(trans, max_steps=10000, time_limit=remaining - 10)
                if thr2 > best_threshold:
                    best_threshold = thr2
                    best_trans = trans2
    
    if best_trans is None:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_trans

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
