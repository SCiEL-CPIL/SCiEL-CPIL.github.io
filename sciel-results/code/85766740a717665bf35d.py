"""Černý problem, N=12: maximize the exact shortest reset-word length.

Strategy:
1. Systematically enumerate known optimal structural families (Pinwheel,
   rank-2 cycle, parameterized variants with multiple pins/defects).
2. For each candidate, compute the exact shortest reset word length via
   optimized BFS on the subset automaton.
3. Return the best synchronizing automaton found.
4. Use remaining time for targeted local search on the best candidate.

The BFS is optimized using precomputed transition tables and bitmask operations.
"""
import time
import random
from collections import deque

N = 12
_DEADLINE = 80.0  # seconds; evaluator kills at 90s, leave a safety margin


def _time_left():
    return _DEADLINE - time.time()


def _bfs_reset_length(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton.

    Returns (length, reset_word) where reset_word is a list of 0/1 letters.
    Returns (None, None) if the automaton does not synchronize.
    """
    full = (1 << N) - 1
    # Precompute image of every subset under each letter.
    img_a = [0] * (1 << N)
    img_b = [0] * (1 << N)
    for s in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if s >> i & 1:
                sa |= 1 << a[i]
                sb |= 1 << b[i]
        img_a[s] = sa
        img_b[s] = sb

    dist = [-1] * (1 << N)
    dist[full] = 0
    q = deque([full])
    while q:
        s = q.popleft()
        if s & (s - 1) == 0 and s != 0:  # a non-empty singleton
            return dist[s], None
        ns_a = img_a[s]
        ns_b = img_b[s]
        if dist[ns_a] == -1:
            dist[ns_a] = dist[s] + 1
            q.append(ns_a)
        if dist[ns_b] == -1:
            dist[ns_b] = dist[s] + 1
            q.append(ns_b)
    return None, None


def _pinwheel(pin_pos=0, pin_dest=1):
    """Vershik–Kondakov pinwheel automaton on N states with configurable pin.

    letter 'a': successor on the cycle
    letter 'b': predecessor on the cycle, except state pin_pos maps to pin_dest
    """
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    b[pin_pos] = pin_dest
    return a, b


def _cycle_with_defects(defect_positions, defect_dests):
    """Cycle automaton with specific defects (positions where b differs from cycle).

    letter 'a': successor on the cycle
    letter 'b': successor on the cycle, except at defect_positions where b[i]=defect_dests[j]
    """
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    for pos, dest in zip(defect_positions, defect_dests):
        b[pos] = dest
    return a, b


def _rank2_cycle():
    """Rank-2 cycle: a is full cycle, b maps all to 0 except state 0 -> 1."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    b[0] = 1
    return a, b


def _double_cycle():
    """Two interleaved cycles with a connecting map."""
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 2) % N for i in range(N)]
    b[0] = 0
    return a, b


def _shift_with_stuck(stuck_pos, stuck_dest):
    """Cycle with one stuck state."""
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[stuck_pos] = stuck_dest
    return a, b


def _two_pin_cycle(pin1_pos, pin1_dest, pin2_pos, pin2_dest):
    """Cycle with two pins (defects)."""
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[pin1_pos] = pin1_dest
    b[pin2_pos] = pin2_dest
    return a, b


def _reverse_cycle_with_pins(pin_positions, pin_dests):
    """Reverse cycle with specific pins."""
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    for pos, dest in zip(pin_positions, pin_dests):
        b[pos] = dest
    return a, b


def _enumerate_families():
    """Yield candidate (a, b) pairs from known structural families."""
    candidates = []
    
    # Family 1: Pinwheel with different pin positions and destinations
    for pin_pos in range(N):
        for pin_dest in range(N):
            if pin_dest != (pin_pos - 1) % N:  # skip if same as default
                candidates.append(_pinwheel(pin_pos, pin_dest))
    
    # Family 2: Cycle with single defect
    for pos in range(N):
        for dest in range(N):
            if dest != (pos + 1) % N:
                candidates.append(_cycle_with_defects([pos], [dest]))
    
    # Family 3: Rank-2 cycle variants
    candidates.append(_rank2_cycle())
    candidates.append(_double_cycle())
    
    # Family 4: Shift with stuck state
    for pos in range(N):
        for dest in range(N):
            if dest != (pos + 1) % N:
                candidates.append(_shift_with_stuck(pos, dest))
    
    # Family 5: Two-pin cycle (sample subset to keep it tractable)
    for p1 in range(N):
        for d1 in range(N):
            if d1 == (p1 + 1) % N:
                continue
            for p2 in range(N):
                if p2 == p1:
                    continue
                for d2 in range(N):
                    if d2 == (p2 + 1) % N:
                        continue
                    candidates.append(_two_pin_cycle(p1, d1, p2, d2))
    
    # Family 6: Reverse cycle with single pin (already covered by pinwheel, but try more)
    for pin_pos in range(N):
        for pin_dest in range(N):
            if pin_dest != (pin_pos - 1) % N:
                candidates.append(_reverse_cycle_with_pins([pin_pos], [pin_dest]))
    
    return candidates


def _local_search(a0, b0, time_limit=60.0):
    """Local search to find an automaton with a longer reset word.
    
    Mutates the transition tables randomly and checks synchronizing property.
    Returns the best (a, b) found with its reset length, or (a0, b0, 0) if no improvement.
    """
    best_a, best_b = list(a0), list(b0)
    best_len, _ = _bfs_reset_length(best_a, best_b)
    if best_len is None:
        best_len = 0
    
    start_time = time.time()
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if _time_left() < 5.0:
            break
        
        # Mutate: pick a random state and letter, change its destination
        a = list(best_a)
        b = list(best_b)
        
        # Try a few mutations in one step
        for _ in range(2):
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            if letter == 0:
                a[state] = new_dest
            else:
                b[state] = new_dest
        
        # Check if synchronizing and get reset length
        length, _ = _bfs_reset_length(a, b)
        if length is not None and length > best_len:
            best_a, best_b = a, b
            best_len = length
    
    return best_a, best_b, best_len


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Step 1: Enumerate known structural families
    candidates = _enumerate_families()
    
    for a, b in candidates:
        if _time_left() < 3.0:
            break
        length, _ = _bfs_reset_length(a, b)
        if length is not None and length > best_len:
            best_len = length
            best_a = a
            best_b = b
    
    # Step 2: If we found something, try local search to improve
    if best_a is not None and _time_left() > 10.0:
        a, b, improved_len = _local_search(best_a, best_b, time_limit=_time_left() - 5.0)
        if improved_len > best_len:
            best_a, best_b, best_len = a, b, improved_len
    
    # Step 3: Fallback if nothing found
    if best_a is None:
        a, b = _pinwheel(0, 1)
        length, _ = _bfs_reset_length(a, b)
        if length is not None:
            return a, b
        return [(i + 1) % N for i in range(N)], [0] * N
    
    return best_a, best_b
# EVOLVE-BLOCK-END
