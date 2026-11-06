"""Černý problem, N=12: maximize the exact shortest reset-word length.

Strategy (structural change):
1. Start with the Vershik–Kondakov "pinwheel" automaton (known threshold 2N-3 = 21).
2. Use an optimized BFS on the subset automaton with bitmask operations for exact
   threshold computation.
3. Perform SIMULATED ANNEALING with a composite objective function that rewards
   both synchronizability and long reset words. This allows the search to accept
   temporarily worse solutions to escape local minima.
4. Use DIRECTED PERTURBATIONS that target structural properties known to
   increase reset length: creating "bottleneck" states with high in-degree
   concentration and introducing "delay" structures.
5. Keep the best valid synchronizing automaton found within the time budget.
6. Return the best automaton found.

The BFS is optimized using precomputed transition tables and bitmask operations.
The annealing schedule uses exponential cooling with periodic reheating.
"""
import time
import random
import math
from collections import deque

N = 12
_DEADLINE = 85.0  # seconds; evaluator kills at 90s, leave a safety margin


def _time_left():
    return _DEADLINE - time.time()


def _pinwheel():
    """Vershik–Kondakov pinwheel automaton on N states.

    letter 'a': successor on the cycle
    letter 'b': predecessor on the cycle, except state 0 maps to 1 (the pin)
    """
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    b[0] = 1  # the pin
    return a, b


def _seed():
    """Fallback: simple synchronizing automaton (cycle + map to 0)."""
    return [(i + 1) % N for i in range(N)], [0] * N


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
    prev = [(-1, -1)] * (1 << N)
    dist[full] = 0
    q = deque([full])
    while q:
        s = q.popleft()
        if s & (s - 1) == 0 and s != 0:  # a non-empty singleton
            # reconstruct word
            word = []
            cur = s
            while cur != full:
                p, letter = prev[cur]
                word.append(letter)
                cur = p
            word.reverse()
            return dist[s], word
        for letter, img in ((0, img_a), (1, img_b)):
            ns = img[s]
            if dist[ns] == -1:
                dist[ns] = dist[s] + 1
                prev[ns] = (s, letter)
                q.append(ns)
    return None, None


def _bfs_reset_length_fast(a, b):
    """Optimized BFS that only returns the length (no word reconstruction).
    Uses a slightly different approach for speed: we don't need the word,
    just the length. Also uses a flat array for prev to reduce overhead.
    """
    full = (1 << N) - 1
    # Precompute image of every subset under each letter using a faster method
    # For each state i, precompute its contribution to the image
    # img[s] = OR over i in s of (1 << letter[i])
    
    # Fast precomputation using DP: img[s] = img[s without lowest bit] | (1 << letter[lowest_bit])
    img_a = [0] * (1 << N)
    img_b = [0] * (1 << N)
    
    for s in range(1, 1 << N):
        # Find lowest set bit
        lb = s & (-s)
        i = lb.bit_length() - 1
        prev_s = s ^ lb
        img_a[s] = img_a[prev_s] | (1 << a[i])
        img_b[s] = img_b[prev_s] | (1 << b[i])

    # BFS
    visited = bytearray(1 << N)  # 0 = unvisited, 1 = visited
    visited[full] = 1
    dist = 0
    current_layer = [full]
    
    while current_layer:
        next_layer = []
        for s in current_layer:
            # Check if singleton
            if s & (s - 1) == 0 and s != 0:
                return dist
            # Explore neighbors
            ns_a = img_a[s]
            ns_b = img_b[s]
            if not visited[ns_a]:
                visited[ns_a] = 1
                next_layer.append(ns_a)
            if not visited[ns_b]:
                visited[ns_b] = 1
                next_layer.append(ns_b)
        current_layer = next_layer
        dist += 1
    
    return None


def _compute_indegree_concentration(a, b):
    """Compute a proxy for how 'bottlenecked' the automaton is.
    
    Higher concentration means more states map to the same destination under
    one or both letters, which can create delays in synchronization.
    """
    # For each letter, compute the in-degree distribution
    conc = 0
    for letter in [a, b]:
        from collections import Counter
        counts = Counter(letter)
        # Sum of squares of in-degrees (higher = more concentrated)
        conc += sum(c * c for c in counts.values())
    return conc


def _directed_perturbation(a, b, rng):
    """Apply a directed perturbation that targets structural properties
    known to increase reset length.
    
    Strategy:
    1. Identify states with low in-degree (potential bottlenecks)
    2. Redirect some states to create higher concentration
    3. Introduce 'delay' by making certain states require more steps to synchronize
    """
    a = list(a)
    b = list(b)
    
    # Compute current in-degrees
    from collections import Counter
    in_a = Counter(a)
    in_b = Counter(b)
    
    # Find states with low in-degree under 'a' (potential bottlenecks)
    low_in_a = [s for s in range(N) if in_a.get(s, 0) <= 1]
    low_in_b = [s for s in range(N) if in_b.get(s, 0) <= 1]
    
    # With some probability, create a bottleneck by redirecting multiple states
    # to the same destination under one letter
    if rng.random() < 0.3 and low_in_a:
        # Pick a target state with low in-degree
        target = rng.choice(low_in_a)
        # Pick 1-2 source states to redirect to target
        for _ in range(rng.randint(1, 2)):
            source = rng.randint(0, N - 1)
            if a[source] != target:
                a[source] = target
    
    if rng.random() < 0.3 and low_in_b:
        # Pick a target state with low in-degree
        target = rng.choice(low_in_b)
        # Pick 1-2 source states to redirect to target
        for _ in range(rng.randint(1, 2)):
            source = rng.randint(0, N - 1)
            if b[source] != target:
                b[source] = target
    
    # With some probability, introduce a 'delay' structure:
    # Make a state that is a sink under one letter also a sink under the other
    # This creates a state that is hard to escape
    if rng.random() < 0.2:
        # Find states that are sinks under 'a' (map to themselves)
        sinks_a = [s for s in range(N) if a[s] == s]
        if sinks_a:
            sink = rng.choice(sinks_a)
            # With some probability, also make it a sink under 'b'
            if rng.random() < 0.5:
                b[sink] = sink
            else:
                # Or make it map to a state that is a sink under 'a'
                if sinks_a:
                    b[sink] = rng.choice(sinks_a)
    
    return a, b


def _simulated_annealing(a0, b0, time_limit=60.0):
    """Simulated annealing with composite objective and directed perturbations.
    
    The objective function rewards:
    1. Synchronizability (hard requirement)
    2. Long reset word length (primary goal)
    3. High in-degree concentration (proxy for potential delay structures)
    
    The annealing schedule uses exponential cooling with periodic reheating
    to escape local minima.
    """
    best_a, best_b = list(a0), list(b0)
    best_len = _bfs_reset_length_fast(best_a, best_b)
    if best_len is None:
        best_len = 0
    best_obj = best_len * 10 + _compute_indegree_concentration(best_a, best_b)
    
    start_time = time.time()
    rng = random.Random(54321)
    
    # Annealing parameters
    T0 = 100.0  # initial temperature
    T_min = 0.01  # minimum temperature
    cooling_rate = 0.995  # cooling per step
    reheat_interval = 200  # reheat every N steps
    reheat_factor = 0.5  # reheat to this fraction of T0
    
    T = T0
    current_a, current_b = list(a0), list(b0)
    current_len = best_len
    current_obj = best_obj
    
    step_count = 0
    max_steps = 100000
    
    while time.time() - start_time < time_limit and step_count < max_steps:
        if _time_left() < 3.0:
            break
        
        step_count += 1
        
        # Periodic reheating
        if step_count % reheat_interval == 0:
            T = max(T, T0 * reheat_factor)
        
        # Choose perturbation strategy
        if rng.random() < 0.4:
            # Directed perturbation
            new_a, new_b = _directed_perturbation(current_a, current_b, rng)
        else:
            # Random single mutation
            new_a, new_b = list(current_a), list(current_b)
            letter = rng.randint(0, 1)
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            if letter == 0:
                new_a[state] = new_dest
            else:
                new_b[state] = new_dest
        
        # Evaluate new solution
        new_len = _bfs_reset_length_fast(new_a, new_b)
        if new_len is not None:
            new_obj = new_len * 10 + _compute_indegree_concentration(new_a, new_b)
        else:
            # Non-synchronizing: penalize heavily
            new_obj = -1000 + _compute_indegree_concentration(new_a, new_b)
        
        # Compute acceptance probability
        delta_obj = new_obj - current_obj
        if delta_obj >= 0:
            # Always accept improvements
            accept = True
        else:
            # Accept with probability exp(-delta_obj / T)
            if T > T_min:
                accept = rng.random() < math.exp(delta_obj / T)
            else:
                accept = False
        
        if accept:
            current_a, current_b = new_a, new_b
            current_len = new_len if new_len is not None else 0
            current_obj = new_obj
            
            # Update best if better
            if new_len is not None and new_len > best_len:
                best_a, best_b = list(new_a), list(new_b)
                best_len = new_len
                best_obj = new_obj
        
        # Cool down
        T = max(T * cooling_rate, T_min)
    
    return best_a, best_b, best_len


# EVOLVE-BLOCK-START
def construct():
    # Start with the pinwheel automaton
    a, b = _pinwheel()
    
    # Verify the pinwheel's exact threshold
    length = _bfs_reset_length_fast(a, b)
    if length is None:
        # Fallback to seed if pinwheel doesn't synchronize (shouldn't happen)
        return _seed()
    
    best_len = length
    
    # Try simulated annealing to improve
    if _time_left() > 10.0:
        a, b, best_len = _simulated_annealing(a, b, time_limit=_time_left() - 5.0)
    
    return a, b
# EVOLVE-BLOCK-END
