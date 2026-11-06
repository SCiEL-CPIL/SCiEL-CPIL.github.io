import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    """Precompute subset transitions for one letter using dynamic programming."""
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """
    Computes the length of the shortest reset word for the automaton defined by
    trans_a and trans_b. Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]

        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)

    return 0


def _valid_ping_pong():
    """
    Returns the standard Ping-Pong automaton which is known to have a 
    synchronization threshold of N^2 - 3N + 4.
    For N=12, this is 112.
    Letter a: Cycle (i -> i+1 mod N)
    Letter b: 0->0, i->i-1 for i>0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    return a, b


def _try_pinwheels():
    """
    Try pinwheel-type constructions which are known to sometimes exceed ping-pong.
    These involve a cyclic permutation for one letter and a specific 'defective'
    cycle for the other.
    """
    best_a = None
    best_b = None
    best_score = 0

    a_cycle = [(i + 1) % N for i in range(N)]

    # Standard ping-pong is already covered, but let's try other "defective cycles"
    # A defective cycle has one state mapping to something other than the next in the cycle.
    # For maximum threshold, the defect should be a "long" jump or a specific structure.
    
    # Known best constructions often involve:
    # a: i -> i+1
    # b: i -> i+1 for i != k, b[k] = 0 (or similar)
    # Or variations where the cycle is broken in a way that forces a long path.
    
    # Let's iterate over possible defects in a second cycle
    for defect_pos in range(N):
        for defect_target in range(N):
            b = [(i + 1) % N for i in range(N)]
            b[defect_pos] = defect_target
            
            # Skip if it's just the identity or trivially small
            # Quick check: if b is a single cycle, it's a permutation, so no sync unless a breaks it
            # Actually, if both are permutations, no sync. 
            # If b is a cycle with one defect, it has a tail into a cycle.
            
            s = compute_threshold(a_cycle, b)
            if s > best_score:
                best_score = s
                best_a = a_cycle[:]
                best_b = b[:]
                
    return best_a, best_b, best_score


def _random_start():
    """Generate a random starting point, biased towards cyclic structures."""
    a = list(range(N))
    b = list(range(N))
    
    # Shuffle to create random permutations, then introduce defects
    random.shuffle(a)
    random.shuffle(b)
    
    # Introduce defects to ensure synchronizability
    for _ in range(random.randint(1, 3)):
        i = random.randint(0, N - 1)
        a[i] = random.randint(0, N - 1)
    for _ in range(random.randint(1, 3)):
        i = random.randint(0, N - 1)
        b[i] = random.randint(0, N - 1)
        
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # 1) Initialize with the valid Ping-Pong automaton (Score ~112)
    ha, hb = _valid_ping_pong()
    hs = compute_threshold(ha, hb)
    if hs > best_score:
        best_score = hs
        best_a = ha[:]
        best_b = hb[:]

    # 2) Try Pinwheel constructions (structured search)
    if time.time() < deadline - 20.0:
        pa, pb, ps = _try_pinwheels()
        if pa is not None and ps > best_score:
            best_score = ps
            best_a = pa[:]
            best_b = pb[:]

    # 3) Hill Climbing from the best found so far
    if best_a is not None and time.time() < deadline - 10.0:
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_score = best_score
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random transition to mutate
            if random.random() < 0.5:
                state = random.randint(0, N - 1)
                old = cur_a[state]
                new = random.randint(0, N - 1)
                if new == old:
                    continue
                cur_a[state] = new
                s = compute_threshold(cur_a, cur_b)
                if s > cur_score:
                    cur_score = s
                    no_improve_count = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[state] = old
                    no_improve_count += 1
            else:
                state = random.randint(0, N - 1)
                old = cur_b[state]
                new = random.randint(0, N - 1)
                if new == old:
                    continue
                cur_b[state] = new
                s = compute_threshold(cur_a, cur_b)
                if s > cur_score:
                    cur_score = s
                    no_improve_count = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[state] = old
                    no_improve_count += 1

    # 4) If hill climbing stalled or didn't start, try a few random restarts
    if time.time() < deadline - 5.0:
        restarts = 0
        max_restarts = 10
        while time.time() < deadline - 5.0 and restarts < max_restarts:
            ra, rb = _random_start()
            rs = compute_threshold(ra, rb)
            if rs > best_score:
                best_score = rs
                best_a = ra[:]
                best_b = rb[:]
                restarts = 0
            else:
                restarts += 1

    # Ensure valid return
    if best_a is None or best_score == 0:
        # Fallback to valid ping-pong if something went wrong
        a, b = _valid_ping_pong()
        return [a, b]

    return [best_a, best_b]
