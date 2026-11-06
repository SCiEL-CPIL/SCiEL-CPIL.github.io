import time
import random
import sys
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0  # Safety margin before 90s kill

# Precompute subset transitions helper
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
        # Check if singleton
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

def _structural_starts():
    """
    Generate diverse, structure-aware starting automata.
    These are known to be in the vicinity of high-threshold solutions.
    """
    starts = []
    a_cycle = [(i + 1) % N for i in range(N)]
    a_rev = [(i - 1) % N for i in range(N)]

    # 1. Standard Ping-Pong: a=cycle, b=reset-to-0-down
    b_ping_pong = [0] + [i - 1 for i in range(1, N)]
    starts.append((a_cycle[:], b_ping_pong[:]))
    starts.append((a_rev[:], b_ping_pong[:]))

    # 2. Defective Cycles: a=cycle, b=cycle with one defect
    # This often creates longer synchronization paths as the defect "traps" states
    for dp in range(N):
        for dd in range(N):
            b = [(i + 1) % N for i in range(N)]
            # Ensure it's actually a defect (change destination)
            if dd != (dp + 1) % N:
                b[dp] = dd
                starts.append((a_cycle[:], b[:]))
    
    # 3. Pinwheel/Rotation: a=cycle, b=shift by k
    for shift in range(1, N):
        b = [(i + shift) % N for i in range(N)]
        starts.append((a_cycle[:], b[:]))

    return starts

def _mutate(cur_a, cur_b):
    """
    Apply one structure-aware mutation.
    Returns a new tuple (new_a, new_b) to allow safe rollback if needed.
    """
    new_a = cur_a[:]
    new_b = cur_b[:]
    r = random.random()
    
    if r < 0.30:
        # Cycle defect: change one destination to break symmetry
        state = random.randint(0, N - 1)
        letter = random.choice([0, 1])
        arr = new_a if letter == 0 else new_b
        new_dest = random.randint(0, N - 1)
        if new_dest == arr[state]:
            new_dest = (new_dest + 1) % N
        if letter == 0:
            new_a[state] = new_dest
        else:
            new_b[state] = new_dest
    elif r < 0.55:
        # Reset adjustment: modify the 'b' letter specifically
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        if new_dest == new_b[state]:
            new_dest = (new_dest + 1) % N
        new_b[state] = new_dest
    elif r < 0.75:
        # Asymmetry: swap destinations of two states in one letter
        i, j = random.sample(range(N), 2)
        letter = random.choice([0, 1])
        if letter == 0:
            new_a[i], new_a[j] = new_a[j], new_a[i]
        else:
            new_b[i], new_b[j] = new_b[j], new_b[i]
    else:
        # Random single-transition edit
        state = random.randint(0, N - 1)
        letter = random.choice([0, 1])
        new_dest = random.randint(0, N - 1)
        if letter == 0:
            new_a[state] = new_dest
        else:
            new_b[state] = new_dest
            
    return new_a, new_b

def _anneal(start_a, start_b, deadline):
    """
    Simulated Annealing to maximize the reset threshold.
    """
    cur_a = start_a[:]
    cur_b = start_b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    
    # If initial is not synchronizing, we can't start annealing from here
    # We need a valid starting point. If 0, return 0.
    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    # Temperature schedule
    T0 = 15.0
    Tend = 0.05
    # We estimate iterations based on time, but let's use a fixed number 
    # and check time inside the loop.
    # 1000 iterations is a reasonable upper bound for the time budget if fast.
    n_iter = 1500 
    cool = (Tend / T0) ** (1.0 / n_iter) if T0 > 0 else 1.0

    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        T = T0 * (cool ** it)

        new_a, new_b = _mutate(cur_a, cur_b)
        new_score = compute_threshold(new_a, new_b)

        if new_score > cur_score:
            cur_a, cur_b, cur_score = new_a, new_b, new_score
            if cur_score > best_score:
                best_score = cur_score
                best_a, best_b = cur_a[:], cur_b[:]
        else:
            # Accept worse with probability exp((new - cur)/T)
            # If new_score is 0, it's not synchronizing, usually we reject immediately
            # unless we want to allow non-synchronizing states to bridge gaps, 
            # but compute_threshold returns 0 for non-sync, which is very bad.
            # So we only accept if new_score > 0 or if T is very high?
            # Actually, if new_score is 0, the "energy" is infinite.
            # Let's only accept non-zero worse scores.
            if new_score > 0:
                delta = new_score - cur_score  # negative
                # Avoid exp overflow/underflow issues
                if delta > -T * 100: # heuristic cutoff
                    if random.random() < math.exp(delta / max(T, 1e-9)):
                        cur_a, cur_b, cur_score = new_a, new_b, new_score

    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # 1) Evaluate diverse structural starts
    # This is cheap and ensures we have a strong baseline (likely 112+)
    for sa, sb in _structural_starts():
        if time.time() > deadline - 30.0:
            break
        s = compute_threshold(sa, sb)
        if s > best_score:
            best_score = s
            best_a, best_b = sa[:], sb[:]

    # If we didn't find any synchronizing start (unlikely), fallback to ping-pong
    if best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] + [i - 1 for i in range(1, N)]
        best_a, best_b = a, b
        best_score = compute_threshold(best_a, best_b)

    # 2) Simulated Annealing from the best structural start
    # This is the main improvement driver.
    if time.time() < deadline - 5.0:
        ba, bb, bs = _anneal(best_a, best_b, deadline - 5.0)
        if bs > best_score:
            best_score = bs
            best_a, best_b = ba, bb

    # 3) If time remains, try annealing from a few other promising starts
    # or random starts to find other local optima
    if time.time() < deadline - 2.0:
        # Pick a few other structural starts that were close but not best
        # Or just try random restarts with annealing
        for _ in range(2): # Max 2 more annealing runs
            if time.time() > deadline - 2.0:
                break
            
            # Generate a random start biased towards cycles
            # Or pick another structural start
            if random.random() < 0.5:
                sa, sb = random.choice(_structural_starts())
            else:
                # Random permutation start
                perm_a = list(range(N))
                random.shuffle(perm_a)
                perm_b = list(range(N))
                random.shuffle(perm_b)
                sa, sb = perm_a, perm_b
                
            rs = compute_threshold(sa, sb)
            if rs == 0:
                continue
                
            # If this random/structural start is already better, update
            if rs > best_score:
                best_score = rs
                best_a, best_b = sa[:], sb[:]
                continue
                
            # Anneal from this start
            ba, bb, bs = _anneal(sa, sb, deadline - 2.0)
            if bs > best_score:
                best_score = bs
                best_a, best_b = ba, bb

    # Ensure valid return
    if best_a is None or best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]
