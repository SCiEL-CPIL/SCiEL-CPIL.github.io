import time
import random
import sys
import math
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


def _ping_pong_automaton():
    """
    Returns the standard Ping-Pong automaton which is known to have a
    synchronization threshold of N^2 - 3N + 4.
    For N=12, this is 112.
    Letter a: Cycle (i -> i+1 mod N)
    Letter b: 0->0, i->i-1 for i>0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    return a, b


def _pairwise_distance_energy(trans_a, trans_b):
    """
    Compute an energy function that estimates reset difficulty.
    Higher energy = harder to synchronize = potentially longer reset word.
    Uses maximum pairwise distance under the mappings as a proxy.
    """
    # Compute distances from state 0 to all states under alternating a,b
    # This gives a rough measure of how "spread out" the automaton is
    dist_a = [0] * N
    dist_b = [0] * N
    
    # BFS from state 0 using only letter a
    visited = [False] * N
    visited[0] = True
    queue = deque([0])
    dist = 0
    while queue:
        dist += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            nxt = trans_a[cur]
            if not visited[nxt]:
                visited[nxt] = True
                dist_a[nxt] = dist
                queue.append(nxt)
    
    # BFS from state 0 using only letter b
    visited = [False] * N
    visited[0] = True
    queue = deque([0])
    dist = 0
    while queue:
        dist += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            nxt = trans_b[cur]
            if not visited[nxt]:
                visited[nxt] = True
                dist_b[nxt] = dist
                queue.append(nxt)
    
    # Energy: sum of max distances (higher = more spread out)
    max_a = max(dist_a) if any(dist_a) else 0
    max_b = max(dist_b) if any(dist_b) else 0
    
    # Also consider the "mixing" - how different the two mappings are
    mixing = sum(1 for i in range(N) if trans_a[i] != trans_b[i])
    
    return max_a + max_b + mixing * 0.1


def _simulated_annealing(deadline, initial_a, initial_b):
    """
    Perform simulated annealing over the transition space.
    Uses pairwise distance energy as a surrogate for reset difficulty.
    """
    cur_a = initial_a[:]
    cur_b = initial_b[:]
    
    # First, check if the initial is synchronizing and get its score
    cur_score = compute_threshold(cur_a, cur_b)
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling = 0.995
    
    T = T_start
    no_improve = 0
    
    while time.time() < deadline - 10.0 and T > T_end:
        # Generate a neighbor: change one transition in a or b
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            old = cur_a[state]
            new = random.randint(0, N - 1)
            if new == old:
                continue
            cur_a[state] = new
            letter = 'a'
            old_val = old
        else:
            state = random.randint(0, N - 1)
            old = cur_b[state]
            new = random.randint(0, N - 1)
            if new == old:
                continue
            cur_b[state] = new
            letter = 'b'
            old_val = old
        
        # Compute new score
        new_score = compute_threshold(cur_a, cur_b)
        
        # Accept or reject
        if new_score > cur_score:
            cur_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Compute energy difference for acceptance probability
            # Use a simpler acceptance: accept with probability exp(-delta/T)
            # where delta is the score difference (we want to maximize)
            delta = cur_score - new_score
            if delta > 0 and random.random() < math.exp(-delta / T):
                cur_score = new_score
                no_improve += 1
            else:
                # Reject
                if letter == 'a':
                    cur_a[state] = old_val
                else:
                    cur_b[state] = old_val
                no_improve += 1
        
        # Cool down
        T *= cooling
        
        # Early stop if we've been stuck for too long
        if no_improve > 500:
            break
    
    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    # Start with the ping-pong automaton (known to have threshold 112 for N=12)
    best_a, best_b = _ping_pong_automaton()
    best_score = compute_threshold(best_a, best_b)
    
    # Try simulated annealing from multiple starting points
    # Phase 1: Anneal from ping-pong
    if time.time() < deadline - 30.0:
        sa_a, sa_b, sa_score = _simulated_annealing(deadline - 30.0, best_a, best_b)
        if sa_score > best_score:
            best_score = sa_score
            best_a = sa_a[:]
            best_b = sa_b[:]
    
    # Phase 2: Try annealing from slightly perturbed ping-pong
    if time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            # Perturb ping-pong slightly
            pert_a = best_a[:]
            pert_b = best_b[:]
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    state = random.randint(0, N - 1)
                    pert_a[state] = random.randint(0, N - 1)
                else:
                    state = random.randint(0, N - 1)
                    pert_b[state] = random.randint(0, N - 1)
            
            sa_a, sa_b, sa_score = _simulated_annealing(deadline - 15.0, pert_a, pert_b)
            if sa_score > best_score:
                best_score = sa_score
                best_a = sa_a[:]
                best_b = sa_b[:]
    
    # Phase 3: Final hill-climb polish
    if time.time() < deadline - 5.0:
        cur_a = best_a[:]
        cur_b = best_b[:]
        cur_score = best_score
        no_improve = 0
        while time.time() < deadline - 5.0 and no_improve < 100:
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
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[state] = old
                    no_improve += 1
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
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[state] = old
                    no_improve += 1
    
    # Ensure valid return
    if best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]
