import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 86.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """BFS on subset automaton; returns exact shortest reset word length, 0 if not synchronizing."""
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


def _threshold_cycle_compress(n, e, t, w):
    """Analytic threshold for a = (i+1)%n, b compresses to [0,w-1] except escape e->t.
    Requires the escape state to be outside the compressed range.
    Returns the predicted shortest reset word length."""
    if not (e >= w and t >= w):
        return -1
    return (n - 1) - w - ((e - t - 1) % n)


def _construct_cycle_compress(n, e, t, w):
    """Build automaton: a is full cycle, b maps [0,w-1]->identity, e->t, all others->0."""
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(w):
        b[i] = i
    b[e] = t
    return a, b


def _pairwise_separation_energy(a, b):
    """
    Custom energy function: estimate reset difficulty by maximizing
    the maximum pairwise distance between states under current mappings.
    
    For each pair of states (i, j), compute the minimum number of steps
    needed to merge them. Higher energy = harder to synchronize = potentially
    longer reset word.
    """
    n = len(a)
    # For each state pair, estimate separation
    # Use BFS on state pairs to find minimum merge distance
    # This is expensive, so we use a heuristic: count how many states
    # can reach each other in 1-2 steps
    
    # Heuristic: for each state, count distinct images after applying a or b
    # More distinct images = harder to synchronize
    
    # Compute forward reachability in 2 steps for each state
    two_step_reach = [set() for _ in range(n)]
    for i in range(n):
        # After 'a'
        ia = a[i]
        two_step_reach[i].add(ia)
        two_step_reach[i].add(a[ia])
        two_step_reach[i].add(b[ia])
        # After 'b'
        ib = b[i]
        two_step_reach[i].add(ib)
        two_step_reach[i].add(a[ib])
        two_step_reach[i].add(b[ib])
    
    # Energy = sum of sizes of 2-step reach sets (higher = more separated)
    energy = 0
    for i in range(n):
        energy += len(two_step_reach[i])
    
    return energy


def _simulated_annealing(a_init, b_init, rng, deadline, best_holder):
    """
    Simulated Annealing over the full transition space.
    
    State: (a, b) - two transition functions
    Energy: _pairwise_separation_energy (we want to maximize this)
    But we also need to ensure synchronizability, so we combine:
    - If not synchronizing: penalty
    - If synchronizing: use actual threshold as primary objective
    
    Strategy: 
    - Start from a good initial solution
    - Mutate one transition entry at a time
    - Accept worse solutions with probability exp(-delta/T)
    - Cool down over time
    """
    n = len(a_init)
    cur_a = a_init[:]
    cur_b = b_init[:]
    cur_threshold = compute_threshold(cur_a, cur_b)
    
    if cur_threshold == 0:
        # Not synchronizing, try to fix
        return None, None, 0
    
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_threshold = cur_threshold
    
    if best_threshold > best_holder[0]:
        best_holder[0] = best_threshold
        best_holder[1] = best_a[:]
        best_holder[2] = best_b[:]
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    T = T_start
    steps_since_best = 0
    max_steps_since_best = 1000
    
    while time.time() < deadline and steps_since_best < max_steps_since_best:
        # Choose which transition to mutate and which state
        mutate_a = rng.random() < 0.3  # 30% chance to mutate a
        state = rng.randint(0, n - 1)
        
        if mutate_a:
            old = cur_a[state]
            new_dest = rng.randint(0, n - 1)
            if new_dest == old:
                continue
            cur_a[state] = new_dest
        else:
            old = cur_b[state]
            new_dest = rng.randint(0, n - 1)
            if new_dest == old:
                continue
            cur_b[state] = new_dest
        
        new_threshold = compute_threshold(cur_a, cur_b)
        
        if new_threshold == 0:
            # Not synchronizing - reject with high probability
            if rng.random() < 0.9:
                if mutate_a:
                    cur_a[state] = old
                else:
                    cur_b[state] = old
                T *= cooling_rate
                continue
            # Otherwise keep it (exploration)
            delta = -cur_threshold  # big penalty
        else:
            delta = new_threshold - cur_threshold
        
        # Acceptance criterion
        if delta >= 0:
            # Better or equal - accept
            cur_threshold = new_threshold
            if new_threshold > best_threshold:
                best_threshold = new_threshold
                best_a = cur_a[:]
                best_b = cur_b[:]
                steps_since_best = 0
                if best_threshold > best_holder[0]:
                    best_holder[0] = best_threshold
                    best_holder[1] = best_a[:]
                    best_holder[2] = best_b[:]
        else:
            # Worse - accept with probability exp(delta/T)
            if T > 0:
                prob = min(1.0, -delta / T)
                if rng.random() < prob:
                    cur_threshold = new_threshold
                else:
                    # Reject
                    if mutate_a:
                        cur_a[state] = old
                    else:
                        cur_b[state] = old
            else:
                # Reject
                if mutate_a:
                    cur_a[state] = old
                else:
                    cur_b[state] = old
        
        steps_since_best += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    return best_a, best_b, best_threshold


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(42)

    best_holder = [0, None, None]  # [score, a, b]

    # Phase 1: exhaustive analytic scan of the cycle+compress family (very fast)
    n = N
    best_family = None
    best_family_score = 0
    for w in range(1, n // 2 + 1):
        for e in range(w, n):
            for t in range(w, n):
                if t == e:
                    continue
                score = _threshold_cycle_compress(n, e, t, w)
                if score > best_family_score:
                    best_family_score = score
                    best_family = (e, t, w)
    # Verify the family best with real BFS and store
    if best_family is not None:
        e, t, w = best_family
        a, b = _construct_cycle_compress(n, e, t, w)
        real = compute_threshold(a, b)
        if real > best_holder[0]:
            best_holder[0] = real
            best_holder[1] = a[:]
            best_holder[2] = b[:]

    # Phase 2: Simulated Annealing from the analytic optimum
    if time.time() < deadline - 4.0 and best_holder[1] is not None:
        ra, rb, rs = _simulated_annealing(best_holder[1], best_holder[2], rng, deadline - 4.0, best_holder)

    # Phase 3: try a few random perturbations of the family + simulated annealing
    if time.time() < deadline - 4.0 and best_family is not None:
        e, t, w = best_family
        for strength in (0.15, 0.3, 0.45):
            if time.time() > deadline - 4.0:
                break
            a0, b0 = _construct_cycle_compress(n, e, t, w)
            a0 = a0[:]
            b0 = b0[:]
            # Randomize b
            for i in range(n):
                if rng.random() < strength:
                    b0[i] = rng.randint(0, n - 1)
            score0 = compute_threshold(a0, b0)
            if score0 == 0:
                continue
            if score0 > best_holder[0]:
                best_holder[0] = score0
                best_holder[1] = a0[:]
                best_holder[2] = b0[:]
            if time.time() < deadline - 4.0:
                _simulated_annealing(a0, b0, rng, deadline - 4.0, best_holder)

    # Fallback: guaranteed synchronizing automaton
    if best_holder[1] is None or best_holder[0] == 0:
        a = [(i + 1) % n for i in range(n)]
        b = [0] * n
        return [a, b]

    return [best_holder[1], best_holder[2]]


if __name__ == "__main__":
    a, b = construct()
    print("Best score:", compute_threshold(a, b))
    print("A:", a)
    print("B:", b)
