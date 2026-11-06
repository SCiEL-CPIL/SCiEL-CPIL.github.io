"""Search for a 12-state binary synchronizing automaton with a long reset threshold."""
import random
import time
import sys

N = 12
DEADLINE = 80.0  # seconds; evaluator kills at 90 s, keep margin
RNG = random.Random(12345)


def _bfs_reset_length(a, b):
    """Return the exact shortest reset word length for automaton (a, b).
    Returns None if the automaton does not synchronize.
    Uses BFS over the subset automaton (frozensets of states)."""
    # Map each state to its image under a and b
    # a[i] = image of state i under letter a
    # b[i] = image of state i under letter b
    full = frozenset(range(N))
    visited = {full}
    queue = [full]
    depth = 0
    while queue:
        depth += 1
        next_queue = []
        for s in queue:
            # Apply letter a
            sa = frozenset(a[i] for i in s)
            if len(sa) == 1:
                return depth
            if sa not in visited:
                visited.add(sa)
                next_queue.append(sa)
            # Apply letter b
            sb = frozenset(b[i] for i in s)
            if len(sb) == 1:
                return depth
            if sb not in visited:
                visited.add(sb)
                next_queue.append(sb)
        queue = next_queue
    return None  # not synchronizing


def _construct_baseline():
    """Cycle plus a constant map to state 0. This synchronizes."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    return a, b


def _random_transition(a, b):
    """Return a random valid transition list (complete deterministic)."""
    return [RNG.randrange(N) for _ in range(N)], [RNG.randrange(N) for _ in range(N)]


def _mutate(a, b):
    """Mutate one random transition in one random letter."""
    letter = RNG.randrange(2)
    state = RNG.randrange(N)
    new_val = RNG.randrange(N)
    if letter == 0:
        a2 = a[:]
        a2[state] = new_val
        return a2, b
    else:
        b2 = b[:]
        b2[state] = new_val
        return a, b2


def construct():
    start_time = time.monotonic()
    best_a, best_b = _construct_baseline()
    best_score = _bfs_reset_length(best_a, best_b)
    if best_score is None:
        # Should not happen for baseline, but guard
        best_score = 0

    # Local search with random restarts
    iteration = 0
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > DEADLINE - 2.0:
            break

        # Decide whether to restart or continue local search
        # Use a mix: mostly local search, occasional restart
        if iteration % 500 == 0 and iteration > 0:
            # Occasionally restart from a new random automaton
            if RNG.random() < 0.3:
                cur_a, cur_b = _random_transition(None, None)
                cur_score = _bfs_reset_length(cur_a, cur_b)
                if cur_score is None:
                    cur_a, cur_b = best_a, best_b
                    cur_score = best_score
                else:
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a, best_b = cur_a, cur_b
            else:
                # Restart from best
                cur_a, cur_b = best_a[:], best_b[:]
                cur_score = best_score
        else:
            cur_a, cur_b = best_a[:], best_b[:]
            cur_score = best_score

        # Local search: mutate and accept improvements
        local_iter = 0
        no_improve = 0
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > DEADLINE - 2.0:
                break

            new_a, new_b = _mutate(cur_a, cur_b)
            new_score = _bfs_reset_length(new_a, new_b)

            if new_score is not None and new_score > cur_score:
                cur_a, cur_b = new_a, new_b
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a, cur_b_best = cur_a, cur_b
                    best_a = cur_a
                    best_b = cur_b
            else:
                no_improve += 1
                # Accept small regressions with some probability (simulated annealing-like)
                # to escape local optima
                if new_score is not None and no_improve > 20 and RNG.random() < 0.05:
                    cur_a, cur_b = new_a, new_b
                    cur_score = new_score

            local_iter += 1
            if no_improve > 200:
                break  # stuck, break out of local search
            if local_iter > 200:
                break  # limit local search steps

        iteration += 1

        # Safety: check time more frequently
        if time.monotonic() - start_time > DEADLINE - 2.0:
            break

    return best_a, best_b
