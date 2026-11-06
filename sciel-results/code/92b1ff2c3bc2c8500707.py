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
    Computes the length of the shortest reset word for the automaton defined by trans_a and trans_b.
    Returns 0 if not synchronizing.
    Uses BFS on the subset automaton with precomputed transitions.
    """
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    # BFS from FULL_MASK
    size = 1 << N
    # Use bytearray for memory efficiency and speed
    dist = bytearray([255] * size) 
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()

        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]

        d = dist[cur] + 1
        nxt_a = next_a[cur]
        if dist[nxt_a] == 255:
            dist[nxt_a] = d
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == 255:
            dist[nxt_b] = d
            queue.append(nxt_b)

    return 0


def _generate_ping_pong_structures():
    """Generate ping-pong style automata known to have high thresholds."""
    structures = []

    # The canonical Ping-Pong automaton
    # a: shift +1, b: shift -1
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    structures.append((a[:], b[:]))
    
    # Other shifts
    for shift_a in range(1, N):
        for shift_b in range(1, N):
            if shift_a == shift_b:
                continue
            a = [(i + shift_a) % N for i in range(N)]
            b = [(i + shift_b) % N for i in range(N)]
            structures.append((a[:], b[:]))
            
    # Structure 2: One full cycle, one with a "defect"
    for defect_pos in range(N):
        for defect_dest in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [(i - 1) % N for i in range(N)]
            b[defect_pos] = defect_dest
            if b[defect_pos] != (defect_pos - 1) % N:
                structures.append((a[:], b[:]))

    return structures


def _refine_automaton(a, b, start_time, deadline):
    """
    Local search to refine an automaton, trying to maximize threshold.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)

    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    no_improve = 0
    max_no_improve = 100

    while time.time() < deadline:
        if no_improve > max_no_improve:
            break

        # Prioritize swapping destinations in the same letter, 
        # as this preserves permutation properties which are good for high thresholds.
        mutation_type = random.choice(['swap_a', 'swap_b', 'random_a', 'random_b'])

        if mutation_type == 'swap_a':
            i, j = random.sample(range(N), 2)
            cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[i], cur_a[j] = cur_a[j], cur_a[i]
                no_improve += 1

        elif mutation_type == 'swap_b':
            i, j = random.sample(range(N), 2)
            cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_b[i], cur_b[j] = cur_b[j], cur_b[i]
                no_improve += 1

        elif mutation_type == 'random_a':
            state = random.randint(0, N - 1)
            old = cur_a[state]
            new_dest = random.randint(0, N - 1)
            if new_dest != old:
                cur_a[state] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[state] = old
                    no_improve += 1
            else:
                no_improve += 1

        else:  # random_b
            state = random.randint(0, N - 1)
            old = cur_b[state]
            new_dest = random.randint(0, N - 1)
            if new_dest != old:
                cur_b[state] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[state] = old
                    no_improve += 1
            else:
                no_improve += 1

    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    random.seed(42)

    best_a = None
    best_b = None
    best_score = 0

    # Phase 1: Evaluate structured candidates
    structures = _generate_ping_pong_structures()
    random.shuffle(structures)

    for a_struct, b_struct in structures:
        if time.time() > deadline - 5.0:
            break

        score = compute_threshold(a_struct, b_struct)
        if score > best_score:
            best_score = score
            best_a = a_struct[:]
            best_b = b_struct[:]
            
            # If we hit the theoretical max (121) or a very high score, we can be more aggressive
            # but for now we continue to ensure we don't miss a slightly better one if 112 isn't max.
            # However, if best_score is very high, we can reduce time spent on lower candidates.

    # Phase 2: Local search refinement from best structured candidate
    if best_a is not None and time.time() < deadline - 3.0:
        refined_a, refined_b, refined_score = _refine_automaton(
            best_a, best_b, start_time, deadline - 3.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b

    # Phase 3: Try random starts with refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # If we already have a high score, use less time on random starts
        num_random_starts = min(5, int(remaining_time / 5))
        for _ in range(num_random_starts):
            if time.time() > deadline - 3.0:
                break

            a_rand = [random.randint(0, N - 1) for _ in range(N)]
            b_rand = [random.randint(0, N - 1) for _ in range(N)]

            score = compute_threshold(a_rand, b_rand)
            if score == 0:
                continue

            if score > best_score:
                best_score = score
                best_a = a_rand[:]
                best_b = b_rand[:]

            if score >= best_score - 5 and time.time() < deadline - 3.0:
                refined_a, refined_b, refined_score = _refine_automaton(
                    a_rand, b_rand, start_time, deadline - 3.0
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [(i - 1) % N for i in range(N)]
        return [a, b]

    return [best_a, best_b]


if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
