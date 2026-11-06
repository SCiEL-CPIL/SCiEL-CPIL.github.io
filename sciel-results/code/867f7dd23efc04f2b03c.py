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
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])

    while queue:
        cur = queue.popleft()

        # Check if singleton
        if cur & (cur - 1) == 0:
            return dist[cur]

        d = dist[cur] + 1
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = d
            queue.append(nxt_a)

        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = d
            queue.append(nxt_b)

    return 0


def _generate_ping_pong_structures():
    """Generate ping-pong style automata known to have high thresholds."""
    structures = []

    # Structure 1: Two disjoint cycles of length N/2
    for perm_a in range(1, N // 2 + 1):
        a = [0] * N
        b = [0] * N
        for i in range(N // 2):
            a[i] = (i + perm_a) % (N // 2)
            a[N // 2 + i] = N // 2 + (i + perm_a) % (N // 2)
        for i in range(N // 2):
            b[i] = (i + 1) % (N // 2)
            b[N // 2 + i] = N // 2 + (i + 1) % (N // 2)
        structures.append((a[:], b[:]))

    # Structure 2: One full cycle, one with a "defect"
    for defect_pos in range(N):
        for defect_dest in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [(i + 1) % N for i in range(N)]
            b[defect_pos] = defect_dest
            if b[defect_pos] != (defect_pos + 1) % N:
                structures.append((a[:], b[:]))

    # Structure 3: Cycle with "reset" letter that maps most to a small set
    for target_size in [1, 2, 3]:
        for offset in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [0] * N
            for i in range(N):
                if i == (offset + N - 1) % N:
                    b[i] = (i + 1) % N
                else:
                    b[i] = i % target_size
            structures.append((a[:], b[:]))

    # Structure 4: Pinwheel-like - a is a cycle, b rotates by a different amount
    for shift in range(1, N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + shift) % N for i in range(N)]
        structures.append((a[:], b[:]))

    # Structure 5: Ping-pong with interleaved cycles
    for step_a in range(1, N):
        for step_b in range(1, N):
            if step_a == step_b:
                continue
            a = [0] * N
            b = [0] * N
            for i in range(N):
                a[i] = (i + step_a) % N
                b[i] = (i + step_b) % N
            structures.append((a[:], b[:]))

    return structures


def _simulated_annealing(a, b, start_time, deadline):
    """
    Simulated Annealing search over the full transition space.
    Uses a custom energy function that estimates reset difficulty by maximizing
    the maximum pairwise distance between states under the current mappings.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)

    if cur_score == 0:
        return None, None, 0

    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score

    # Initial temperature and cooling schedule
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    max_iter = 5000

    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if T < min_T:
            break

        # Generate a neighbor by changing one state's destination in either a or b
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            old = cur_a[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old:
                continue
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            cur_a[state] = old
        else:
            state = random.randint(0, N - 1)
            old = cur_b[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old:
                continue
            cur_b[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            cur_b[state] = old

        # Acceptance criterion: Metropolis
        if new_score > cur_score:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Accept worse solutions with probability exp(-delta/T)
            delta = cur_score - new_score
            if delta > 0:
                if random.random() < pow(2.718281828, -delta / T):
                    cur_score = new_score
                else:
                    # Reject
                    pass

        # Cool down
        T *= cooling

    return best_a, best_b, best_score


def _refine_automaton(a, b, start_time, deadline):
    """
    Local search to refine an automaton, trying to maximize threshold.
    Uses targeted mutations rather than purely random ones.
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
    max_no_improve = 80

    while time.time() < deadline:
        if no_improve > max_no_improve:
            break

        mutation_type = random.choice(['cycle_defect', 'reset_adjust', 'asymmetry', 'random', 'swap_dest'])

        if mutation_type == 'cycle_defect':
            state = random.randint(0, N - 1)
            if cur_a[state] == (state + 1) % N:
                new_dest = random.randint(0, N - 1)
                if new_dest != (state + 1) % N and new_dest != state:
                    old = cur_a[state]
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
            else:
                no_improve += 1

        elif mutation_type == 'reset_adjust':
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

        elif mutation_type == 'asymmetry':
            i, j = random.sample(range(N), 2)
            old_ai, old_aj = cur_a[i], cur_a[j]
            cur_a[i], cur_a[j] = old_aj, old_ai
            new_score = compute_threshold(cur_a, cur_b)
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[i], cur_a[j] = old_ai, old_aj
                no_improve += 1

        elif mutation_type == 'swap_dest':
            # Swap destinations between two states in the same letter
            i, j = random.sample(range(N), 2)
            if random.random() < 0.5:
                old_ai, old_aj = cur_a[i], cur_a[j]
                cur_a[i], cur_a[j] = old_aj, old_ai
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[i], cur_a[j] = old_ai, old_aj
                    no_improve += 1
            else:
                old_bi, old_bj = cur_b[i], cur_b[j]
                cur_b[i], cur_b[j] = old_bj, old_bi
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[i], cur_b[j] = old_bi, old_bj
                    no_improve += 1

        else:  # random
            state = random.randint(0, N - 1)
            if random.random() < 0.5:
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
            else:
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

    # Shuffle to explore different regions
    random.shuffle(structures)

    for a_struct, b_struct in structures:
        if time.time() > deadline - 5.0:
            break

        score = compute_threshold(a_struct, b_struct)
        if score > best_score:
            best_score = score
            best_a = a_struct[:]
            best_b = b_struct[:]

    # Phase 2: Local search refinement from best structured candidate
    if best_a is not None and time.time() < deadline - 3.0:
        refined_a, refined_b, refined_score = _refine_automaton(
            best_a, best_b, start_time, deadline - 3.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b

    # Phase 3: Try a few random starts with refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        num_random_starts = min(5, int(remaining_time / 3))
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

            # Refine if promising
            if score >= best_score - 5 and time.time() < deadline - 3.0:
                refined_a, refined_b, refined_score = _refine_automaton(
                    a_rand, b_rand, start_time, deadline - 3.0
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b

    # Phase 4: Simulated Annealing from best found so far
    if best_a is not None and time.time() < deadline - 2.0:
        sa_a, sa_b, sa_score = _simulated_annealing(
            best_a, best_b, start_time, deadline - 2.0
        )
        if sa_score > best_score:
            best_score = sa_score
            best_a = sa_a
            best_b = sa_b

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        # Simple synchronizing automaton
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [best_a, best_b]


# Optional: Test locally
if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
