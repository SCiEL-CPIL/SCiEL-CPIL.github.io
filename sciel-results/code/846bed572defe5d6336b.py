import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)

    vectors_v4 = np.array(v4_list, dtype=np.int64)
    vectors_v1 = np.array(v1_list, dtype=np.int64)

    N = len(vectors_v4)

    # Build conflict graph: conflict if |dot| > 2
    # dot product between two weight-4 vectors is even, so conflict iff dot > 2 or dot < -2
    # i.e., |dot| >= 4 (since dot is even)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2) | (dots < -2)
    np.fill_diagonal(conflicts, False)

    # Simulated annealing to find large independent set
    best_set, best_score = _simulated_annealing(conflicts, N, deadline, start_time)

    # Add all weight-1 vectors (always compatible with each other and with any valid weight-4 set)
    # Verify: dot(weight-4, weight-1) is in {-2, 0, 2}, so |dot| <= 2, always compatible
    # dot(weight-1, weight-1) for distinct: 0, compatible
    final_vectors = np.vstack([vectors_v4[best_set], vectors_v1])

    return final_vectors


def _simulated_annealing(conflicts, N, deadline, start_time):
    """Find large independent set using simulated annealing."""
    rng = np.random.default_rng(12345)

    # Initialize with a random independent set using greedy
    def greedy_init():
        alive = np.ones(N, dtype=bool)
        selected = []
        while np.any(alive):
            idx = rng.choice(np.where(alive)[0])
            selected.append(idx)
            alive[conflicts[idx]] = False
        return np.array(selected, dtype=np.int64)

    current_set = greedy_init()
    current_score = len(current_set)
    best_set = current_set.copy()
    best_score = current_score

    # Temperature schedule
    T0 = 5.0
    T_end = 0.01
    total_steps = 200000
    cooling_rate = (T_end / T0) ** (1.0 / max(total_steps, 1))

    in_set = np.zeros(N, dtype=bool)
    in_set[current_set] = True

    # Precompute degrees for faster conflict checking
    # For a given set, the number of conflicts for a vertex v is:
    #   number of neighbors of v that are in the set

    step = 0
    while time.time() < deadline - 5:
        T = T0 * (cooling_rate ** step)
        if T < T_end:
            T = T_end

        # Try a move: add, remove, or swap
        move_type = rng.integers(3)

        if move_type == 0:  # Add a vertex
            # Find vertices not in set that have no conflicts with current set
            # This is expensive, so sample
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0:
                step += 1
                continue
            v = candidates[rng.integers(len(candidates))]
            # Check if v conflicts with any vertex in set
            if not np.any(in_set[conflicts[v]]):
                # Can add without conflict
                in_set[v] = True
                new_score = current_score + 1
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = np.where(in_set)[0]
            else:
                # Would create conflicts; accept with probability exp(-delta/T)
                n_conflicts = np.sum(in_set[conflicts[v]])
                delta = n_conflicts  # penalty
                if rng.random() < np.exp(-delta / T):
                    in_set[v] = True
                    current_score = current_score + 1 - n_conflicts
                    if current_score > best_score:
                        best_score = current_score
                        best_set = np.where(in_set)[0]

        elif move_type == 1:  # Remove a vertex
            if current_score == 0:
                step += 1
                continue
            v = current_set[rng.integers(len(current_set))]
            in_set[v] = False
            current_score -= 1
            # Update current_set
            current_set = np.where(in_set)[0]

        else:  # Swap: remove one, try to add another
            if current_score == 0:
                step += 1
                continue
            v_remove = current_set[rng.integers(len(current_set))]
            in_set[v_remove] = False
            current_score -= 1

            # Try to add a vertex that was previously conflicting
            candidates = np.where(~in_set)[0]
            if len(candidates) > 0:
                v_add = candidates[rng.integers(len(candidates))]
                if not np.any(in_set[conflicts[v_add]]):
                    in_set[v_add] = True
                    current_score += 1
                else:
                    n_conflicts = np.sum(in_set[conflicts[v_add]])
                    delta = n_conflicts - 1  # net change in score
                    if rng.random() < np.exp(-delta / T):
                        in_set[v_add] = True
                        current_score += 1 - n_conflicts

            current_set = np.where(in_set)[0]
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()

        step += 1

        # Periodic check
        if step % 10000 == 0:
            if time.time() > deadline - 10:
                break

    return best_set, best_score
