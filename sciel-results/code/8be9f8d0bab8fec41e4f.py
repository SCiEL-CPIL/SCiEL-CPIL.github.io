"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill

    # --- Generate candidate vectors (all norm^2 = 4) ---
    # W4: weight 4, entries in {-1,0,1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    # W2: weight 2, entries in {-1,0,1}
    v2_list = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_list.append(v)
    # W1: weight 1, entries in {-2,0,2}
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)

    all_vectors = np.array(v4_list + v2_list + v1_list, dtype=np.int64)
    N = len(all_vectors)

    # --- Build conflict graph ---
    # Conflict if dot > 2 (i.e., squared distance < 4 = max norm^2)
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    # --- Fast local search: greedy init + 1-swap hill climb + perturbation ---
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_init(seed_order=None):
        """Greedy independent set: pick vertices in given order (or min-degree)."""
        in_set = np.zeros(N, dtype=bool)
        if seed_order is not None:
            order = seed_order
        else:
            # Min-degree greedy with random tie-breaking
            order = np.argsort(neighbor_counts + rng.random(N) * 0.001)

        for v in order:
            if in_set[v]:
                continue
            # Check if v conflicts with any in-set vertex
            if np.any(conflicts[v] & in_set):
                continue
            in_set[v] = True

        return in_set

    def hill_climb(in_set, max_iters=2000):
        """1-swap hill climbing: try removing one and adding one to increase size."""
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        in_set_indices = np.where(in_set)[0]
        not_in_set = np.where(~in_set)[0]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            iters += 1
            improved = False
            if len(in_set_indices) == 0:
                break

            # For each in-set vertex, try swapping with a non-conflicting out-set vertex
            # Strategy: try random pairs
            for _ in range(50):
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                # Find out-set vertices that don't conflict with any in-set vertex
                # except possibly v_remove
                conflicts_with_in = conflicts[v_remove] | conflicts[not_in_set]  # not quite
                # Better: for each candidate v_add, check if it conflicts with in_set minus v_remove
                v_add = not_in_set[rng.integers(len(not_in_set))]
                # Check conflicts of v_add with in_set (excluding v_remove)
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                if len(other_conflicts) == 0:
                    # Clean swap: remove v_remove, add v_add
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    in_set_indices = np.where(in_set)[0]
                    not_in_set = np.where(~in_set)[0]
                    current_size += 1
                    improved = True
                    break
            # Also try pure adds
            if not improved:
                for _ in range(20):
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    if not np.any(conflicts[v_add] & in_set):
                        in_set[v_add] = True
                        in_set_indices = np.where(in_set)[0]
                        not_in_set = np.where(~in_set)[0]
                        current_size += 1
                        improved = True
                        break

        return in_set, current_size

    def perturb(in_set, strength=3):
        """Random perturbation: remove 'strength' random in-set vertices."""
        in_set = in_set.copy()
        in_set_indices = np.where(in_set)[0]
        if len(in_set_indices) == 0:
            return in_set
        n_remove = min(strength, len(in_set_indices))
        to_remove = rng.choice(in_set_indices, size=n_remove, replace=False)
        in_set[to_remove] = False
        return in_set

    # --- Main search loop ---
    n_restarts = 0
    max_restarts = 5000

    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1

        # Greedy init with random tie-breaking
        if n_restarts % 3 == 0:
            # Min-degree greedy
            in_set = greedy_init()
        else:
            # Random order greedy
            in_set = greedy_init(seed_order=rng.permutation(N))

        current_size = int(in_set.sum())

        # Hill climb
        in_set, current_size = hill_climb(in_set, max_iters=500)

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Perturbation + re-climb for stuck states
        if n_restarts % 2 == 0:
            perturbed = perturb(best_in_set, strength=rng.integers(2, 6))
            perturbed, pert_size = hill_climb(perturbed, max_iters=300)
            if pert_size > best_size:
                best_size = pert_size
                best_in_set = perturbed.copy()

    # --- Final polish: try to add any non-conflicting vertices ---
    for v in range(N):
        if not best_in_set[v] and not np.any(conflicts[v] & best_in_set):
            best_in_set[v] = True

    best_size = int(best_in_set.sum())

    # --- Build result ---
    final_indices = np.where(best_in_set)[0]
    result = all_vectors[final_indices]

    return result

# EVOLVE-BLOCK-END
