"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.linalg import eigh

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree(seed_offset=0):
        """Greedy independent set with min-degree heuristic, randomized tie-breaking."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        local_rng = np.random.default_rng(42 + seed_offset)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]

            if len(cands) > 1:
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                    else:
                        scores[idx] = 0
                noise = local_rng.random(len(cands)) * 0.1
                scores += noise
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def simulated_annealing(init_set, max_time=10.0):
        x = init_set.copy()
        size = int(x.sum())
        
        # Initial conflict count
        conflict_count = 0
        for i in np.where(x)[0]:
            conflict_count += np.sum(x[neighbors[i]])
        conflict_count //= 2
        
        current_energy = conflict_count - size
        best_energy = current_energy
        best_x = x.copy()
        
        T = 5.0
        min_T = 0.01
        cooling = 0.999
        ls_deadline = time.time() + max_time
        local_rng = np.random.default_rng(42)
        stuck_counter = 0
        max_stuck = 500
        
        while time.time() < ls_deadline:
            r = local_rng.random()
            move_executed = False
            
            if r < 0.4:
                # Add or Remove
                v = local_rng.integers(N4)
                if not x[v]:
                    new_conflicts = int(np.sum(x[neighbors[v]]))
                    delta_E = new_conflicts - 1
                    if local_rng.random() < np.exp(-delta_E / T):
                        x[v] = True
                        size += 1
                        conflict_count += new_conflicts
                        current_energy = conflict_count - size
                        move_executed = True
                else:
                    removed_conflicts = int(np.sum(x[neighbors[v]]))
                    delta_E = -removed_conflicts + 1
                    if local_rng.random() < np.exp(-delta_E / T):
                        x[v] = False
                        size -= 1
                        conflict_count -= removed_conflicts
                        current_energy = conflict_count - size
                        move_executed = True
                        
            elif r < 0.7:
                # Swap
                u = local_rng.integers(N4)
                v = local_rng.integers(N4)
                if u != v:
                    if x[u] and not x[v]:
                        removed_edges = int(np.sum(x[neighbors[u]]))
                        added_edges = int(np.sum(x[neighbors[v]]))
                        if conflicts[u, v]:
                            added_edges -= 1
                        delta_C = added_edges - removed_edges
                        delta_E = delta_C
                        if local_rng.random() < np.exp(-delta_E / T):
                            x[u] = False
                            x[v] = True
                            conflict_count += delta_C
                            current_energy = conflict_count - size
                            move_executed = True
                    elif x[v] and not x[u]:
                        removed_edges = int(np.sum(x[neighbors[v]]))
                        added_edges = int(np.sum(x[neighbors[u]]))
                        if conflicts[u, v]:
                            added_edges -= 1
                        delta_C = added_edges - removed_edges
                        delta_E = delta_C
                        if local_rng.random() < np.exp(-delta_E / T):
                            x[v] = False
                            x[u] = True
                            conflict_count += delta_C
                            current_energy = conflict_count - size
                            move_executed = True

            else:
                # 2-for-1
                u1 = local_rng.integers(N4)
                u2 = local_rng.integers(N4)
                v = local_rng.integers(N4)
                
                if u1 != u2 and u1 != v and u2 != v:
                    if x[u1] and x[u2] and not x[v]:
                        deg_u1 = int(np.sum(x[neighbors[u1]]))
                        deg_u2 = int(np.sum(x[neighbors[u2]]))
                        deg_v = int(np.sum(x[neighbors[v]]))
                        
                        R = deg_u1 + deg_u2
                        if conflicts[u1, u2]:
                            R -= 1
                            
                        A = deg_v
                        if conflicts[v, u1]:
                            A -= 1
                        if conflicts[v, u2]:
                            A -= 1
                            
                        delta_C = A - R
                        delta_S = -1
                        delta_E = delta_C - delta_S
                        
                        if local_rng.random() < np.exp(-delta_E / T):
                            x[u1] = False
                            x[u2] = False
                            x[v] = True
                            size -= 1
                            conflict_count += delta_C
                            current_energy = conflict_count - size
                            move_executed = True

            if move_executed:
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            T *= cooling
            if T < min_T:
                T = min_T
                
            if stuck_counter > max_stuck:
                n_flip = max(1, int(0.05 * size)) if size > 0 else 1
                flip_indices = local_rng.integers(N4, size=n_flip)
                for f in flip_indices:
                    x[f] = not x[f]
                
                size = int(x.sum())
                conflict_count = 0
                for i in np.where(x)[0]:
                    conflict_count += np.sum(x[neighbors[i]])
                conflict_count //= 2
                current_energy = conflict_count - size
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                
                stuck_counter = 0
                T = max(T, 2.0)

        # Final polish
        final_set = best_x.copy()
        final_size = int(final_set.sum())
        changed = True
        while changed:
            changed = False
            for i in range(N4):
                if not final_set[i]:
                    if np.sum(final_set[neighbors[i]]) == 0:
                        final_set[i] = True
                        final_size += 1
                        changed = True
                        
        return final_set, final_size

    # Phase 1: Greedy min-degree with multiple seeds
    greedy_inits = []
    for seed in range(20):
        if time.time() > deadline - 40:
            break
        in_set = greedy_min_degree(seed)
        greedy_inits.append(in_set)

    # Phase 2: Simulated Annealing on each initial solution
    for init in greedy_inits:
        if time.time() > deadline - 10:
            break
        init_size = int(init.sum())
        if init_size > best_size:
            best_size = init_size
            best_in_set = init.copy()

        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        sa_time = min(remaining * 0.3, 6.0)
        if sa_time < 1.5:
            sa_time = 1.5

        sa_result, sa_size = simulated_annealing(init, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # Phase 3: Additional SA rounds on best solution with perturbations
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        for perturb_round in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb: remove some random vertices from best
            perturbed = best_in_set.copy()
            in_indices = np.where(perturbed)[0]
            if len(in_indices) > 10:
                n_rem = max(3, int(0.15 * len(in_indices)))
                rem_idx = rng.choice(len(in_indices), size=n_rem, replace=False)
                for ri in rem_idx:
                    perturbed[in_indices[ri]] = False

            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            sa_time = min(remaining * 0.4, 8.0)
            if sa_time < 2:
                sa_time = 2.0

            sa_result, sa_size = simulated_annealing(perturbed, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # Final polish is included in simulated_annealing, but we can do one more quick pass
    # The SA function already does a greedy polish at the end.

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
