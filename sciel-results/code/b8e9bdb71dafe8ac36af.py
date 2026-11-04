import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)

    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4.
    indices = list(range(D))
    supports_D = list(itertools.combinations(indices, 4))
    N_S_D = len(supports_D)

    # Represent supports as bitmasks for fast intersection check
    support_masks_D = []
    for s in supports_D:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks_D.append(mask)
    support_masks_D = np.array(support_masks_D, dtype=np.int64)

    # Precompute conflict matrix for Type D: conflict if intersection size >= 3
    conflict_D = np.zeros((N_S_D, N_S_D), dtype=bool)
    for i in range(N_S_D):
        for j in range(i + 1, N_S_D):
            inter = support_masks_D[i] & support_masks_D[j]
            if bin(inter).count('1') >= 3:
                conflict_D[i, j] = True
                conflict_D[j, i] = True

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set_D = set()
    best_size_D = 0

    # Simulated Annealing for Maximum Independent Set on Type D supports
    def greedy_init(r):
        order = np.arange(N_S_D)
        rng.shuffle(order)
        x = np.zeros(N_S_D, dtype=bool)
        blocked = np.zeros(N_S_D, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict_D[v]
        return x

    def eval_state(x):
        return np.sum(x)

    def get_candidates(x):
        in_set = np.where(x)[0]
        not_in_set = np.where(~x)[0]
        
        if len(in_set) > 0:
            blocked = np.zeros(N_S_D, dtype=bool)
            for v in in_set:
                blocked |= conflict_D[v]
        else:
            blocked = np.zeros(N_S_D, dtype=bool)
        
        add_candidates = [v for v in not_in_set if not blocked[v]]
        remove_candidates = list(in_set)
        
        return add_candidates, remove_candidates

    T0 = 1.0
    T_end = 0.01
    num_iterations = 20000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(50):
        if time.time() > deadline - 25:
            break

        x = greedy_init(restart)
        current_size = eval_state(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            add_candidates, remove_candidates = get_candidates(x)

            if not add_candidates and not remove_candidates:
                break

            move_type = py_rng.random()
            if move_type < 0.5 and add_candidates:
                v = py_rng.choice(add_candidates)
                if x[v]:
                    continue
                in_set = np.where(x)[0]
                if len(in_set) > 0 and np.any(conflict_D[v, in_set]):
                    continue
                x[v] = True
                delta = 1
                new_size = current_size + delta
                if new_size > best_current_size:
                    best_current_size = new_size
                    best_x = x.copy()
                current_size = new_size
            elif move_type < 0.8 and remove_candidates:
                v = py_rng.choice(remove_candidates)
                if not x[v]:
                    continue
                x[v] = False
                delta = -1
                new_size = current_size + delta
                if py_rng.random() < np.exp(delta / T):
                    current_size = new_size
                else:
                    x[v] = True
            else:
                if not add_candidates or not remove_candidates:
                    continue
                v_remove = py_rng.choice(remove_candidates)
                v_add = py_rng.choice(add_candidates)
                if v_remove == v_add:
                    continue
                if not x[v_remove] or x[v_add]:
                    continue
                in_set = np.where(x)[0]
                in_set_no_remove = [v for v in in_set if v != v_remove]
                if len(in_set_no_remove) > 0 and np.any(conflict_D[v_add, in_set_no_remove]):
                    continue
                x[v_remove] = False
                x[v_add] = True
                current_size = current_size

            T *= cooling

        if best_current_size > best_size_D:
            best_size_D = best_current_size
            best_support_set_D = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set_D:
        support = supports_D[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Type E: Vectors with eight non-zero entries +/- 1
    # Norm squared = 8.
    supports_E = list(itertools.combinations(indices, 8))
    N_S_E = len(supports_E)
    
    support_masks_E = []
    for s in supports_E:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks_E.append(mask)
    support_masks_E = np.array(support_masks_E, dtype=np.int64)

    # Conflict between Type E and Type D: intersection >= 3
    # Conflict between Type E and Type E: intersection >= 5 (since we need dot <= 4, max dot is intersection size)
    # Wait, for Type E, max dot is intersection size. We need dot <= 4.
    # So intersection must be <= 4.
    # So conflict if intersection >= 5.
    
    conflict_E_E = np.zeros((N_S_E, N_S_E), dtype=bool)
    for i in range(N_S_E):
        for j in range(i + 1, N_S_E):
            inter = support_masks_E[i] & support_masks_E[j]
            if bin(inter).count('1') >= 5:
                conflict_E_E[i, j] = True
                conflict_E_E[j, i] = True

    # Find MIS on Type E supports, with constraint that they don't conflict with selected Type D supports
    # Conflict with Type D if intersection >= 3.
    valid_E_indices = []
    for i in range(N_S_E):
        is_valid = True
        for d_idx in best_support_set_D:
            inter = support_masks_E[i] & support_masks_D[d_idx]
            if bin(inter).count('1') >= 3:
                is_valid = False
                break
        if is_valid:
            valid_E_indices.append(i)
            
    # Map valid_E_indices to a smaller graph for MIS
    N_valid_E = len(valid_E_indices)
    if N_valid_E > 0:
        conflict_E_valid = np.zeros((N_valid_E, N_valid_E), dtype=bool)
        for i in range(N_valid_E):
            for j in range(i + 1, N_valid_E):
                if conflict_E_E[valid_E_indices[i], valid_E_indices[j]]:
                    conflict_E_valid[i, j] = True
                    conflict_E_valid[j, i] = True

        best_support_set_E = set()
        best_size_E = 0

        def greedy_init_E(r):
            order = np.arange(N_valid_E)
            rng.shuffle(order)
            x = np.zeros(N_valid_E, dtype=bool)
            blocked = np.zeros(N_valid_E, dtype=bool)
            for v in order:
                if not blocked[v] and not x[v]:
                    x[v] = True
                    blocked |= conflict_E_valid[v]
            return x

        def eval_state_E(x):
            return np.sum(x)

        def get_candidates_E(x):
            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]
            if len(in_set) > 0:
                blocked = np.zeros(N_valid_E, dtype=bool)
                for v in in_set:
                    blocked |= conflict_E_valid[v]
            else:
                blocked = np.zeros(N_valid_E, dtype=bool)
            add_candidates = [v for v in not_in_set if not blocked[v]]
            remove_candidates = list(in_set)
            return add_candidates, remove_candidates

        T0_E = 1.0
        T_end_E = 0.01
        num_iterations_E = 10000
        cooling_E = (T_end_E / T0_E) ** (1.0 / num_iterations_E)

        for restart in range(20):
            if time.time() > deadline - 10:
                break
            x = greedy_init_E(restart)
            current_size = eval_state_E(x)
            best_x = x.copy()
            best_current_size = current_size

            T = T0_E
            for it in range(num_iterations_E):
                if time.time() > deadline - 10:
                    break
                add_candidates, remove_candidates = get_candidates_E(x)
                if not add_candidates and not remove_candidates:
                    break
                move_type = py_rng.random()
                if move_type < 0.5 and add_candidates:
                    v = py_rng.choice(add_candidates)
                    if x[v]: continue
                    in_set = np.where(x)[0]
                    if len(in_set) > 0 and np.any(conflict_E_valid[v, in_set]): continue
                    x[v] = True
                    current_size += 1
                    if current_size > best_current_size:
                        best_current_size = current_size
                        best_x = x.copy()
                elif move_type < 0.8 and remove_candidates:
                    v = py_rng.choice(remove_candidates)
                    if not x[v]: continue
                    x[v] = False
                    current_size -= 1
                    if py_rng.random() < np.exp(-1 / T):
                        pass
                    else:
                        x[v] = True
                        current_size += 1
                else:
                    if not add_candidates or not remove_candidates: continue
                    v_remove = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(add_candidates)
                    if v_remove == v_add: continue
                    if not x[v_remove] or x[v_add]: continue
                    in_set = np.where(x)[0]
                    in_set_no_remove = [v for v in in_set if v != v_remove]
                    if len(in_set_no_remove) > 0 and np.any(conflict_E_valid[v_add, in_set_no_remove]): continue
                    x[v_remove] = False
                    x[v_add] = True

                T *= cooling_E

            if best_current_size > best_size_E:
                best_size_E = best_current_size
                best_support_set_E = set(np.where(best_x)[0].tolist())
        
        # Generate Type E vectors
        selected_vectors_E = []
        for v_idx in best_support_set_E:
            s_idx = valid_E_indices[v_idx]
            support = supports_E[s_idx]
            # Select a code of size 20 with min distance 4
            # Greedy selection from 256 patterns
            patterns = list(itertools.product([1, -1], repeat=8))
            selected_patterns = []
            for p in patterns:
                if len(selected_patterns) >= 20:
                    break
                is_ok = True
                for q in selected_patterns:
                    dot = sum(a*b for a,b in zip(p,q))
                    if dot > 4:
                        is_ok = False
                        break
                if is_ok:
                    selected_patterns.append(p)
            
            for signs in selected_patterns:
                v = [0] * D
                for idx, sign in zip(support, signs):
                    v[idx] = sign
                selected_vectors_E.append(v)
        selected_vectors_E = np.array(selected_vectors_E, dtype=np.int64)
    else:
        selected_vectors_E = np.array([], dtype=np.int64).reshape(0, D)

    final_vectors = np.vstack([vectors_A, selected_vectors_D, selected_vectors_E])
    return final_vectors
