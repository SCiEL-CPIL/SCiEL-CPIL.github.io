<<<<<<< SEARCH
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
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
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    # Represent supports as bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Fast Local Search for Maximum Independent Set
    # State: x (bool array), size (int)
    # Blocked: computed as conflict[x].any(axis=0)

    def greedy_init(r):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x, np.sum(x)

    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            # Compute blocked nodes vectorized
            # conflict[x] selects rows where x is True
            # .any(axis=0) checks if any selected node conflicts with column j
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]
            
            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break
                
            move_type = py_rng.random()
            if move_type < 0.6 and len(add_candidates) > 0:
                # Add
                v = py_rng.choice(add_candidates)
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.8 and len(remove_candidates) > 0:
                # Remove
                v = py_rng.choice(remove_candidates)
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    # Accept with probability exp(delta/T)
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True # Revert
            else:
                # Swap
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    v_rem = py_rng.choice(remove_candidates)
                    if v_add != v_rem and not x[v_add] and x[v_rem]:
                        # Check if v_add conflicts with any other selected node
                        # Create a temporary mask without v_rem
                        temp_x = x.copy()
                        temp_x[v_rem] = False
                        if not np.any(conflict[v_add, temp_x]):
                            x[v_rem] = False
                            x[v_add] = True
                            # Size unchanged, but structure changed
                            # We don't update best_size_local here as size is same
                            # But we might find a better state later
                            pass
            
            T *= cooling

        return best_x, best_size_local

    num_restarts = 100
    iterations_per_restart = 5000
    T0 = 1.0
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        x, size = greedy_init(restart)
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())
=======
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
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
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    # Represent supports as bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Complement: compatibility matrix (True means NOT conflicting)
    compat = ~conflict

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_set_cover(max_iterations=2000):
        """
        Vectorized greedy set-cover heuristic for maximum independent set.
        Iteratively selects the support maximizing marginal gain / penalty ratio.
        """
        x = np.zeros(N_S, dtype=bool)
        # Track which nodes are still available (not yet selected or invalidated)
        available = np.ones(N_S, dtype=bool)
        
        for it in range(max_iterations):
            if time.time() > deadline - 15:
                break
            if not available.any():
                break
            
            # For each available node, compute:
            # - gain: number of available nodes it would INVALIDATE (its neighbors in conflict graph)
            # - We want to minimize gain (penalty) while maximizing the set size
            # Actually for MIS: we want nodes with fewest conflicts among available nodes
            # This is the greedy coloring / greedy MIS heuristic
            
            # Number of conflicts each available node has with other available nodes
            if available.sum() > 0:
                # conflict[available, :] gives conflict rows for available nodes
                # Sum over available columns to get conflict count within available set
                conflict_counts = np.zeros(N_S, dtype=np.int32)
                # Vectorized: for each available node, count how many available nodes it conflicts with
                avail_idx = np.where(available)[0]
                if len(avail_idx) > 0:
                    # conflict[avail_idx, :][:, avail_idx] gives submatrix
                    sub_conflict = conflict[np.ix_(avail_idx, avail_idx)]
                    counts = sub_conflict.sum(axis=1)
                    conflict_counts[avail_idx] = counts
                else:
                    break
                
                # Select the node with minimum conflict count (tie-break randomly)
                min_count = conflict_counts[avail_idx].min()
                candidates = avail_idx[conflict_counts[avail_idx] == min_count]
                if len(candidates) > 0:
                    v = candidates[py_rng.randint(0, len(candidates))]
                else:
                    break
                
                x[v] = True
                # Invalidate v and all its neighbors
                to_invalidate = conflict[v] | (np.arange(N_S) == v)
                available[to_invalidate] = False
        
        return x, int(x.sum())

    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            if current_size > 0:
                blocked = conflict[x].any(axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = np.where(x)[0]
            
            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break
                
            move_type = py_rng.random()
            if move_type < 0.6 and len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                if not x[v]:
                    x[v] = True
                    new_size = current_size + 1
                    if new_size > best_size_local:
                        best_size_local = new_size
                        best_x = x.copy()
                    current_size = new_size
            elif move_type < 0.8 and len(remove_candidates) > 0:
                v = py_rng.choice(remove_candidates)
                if x[v]:
                    x[v] = False
                    new_size = current_size - 1
                    if py_rng.random() < np.exp(-1.0 / T):
                        current_size = new_size
                    else:
                        x[v] = True
            else:
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_add = py_rng.choice(add_candidates)
                    v_rem = py_rng.choice(remove_candidates)
                    if v_add != v_rem and not x[v_add] and x[v_rem]:
                        temp_x = x.copy()
                        temp_x[v_rem] = False
                        if not np.any(conflict[v_add, temp_x]):
                            x[v_rem] = False
                            x[v_add] = True
            
            T *= cooling

        return best_x, best_size_local

    # Phase 1: Multiple greedy set-cover runs with different random seeds
    num_greedy = 200
    for restart in range(num_greedy):
        if time.time() > deadline - 40:
            break
        x, size = greedy_set_cover(max_iterations=5000)
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x)[0].tolist())

    # Phase 2: Local search refinement from best solution
    if best_size > 0 and time.time() < deadline - 25:
        x_best = np.zeros(N_S, dtype=bool)
        x_best[list(best_support_set)] = True
        best_x, best_size_local = local_search(x_best, best_size, 8000, 1.0, 0.01)
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Phase 3: Additional greedy + local search cycles
    num_cycles = 50
    for cycle in range(num_cycles):
        if time.time() > deadline - 15:
            break
        x, size = greedy_set_cover(max_iterations=3000)
        if size >= best_size:
            best_x, best_size_local = local_search(x, size, 4000, 1.0, 0.05)
            if best_size_local > best_size:
                best_size = best_size_local
                best_support_set = set(np.where(best_x)[0].tolist())
>>>>>>> REPLACE

