import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate Candidates
    # Type A: 4 non-zeros, each +/-1. Norm^2 = 4.
    # Type B: 1 non-zero, +/-2. Norm^2 = 4.
    
    candidates = []
    # Type A
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    # Type B
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    if N == 0:
        return np.array([], dtype=np.int64)

    # 2. Compute Conflict Matrix
    # Conflict if dot product > 2 (since norm^2=4, threshold is 4/2=2)
    dots = candidates @ candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for fast access
    # neighbors[i] is an array of indices j such that conflicts[i,j] is True
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Global state for best solution
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Helper to check validity and update best
    def update_best(in_set):
        nonlocal best_size, best_in_set
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
    # Greedy Initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # degrees is the current degree in the induced subgraph of 'alive'
        degrees = neighbor_counts.copy()
        
        # To speed up, we can just pick random from 0-conflict in the 'in_set' logic?
        # No, standard greedy IS:
        # 1. Find all vertices with degree 0 in the alive subgraph? No, that's not how it works.
        # Greedy IS:
        # While alive is not empty:
        #   Pick vertex v with min degree in alive subgraph.
        #   Add v to in_set.
        #   Remove v and its neighbors from alive.
        
        # Efficient implementation:
        # We don't need a heap if we just pick min degree among alive.
        # But updating degrees is O(N^2) per step if done naively.
        # Given N=5302, N^2 is 28M. 5302 steps * 28M is too slow.
        
        # Better Greedy:
        # Maintain a list of available vertices (0 conflicts with current set).
        # Initially all are available.
        # Pick one, add to set, remove its neighbors from available.
        
        available = np.ones(N, dtype=bool)
        in_set = np.zeros(N, dtype=bool)
        
        # To make it "min-degree" like, we can weight the choice.
        # But simple random choice from available is fast.
        # Let's try a hybrid: pick from available, but prefer those with fewer neighbors in the *original* graph?
        # Actually, just random is good with restarts.
        
        while True:
            if time.time() > deadline - 10:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Pick a random available vertex
            # To improve, we could sort by neighbor_counts[avail_indices] and pick min?
            # That's O(N log N) per step. 5302 steps * 5302 log 5302 ~ 300M ops. Might be slow.
            # Let's stick to random pick for speed, relying on multiple restarts.
            v = avail_indices[rng.randint(len(avail_indices))]
            
            in_set[v] = True
            # Remove v and its neighbors from available
            available[v] = False
            available[neighbors[v]] = False
            
        return in_set

    # Local Search
    def local_search(in_set_init, seed, time_limit):
        rng = random.Random(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # conflict_count[i] = number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + time_limit
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < ls_deadline:
            improved = False
            
            # Phase 1: Add vertices with 0 conflicts
            # Find vertices not in set with 0 conflicts
            not_in_set = ~in_set
            # conflict_count[not_in_set] == 0
            zero_conf = np.where(not_in_set & (conflict_count == 0))[0]
            
            while len(zero_conf) > 0:
                if time.time() > ls_deadline:
                    break
                v = zero_conf[rng.randint(len(zero_conf))]
                in_set[v] = True
                current_size += 1
                conflict_count[neighbors[v]] += 1
                improved = True
                # Recalculate zero_conf
                zero_conf = np.where(not_in_set & (conflict_count == 0))[0]
                
            if current_size > best_size:
                update_best(in_set)
                
            if improved:
                no_improve_count = 0
                continue
                
            # Phase 2: 1-for-1 Swaps
            # Find u in in_set, v not in in_set such that:
            # 1. conflicts[v, u] is True
            # 2. conflict_count[v] == 1 (so v only conflicts with u)
            
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) > 0:
                # Precompute valid v for each u? No, too expensive.
                # Randomly try swaps.
                swaps_tried = 0
                max_swaps = 500
                while swaps_tried < max_swaps and time.time() < ls_deadline:
                    swaps_tried += 1
                    u = in_set_indices[rng.randint(len(in_set_indices))]
                    # Find v not in set that conflicts with u
                    # v must be in neighbors[u] and not in in_set
                    v_candidates = neighbors[u]
                    # Filter: not in in_set
                    v_candidates = v_candidates[~in_set[v_candidates]]
                    if len(v_candidates) == 0:
                        continue
                    v = v_candidates[rng.randint(len(v_candidates))]
                    
                    # Check if v only conflicts with u
                    if conflict_count[v] == 1:
                        # Perform swap
                        in_set[u] = False
                        in_set[v] = True
                        # Update conflict counts
                        # u removed: neighbors of u lose 1
                        conflict_count[neighbors[u]] -= 1
                        # v added: neighbors of v gain 1
                        conflict_count[neighbors[v]] += 1
                        # Note: u and v are neighbors, so:
                        # conflict_count[u] (now not in set, but count is for "how many in_set neighbors")
                        # Wait, conflict_count[i] is defined as "number of neighbors of i in in_set".
                        # When we remove u, for any w in neighbors[u], conflict_count[w] decreases by 1.
                        # When we add v, for any w in neighbors[v], conflict_count[w] increases by 1.
                        # Since u and v are neighbors:
                        # conflict_count[u] was 1 (neighbor v? No, v was not in set).
                        # Let's trace carefully.
                        # Old state: u in set, v not in set.
                        # conflict_count[v] = 1 (only u conflicts with v).
                        # Remove u:
                        #   For all w in neighbors[u]: conflict_count[w] -= 1.
                        #   This includes v. So conflict_count[v] becomes 0.
                        #   This includes other neighbors of u.
                        # Add v:
                        #   For all w in neighbors[v]: conflict_count[w] += 1.
                        #   This includes u. So conflict_count[u] becomes 1 (was 0, since u was removed from set? No, u is no longer in set, but conflict_count tracks neighbors IN SET).
                        #   Wait, conflict_count is only meaningful for vertices NOT in set? Or all?
                        #   It is useful for all.
                        #   When u is removed, it is no longer in set. So it shouldn't contribute to anyone's count.
                        #   When v is added, it is in set. So it contributes to its neighbors' counts.
                        
                        improved = True
                        no_improve_count = 0
                        # Break to restart phases? Or continue?
                        # Restarting phases is safer to re-evaluate 0-conflict adds.
                        break

            if improved:
                continue
                
            # Phase 3: 2-for-1 Swaps
            # Remove u, add v1, v2.
            # v1, v2 not in set.
            # v1, v2 only conflict with u.
            # v1, v2 do not conflict with each other.
            
            if len(in_set_indices) > 0:
                swaps_tried = 0
                max_swaps = 200
                while swaps_tried < max_swaps and time.time() < ls_deadline:
                    swaps_tried += 1
                    u = in_set_indices[rng.randint(len(in_set_indices))]
                    # Find v1, v2 in neighbors[u] not in set
                    v_candidates = neighbors[u]
                    v_candidates = v_candidates[~in_set[v_candidates]]
                    if len(v_candidates) < 2:
                        continue
                    
                    # Pick two distinct
                    idx1 = rng.randint(len(v_candidates))
                    idx2 = rng.randint(len(v_candidates))
                    while idx2 == idx1:
                        idx2 = rng.randint(len(v_candidates))
                        
                    v1 = v_candidates[idx1]
                    v2 = v_candidates[idx2]
                    
                    # Check conditions
                    if (conflict_count[v1] == 1 and conflict_count[v2] == 1 and not conflicts[v1, v2]):
                        # Perform swap
                        in_set[u] = False
                        in_set[v1] = True
                        in_set[v2] = True
                        
                        # Update counts
                        # Remove u
                        conflict_count[neighbors[u]] -= 1
                        # Add v1
                        conflict_count[neighbors[v1]] += 1
                        # Add v2
                        conflict_count[neighbors[v2]] += 1
                        
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        break

            if improved:
                continue
            
            # Phase 4: Kick (Perturbation)
            no_improve_count += 1
            if no_improve_count > 20:
                # Remove 3-5 random vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) >= 3:
                    k = rng.randint(3, min(5, len(in_set_indices)))
                    to_remove = in_set_indices[rng.sample(len(in_set_indices), k)]
                    for u in to_remove:
                        in_set[u] = False
                        conflict_count[neighbors[u]] -= 1
                    current_size -= k
                    no_improve_count = 0
                else:
                    # Set is too small, restart local search?
                    break
                    
        return in_set

    # Main Loop
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        seed = restart * 7919 + 42
        
        # Initialization
        in_set = greedy_init(seed)
        update_best(in_set)
        
        # Local Search
        # Allocate time based on remaining time
        remaining = deadline - 5 - time.time()
        if remaining < 1.0:
            break
            
        # Give local search a chunk of time, e.g., 2-5 seconds
        ls_time = min(remaining * 0.2, 5.0)
        if ls_time < 0.5:
            ls_time = 0.5
            
        result_set = local_search(in_set, seed, ls_time)
        update_best(result_set)
        
        # If we found a very good set, we might want to polish it more?
        # The loop handles this by restarting, but if we are close to deadline, 
        # we should do a final polish on the best set found so far.
        
    # Final Polish on best_in_set
    if time.time() < deadline - 2:
        # Do a quick local search on the best set found
        final_set = local_search(best_in_set, 99999, min(2.0, deadline - 2 - time.time()))
        update_best(final_set)

    final_indices = np.where(best_in_set)[0]
    return candidates[final_indices]
