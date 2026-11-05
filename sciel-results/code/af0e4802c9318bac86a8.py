import itertools
import random
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as bitmask masks.
    # A transversal = 3 disjoint pairs on 6 points -> 8 transversal triples.
    transversals = []
    seen_masks = set()
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        matchings = set()
        for perm in itertools.permutations(range(6)):
            p1 = tuple(sorted((sp[perm[0]], sp[perm[1]])))
            p2 = tuple(sorted((sp[perm[2]], sp[perm[3]])))
            p3 = tuple(sorted((sp[perm[4]], sp[perm[5]])))
            matchings.add(tuple(sorted((p1, p2, p3))))
        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            mask = 0
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                mask |= 1 << triple_to_idx[tuple(sorted((x, y, z)))]
            if mask not in seen_masks:
                seen_masks.add(mask)
                transversals.append(mask)
    num_trans = len(transversals)

    # Precompute the list of triple indices for each transversal
    trans_triples = []
    for tm in transversals:
        indices = []
        m = tm
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        trans_triples.append(indices)

    # Build triple_to_trans_list: for each triple, which transversals it belongs to
    triple_to_trans = [[] for _ in range(num_triples)]
    for ti in range(num_trans):
        for t_idx in trans_triples[ti]:
            triple_to_trans[t_idx].append(ti)

    best_state = None
    best_count = 0

    def optimize_v2(start_state, start_counts, start_conf, start_count):
        nonlocal best_state, best_count

        cur_state = np.array(start_state, dtype=bool)
        cur_counts = np.array(start_counts, dtype=np.int32)
        cur_conf = start_conf
        cur_count = start_count

        # Precompute triple_to_trans as numpy arrays for faster iteration
        # triple_to_trans is list of lists; convert to list of arrays
        trans_arrays = [np.array(t, dtype=np.int32) for t in triple_to_trans]
        trans_triples_arrays = [np.array(t, dtype=np.int32) for t in trans_triples]

        # Phase 1: Resolve conflicts
        while cur_conf > 0:
            if time.time() > deadline:
                break
            # Find a conflict
            conflict_ti = -1
            for ti in range(num_trans):
                if cur_counts[ti] == 8:
                    conflict_ti = ti
                    break

            if conflict_ti == -1:
                break

            # Pick a triple to remove that breaks the most conflicts
            best_t = -1
            max_breaks = 0
            for t_idx in trans_triples_arrays[conflict_ti]:
                if cur_state[t_idx]:
                    breaks = np.sum(cur_counts[trans_arrays[t_idx]] == 8)
                    if breaks > max_breaks:
                        max_breaks = breaks
                        best_t = t_idx

            if best_t == -1:
                best_t = random.choice(trans_triples_arrays[conflict_ti])

            # Remove best_t
            cur_state[best_t] = False
            cur_count -= 1
            for ti in trans_arrays[best_t]:
                old = cur_counts[ti]
                cur_counts[ti] = old - 1
                if old == 8:
                    cur_conf -= 1

            if cur_conf == 0:
                if cur_count > best_count:
                    best_count = cur_count
                    best_state = cur_state.copy()

        # Phase 2: Add triples greedily
        progress = True
        while progress:
            if time.time() > deadline:
                break
            progress = False
            indices = np.arange(num_triples)
            np.random.shuffle(indices)
            for i in indices:
                if not cur_state[i]:
                    # Check if safe to add (doesn't complete any transversal to 8)
                    if np.any(cur_counts[trans_arrays[i]] == 7):
                        continue
                    cur_state[i] = True
                    cur_count += 1
                    for ti in trans_arrays[i]:
                        cur_counts[ti] += 1
                    progress = True
                    if cur_count > best_count:
                        best_count = cur_count
                        best_state = cur_state.copy()

        # Phase 3: WalkSAT-style local search
        if time.time() < deadline - 5.0:
            # Convert to numpy for speed
            present_mask = cur_state
            absent_mask = ~cur_state
            
            # Precompute for faster access
            num_present = int(np.sum(present_mask))
            num_absent = int(np.sum(absent_mask))
            
            if num_present == 0 or num_absent == 0:
                return cur_state, cur_conf, cur_count

            # WalkSAT parameters
            walk_iters = 5000
            for it in range(walk_iters):
                if time.time() > deadline:
                    break
                
                # If no conflicts, try to add triples
                if cur_conf == 0:
                    # Find all safe absent triples
                    safe_absent = []
                    for i in range(num_triples):
                        if not cur_state[i] and not np.any(cur_counts[trans_arrays[i]] == 7):
                            safe_absent.append(i)
                    
                    if safe_absent:
                        # Add a random safe triple
                        add_i = random.choice(safe_absent)
                        cur_state[add_i] = True
                        cur_count += 1
                        for ti in trans_arrays[add_i]:
                            cur_counts[ti] += 1
                        if cur_count > best_count:
                            best_count = cur_count
                            best_state = cur_state.copy()
                        continue
                
                # If conflicts exist, pick a random conflict
                conflict_indices = [ti for ti in range(num_trans) if cur_counts[ti] == 8]
                if not conflict_indices:
                    continue
                conflict_ti = random.choice(conflict_indices)
                
                # Pick a triple from this conflict to remove
                candidates = [t for t in trans_triples_arrays[conflict_ti] if cur_state[t]]
                if not candidates:
                    continue
                
                # Score each candidate by how many conflicts it resolves
                best_t = -1
                best_score = -1
                for t in candidates:
                    score = int(np.sum(cur_counts[trans_arrays[t]] == 8))
                    # Add small random perturbation for diversity
                    score += random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_t = t
                
                if best_t == -1:
                    best_t = random.choice(candidates)
                
                # Remove best_t
                cur_state[best_t] = False
                cur_count -= 1
                for ti in trans_arrays[best_t]:
                    old = cur_counts[ti]
                    cur_counts[ti] = old - 1
                    if old == 8:
                        cur_conf -= 1
                
                # Try to add a random absent triple that doesn't create new conflicts
                absent_candidates = [i for i in range(num_triples) if not cur_state[i]]
                if absent_candidates:
                    # Shuffle and try a few
                    random.shuffle(absent_candidates)
                    for add_i in absent_candidates[:20]:
                        if not np.any(cur_counts[trans_arrays[add_i]] == 7):
                            cur_state[add_i] = True
                            cur_count += 1
                            for ti in trans_arrays[add_i]:
                                cur_counts[ti] += 1
                            if cur_count > best_count:
                                best_count = cur_count
                                best_state = cur_state.copy()
                            break

        return cur_state.tolist(), cur_conf, cur_count

    # Initialize numpy random seed for reproducibility within runs
    np.random.seed(42)
    
    # Initialization: Start with all triples present
    init_state = [True] * num_triples
    init_counts = [8] * num_trans
    init_conf = num_trans
    init_count = num_triples

    # Run optimization from full set
    optimize_v2(init_state, init_counts, init_conf, init_count)

    # Restart with perturbations
    for restart in range(25):
        if time.time() > deadline - 5.0:
            break

        if best_state is None:
            break

        # best_state is now a numpy array
        perturbed_state = best_state.copy()
        perturbed_counts = np.zeros(num_trans, dtype=np.int32)
        perturbed_conf = 0
        perturbed_count = int(np.sum(perturbed_state))

        # Recalculate counts for perturbed state
        for i in range(num_triples):
            if perturbed_state[i]:
                for ti in triple_to_trans[i]:
                    perturbed_counts[ti] += 1

        perturbed_conf = int(np.sum(perturbed_counts == 8))

        # Remove some random triples to create space
        num_removals = random.randint(15, 60)
        present_indices = np.where(perturbed_state)[0]
        np.random.shuffle(present_indices)
        for i in present_indices[:num_removals]:
            perturbed_state[i] = False
            perturbed_count -= 1
            for ti in triple_to_trans[i]:
                perturbed_counts[ti] -= 1

        # Recalculate conflicts after removal
        perturbed_conf = int(np.sum(perturbed_counts == 8))

        # Optimize
        optimize_v2(perturbed_state.tolist(), perturbed_counts.tolist(), perturbed_conf, perturbed_count)

    if best_state is None:
        return []

    # best_state is a numpy array
    result = [list(all_triples[i]) for i in range(num_triples) if best_state[i]]
    return result
