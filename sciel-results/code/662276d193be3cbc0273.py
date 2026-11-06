import itertools
import random
import time
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560

    # Build the forbidden transversals as lists of triple indices.
    # A transversal = 3 disjoint pairs on 6 points -> 8 transversal triples.
    transversals = []
    seen = set()
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
            triple_indices = []
            for (x, y, z) in ((a, c, e), (a, c, f), (a, d, e), (a, d, f),
                              (b, c, e), (b, c, f), (b, d, e), (b, d, f)):
                triple_indices.append(triple_to_idx[tuple(sorted((x, y, z)))])
            key = tuple(sorted(triple_indices))
            if key not in seen:
                seen.add(key)
                transversals.append(triple_indices)
    num_trans = len(transversals)

    # For each triple, the list of transversals it participates in.
    triple_trans = [[] for _ in range(num_triples)]
    for ti, tlist in enumerate(transversals):
        for idx in tlist:
            triple_trans[idx].append(ti)

    # Precompute: for each transversal, a numpy array of its 8 triple indices
    trans_arrays = [np.array(t, dtype=np.int32) for t in transversals]

    # Precompute: for each triple, which transversals it belongs to (as array)
    triple_trans_arrays = [np.array(t, dtype=np.int32) for t in triple_trans]

    def greedy_init():
        sol = np.zeros(num_triples, dtype=bool)
        counts = np.zeros(num_trans, dtype=np.int32)
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            # Check if adding this triple would complete any transversal
            tts = triple_trans_arrays[idx]
            if len(tts) > 0:
                if np.any(counts[tts] == 7):
                    continue
            sol[idx] = True
            if len(tts) > 0:
                counts[tts] += 1
        return sol, counts, int(sol.sum())

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            tts = triple_trans_arrays[idx]
            if len(tts) > 0:
                if np.any(counts[tts] == 7):
                    new_candidates.append(idx)
                    continue
            sol[idx] = True
            if len(tts) > 0:
                counts[tts] += 1
            added += 1
        return added, new_candidates

    best_solution = None
    best_count = -1

    for restart in range(200):
        if time.time() > deadline - 1.0:
            break

        sol, counts, current_count = greedy_init()
        
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol.copy()
            best_counts = counts.copy()

        stagnation = 0
        max_stagnation = 50
        
        while time.time() < deadline - 0.5:
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol.copy()
                    best_counts = counts.copy()
                continue

            # Find tight configurations (count == 7)
            tight_idx = np.where(counts == 7)[0]
            
            if len(tight_idx) > 0:
                # Pick a random tight transversal, then a random triple in it
                ci = int(random.choice(tight_idx))
                t_arr = trans_arrays[ci]
                # Find triples in this transversal that are currently selected
                selected_in_trans = t_arr[sol[t_arr]]
                if len(selected_in_trans) > 0:
                    kick_idx = int(random.choice(selected_in_trans))
                else:
                    kick_idx = int(random.choice(t_arr))
            else:
                active = np.where(sol)[0]
                if len(active) == 0:
                    break
                kick_idx = int(random.choice(active))
            
            # Kick: remove the triple
            sol[kick_idx] = False
            tts = triple_trans_arrays[kick_idx]
            if len(tts) > 0:
                counts[tts] -= 1
            current_count -= 1
            candidates.append(kick_idx)
            
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol.copy()
                    best_counts = counts.copy()
            else:
                stagnation += 1
                if stagnation > max_stagnation:
                    break

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init()

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution.copy()
    final_counts = best_counts.copy()
    # Remove any remaining conflicts
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            t_arr = trans_arrays[ci]
            selected = t_arr[final_sol[t_arr]]
            if len(selected) == 0:
                break
            t_idx = int(selected[0])
            final_sol[t_idx] = False
            tts = triple_trans_arrays[t_idx]
            if len(tts) > 0:
                final_counts[tts] -= 1
            final_counts[ci] -= 1

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
