import itertools
import random
import time


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

    # Precompute a "density" score for each triple based on how many
    # transversals it shares with other triples. Triples that are in many
    # transversals are more "constrained" - we want to pick them carefully.
    # Actually, for initialization, we want triples that participate in fewer
    # transversals first (less constrained), then add more constrained ones.
    triple_constraint = [len(triple_trans[i]) for i in range(num_triples)]

    def greedy_init(mode=0):
        sol = [False] * num_triples
        counts = [0] * num_trans
        candidates = list(range(num_triples))
        
        if mode == 0:
            # Random order
            random.shuffle(candidates)
        elif mode == 1:
            # Sort by constraint (fewest first)
            candidates.sort(key=lambda i: triple_constraint[i])
            random.shuffle(candidates)
            # Re-shuffle within groups of same constraint
            for i in range(len(candidates) - 1):
                if triple_constraint[candidates[i]] == triple_constraint[candidates[i+1]]:
                    # small random swap
                    if random.random() < 0.5:
                        candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
        elif mode == 2:
            # Sort by constraint (most first) - counter-intuitive but sometimes works
            candidates.sort(key=lambda i: -triple_constraint[i])
            random.shuffle(candidates)
        else:
            # Weighted random: prefer lower constraint
            weights = [1.0 / (1.0 + triple_constraint[i] * 0.1) for i in range(num_triples)]
            total = sum(weights)
            candidates = []
            remaining = list(range(num_triples))
            rem_weights = list(weights)
            for _ in range(num_triples):
                r = random.random() * sum(rem_weights)
                cum = 0
                for j, w in enumerate(rem_weights):
                    cum += w
                    if r <= cum:
                        candidates.append(remaining[j])
                        remaining.pop(j)
                        rem_weights.pop(j)
                        break
        
        for idx in candidates:
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
        return sol, counts, sum(sol)

    def try_add_candidates(sol, counts, candidates):
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            safe = True
            for ci in triple_trans[idx]:
                if counts[ci] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for ci in triple_trans[idx]:
                    counts[ci] += 1
                added += 1
            else:
                new_candidates.append(idx)
        return added, new_candidates

    def local_search(sol, counts, current_count, max_iters=200):
        """Perform local search with kick-and-retry strategy."""
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        
        for _ in range(max_iters):
            if time.time() > deadline - 1.0:
                break
            
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                continue
            
            # No progress - try a kick
            # Find tight configurations
            tight_triples = []
            for ci, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in transversals[ci]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
                            break  # one per tight transversal is enough
            
            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                active = [i for i in range(num_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)
            
            sol[kick_idx] = False
            for ci in triple_trans[kick_idx]:
                counts[ci] -= 1
            current_count -= 1
            candidates.append(kick_idx)
            
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
        
        return current_count

    best_solution = None
    best_count = -1

    # Multiple restarts with different initialization strategies
    num_restarts = 120
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Vary the initialization strategy
        mode = restart % 4
        sol, counts, current_count = greedy_init(mode)
        
        # Run local search
        current_count = local_search(sol, counts, current_count, max_iters=100)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            best_counts = counts[:]

    # Additional focused restarts around the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            
            sol = best_solution[:]
            counts = best_counts[:]
            current_count = best_count
            
            # Perturb: remove 1-3 random triples and re-optimize
            num_remove = random.randint(1, 3)
            active = [i for i in range(num_triples) if sol[i]]
            to_remove = random.sample(active, min(num_remove, len(active)))
            for idx in to_remove:
                sol[idx] = False
                for ci in triple_trans[idx]:
                    counts[ci] -= 1
                current_count -= 1
            
            current_count = local_search(sol, counts, current_count, max_iters=80)
            
            if current_count > best_count:
                best_count = current_count
                best_solution = sol[:]
                best_counts = counts[:]

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init(0)

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution[:]
    final_counts = best_counts[:]
    for ci in range(num_trans):
        while final_counts[ci] == 8:
            for t_idx in transversals[ci]:
                if final_sol[t_idx]:
                    final_sol[t_idx] = False
                    for ci2 in triple_trans[t_idx]:
                        final_counts[ci2] -= 1
                    break

    result = [list(all_triples[i]) for i in range(num_triples) if final_sol[i]]
    return result
