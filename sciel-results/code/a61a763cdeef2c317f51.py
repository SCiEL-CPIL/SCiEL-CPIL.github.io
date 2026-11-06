import itertools
import random
import time
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_trans
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
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

    def local_search(sol, counts, current_count, time_limit):
        """Enhanced local search with simulated annealing flavor."""
        best_sol = sol[:]
        best_counts = counts[:]
        best_count = current_count
        stagnation = 0
        max_stagnation = 50
        
        while time.time() < time_limit:
            # Try to add candidates
            candidates = [i for i in range(num_triples) if not sol[i]]
            random.shuffle(candidates)
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_sol = sol[:]
                    best_counts = counts[:]
                continue

            # Find tight configurations (count == 7)
            tight_triples = []
            for ci, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in transversals[ci]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
            
            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                active = [i for i in range(num_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)
            
            # Remove the kicked triple
            sol[kick_idx] = False
            for ci in triple_trans[kick_idx]:
                counts[ci] -= 1
            current_count -= 1
            
            # Try to add back multiple candidates
            candidates = [i for i in range(num_triples) if not sol[i]]
            random.shuffle(candidates)
            added, _ = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added >= 1:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_sol = sol[:]
                    best_counts = counts[:]
            else:
                # Revert the kick if it didn't help
                sol[kick_idx] = True
                for ci in triple_trans[kick_idx]:
                    counts[ci] += 1
                current_count += 1
                stagnation += 1
                if stagnation > max_stagnation:
                    break

        return best_sol, best_counts, best_count

    def two_opt_swap(sol, counts, current_count, time_limit):
        """Try 2-opt swaps: remove two triples, add up to three."""
        best_sol = sol[:]
        best_counts = counts[:]
        best_count = current_count
        stagnation = 0
        max_stagnation = 30
        
        while time.time() < time_limit:
            active = [i for i in range(num_triples) if sol[i]]
            if len(active) < 2:
                break
            
            # Pick two random active triples to remove
            i, j = random.sample(active, 2)
            sol[i] = False
            sol[j] = False
            for ci in triple_trans[i]:
                counts[ci] -= 1
            for ci in triple_trans[j]:
                counts[ci] -= 1
            current_count -= 2
            
            # Try to add candidates
            candidates = [k for k in range(num_triples) if not sol[k]]
            random.shuffle(candidates)
            added, _ = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added >= 2:
                stagnation = 0
                if current_count > best_count:
                    best_count = current_count
                    best_sol = sol[:]
                    best_counts = counts[:]
            else:
                # Revert
                sol[i] = True
                sol[j] = True
                for ci in triple_trans[i]:
                    counts[ci] += 1
                for ci in triple_trans[j]:
                    counts[ci] += 1
                current_count += 2
                stagnation += 1
                if stagnation > max_stagnation:
                    break

        return best_sol, best_counts, best_count

    def simulated_annealing(sol, counts, current_count, time_limit):
        """Simulated annealing with temperature-controlled swaps."""
        best_sol = sol[:]
        best_counts = counts[:]
        best_count = current_count
        stagnation = 0
        max_stagnation = 100
        temperature = 1.0
        cooling = 0.995
        min_temp = 0.01
        
        import math
        
        while time.time() < time_limit and temperature > min_temp:
            active = [i for i in range(num_triples) if sol[i]]
            if not active:
                break
            
            # Random kick: remove one active triple
            kick_idx = random.choice(active)
            sol[kick_idx] = False
            for ci in triple_trans[kick_idx]:
                counts[ci] -= 1
            old_count = current_count
            current_count -= 1
            
            # Try to add candidates
            candidates = [i for i in range(num_triples) if not sol[i]]
            random.shuffle(candidates)
            added, _ = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            delta = current_count - old_count
            
            if delta >= 0:
                # Accept improvement
                if current_count > best_count:
                    best_count = current_count
                    best_sol = sol[:]
                    best_counts = counts[:]
                stagnation = 0
            else:
                # Accept with probability exp(delta / temperature)
                if random.random() < math.exp(delta / max(temperature, 0.001)):
                    # Accept worsening
                    if current_count > best_count:
                        best_count = current_count
                        best_sol = sol[:]
                        best_counts = counts[:]
                else:
                    # Revert
                    sol[kick_idx] = True
                    for ci in triple_trans[kick_idx]:
                        counts[ci] += 1
                    current_count = old_count
                stagnation += 1
            
            temperature *= cooling
            
            if stagnation > max_stagnation:
                break

        return best_sol, best_counts, best_count

    best_solution = None
    best_count = -1

    # Phase 1: Multiple greedy restarts with local search
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        sol, counts, current_count = greedy_init()
        
        # Run local search
        sol, counts, current_count = local_search(sol, counts, current_count, deadline - 3.0)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]

    # Phase 2: 2-opt swaps from best solution
    if best_solution is not None and time.time() < deadline - 3.0:
        counts = [0] * num_trans
        for i in range(num_triples):
            if best_solution[i]:
                for ci in triple_trans[i]:
                    counts[ci] += 1
        best_solution, counts, best_count = two_opt_swap(
            best_solution, counts, best_count, deadline - 2.0)

    # Phase 3: Simulated annealing from best solution
    if best_solution is not None and time.time() < deadline - 1.0:
        counts = [0] * num_trans
        for i in range(num_triples):
            if best_solution[i]:
                for ci in triple_trans[i]:
                    counts[ci] += 1
        best_solution, counts, best_count = simulated_annealing(
            best_solution, counts, best_count, deadline - 0.5)

    if best_solution is None:
        best_solution, best_counts, _ = greedy_init()

    # Final validation: ensure no transversal is fully present
    final_sol = best_solution[:]
    final_counts = [0] * num_trans
    for i in range(num_triples):
        if final_sol[i]:
            for ci in triple_trans[i]:
                final_counts[ci] += 1
    
    # Remove any remaining conflicts
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
