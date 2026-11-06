"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Primary: Paley tournament on p=67 (known valid).
    Secondary: CSP-guided local search for smaller n (31-66) within time budget.
    """
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill

    # 1. Precompute the known valid Paley tournament on 67 vertices
    p_best = 67
    squares_best = [False] * p_best
    for x in range(1, p_best):
        squares_best[(x * x) % p_best] = True

    orientation_best = []
    for i in range(p_best):
        for j in range(i + 1, p_best):
            diff = (j - i) % p_best
            if squares_best[diff]:
                orientation_best.append(0)
            else:
                orientation_best.append(1)

    best_result = {'n': p_best, 'orientation': orientation_best}

    # 2. Check time remaining
    if time.time() - start_time > deadline - 5:
        return best_result

    # 3. CSP-guided local search for smaller n
    
    def build_in_neighbors(n, orientation):
        """Build in-neighborhood bitmasks directly from orientation list."""
        in_neigh = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orientation[idx] == 0:
                    # i -> j, so i is in-neighbor of j
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i, so j is in-neighbor of i
                    in_neigh[i] |= (1 << j)
                idx += 1
        return in_neigh

    def count_uncovered_and_list(n, in_neigh, max_list=100):
        """
        Count uncovered 4-sets and optionally return a list of them.
        Returns (uncovered_count, list_of_uncovered_4sets).
        """
        uncovered = 0
        uncovered_list = []
        for a in range(n):
            for b in range(a + 1, n):
                ab = in_neigh[a] & in_neigh[b]
                if ab == 0:
                    rem = n - b - 1
                    uncovered += rem * (rem - 1) // 2
                    if len(uncovered_list) < max_list:
                        for c in range(b + 1, n):
                            for d in range(c + 1, n):
                                if len(uncovered_list) < max_list:
                                    uncovered_list.append((a, b, c, d))
                    continue
                for c in range(b + 1, n):
                    abc = ab & in_neigh[c]
                    if abc == 0:
                        rem = n - c - 1
                        uncovered += rem
                        if len(uncovered_list) < max_list:
                            for d in range(c + 1, n):
                                if len(uncovered_list) < max_list:
                                    uncovered_list.append((a, b, c, d))
                        continue
                    for d in range(c + 1, n):
                        if (abc & in_neigh[d]) == 0:
                            uncovered += 1
                            if len(uncovered_list) < max_list:
                                uncovered_list.append((a, b, c, d))
        return uncovered, uncovered_list

    def arc_consistency_prune(n, in_neigh, max_iterations=10):
        """
        Apply arc consistency: if a pair (a,b) has no common dominator,
        then any orientation involving a or b that could create a dominator
        for pairs containing (a,b) is pruned.
        
        Returns a set of (i,j) pairs where i<j that are "forced" to have a specific
        orientation to avoid creating uncovered 4-sets.
        """
        forced = {}
        # For each pair (a,b) with no common dominator, we need to ensure
        # that in any 4-set containing a and b, some vertex dominates all four.
        # This means we need vertices that dominate a and b separately, and then
        # check if they can dominate the other two.
        
        # A simpler approach: identify critical pairs where in_neigh[a] & in_neigh[b] == 0
        critical_pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                if (in_neigh[a] & in_neigh[b]) == 0:
                    critical_pairs.append((a, b))
        
        # For each critical pair (a,b), find vertices that dominate a or b
        # and could potentially dominate both if we flip some edges.
        # This is complex; instead, we'll use a simpler heuristic:
        # For each critical pair, identify vertices that dominate exactly one of a,b.
        # Flipping the edge between such a vertex and the other of a,b might help.
        
        # Return critical pairs for targeted repair
        return critical_pairs

    def targeted_repair(n, orientation, in_neigh, critical_pairs, max_repair_steps=50):
        """
        Attempt to repair uncovered 4-sets by flipping edges involving critical pairs.
        Returns (success, new_orientation).
        """
        if not critical_pairs:
            return False, orientation
        
        orient = orientation[:]
        m = n * (n - 1) // 2
        
        # Build index lookup: pair (i,j) with i<j -> index in orientation list
        pair_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_to_idx[(i, j)] = idx
                idx += 1
        
        # For each critical pair, try flipping edges to create dominators
        for _ in range(max_repair_steps):
            if time.time() - start_time > deadline - 3:
                break
            
            # Pick a random critical pair
            a, b = random.choice(critical_pairs)
            
            # Find vertices that dominate a but not b, or b but not a
            candidates = []
            for v in range(n):
                if v == a or v == b:
                    continue
                dominates_a = (in_neigh[a] >> v) & 1
                dominates_b = (in_neigh[b] >> v) & 1
                if dominates_a != dominates_b:
                    candidates.append(v)
            
            if not candidates:
                continue
            
            # Try flipping edge between a random candidate and the non-dominated vertex
            v = random.choice(candidates)
            if (in_neigh[a] >> v) & 1:
                # v dominates a, not b. Flip edge so v dominates b too.
                # Need to flip edge between v and b.
                if v < b:
                    idx_vb = pair_to_idx[(v, b)]
                else:
                    idx_vb = pair_to_idx[(b, v)]
                # Currently: if orient[idx_vb]==0, v->b (v dominates b), else b->v
                # We want v->b, so set to 0
                if orient[idx_vb] != 0:
                    orient[idx_vb] = 0
                    # Rebuild in_neigh for affected vertices
                    # This is expensive; for now, just try and re-evaluate
                    new_in_neigh = build_in_neighbors(n, orient)
                    uncovered, _ = count_uncovered_and_list(n, new_in_neigh, max_list=1)
                    if uncovered == 0:
                        return True, orient
                    # Keep the flip if it reduced uncovered count
                    old_uncovered, _ = count_uncovered_and_list(n, in_neigh, max_list=1)
                    if uncovered < old_uncovered:
                        in_neigh = new_in_neigh
                        # Update critical pairs
                        critical_pairs = arc_consistency_prune(n, in_neigh)
                    else:
                        # Revert
                        orient[idx_vb] = 1 - orient[idx_vb]
            else:
                # v dominates b, not a. Flip edge so v dominates a too.
                if v < a:
                    idx_va = pair_to_idx[(v, a)]
                else:
                    idx_va = pair_to_idx[(a, v)]
                if orient[idx_va] != 0:
                    orient[idx_va] = 0
                    new_in_neigh = build_in_neighbors(n, orient)
                    uncovered, _ = count_uncovered_and_list(n, new_in_neigh, max_list=1)
                    if uncovered == 0:
                        return True, orient
                    old_uncovered, _ = count_uncovered_and_list(n, in_neigh, max_list=1)
                    if uncovered < old_uncovered:
                        in_neigh = new_in_neigh
                        critical_pairs = arc_consistency_prune(n, in_neigh)
                    else:
                        orient[idx_va] = 1 - orient[idx_va]
        
        return False, orient

    def random_orientation(n):
        m = n * (n - 1) // 2
        return list(np.random.randint(0, 2, size=m))

    def flip_bits(orientation, n, num_flips):
        new_orient = orientation[:]
        m = n * (n - 1) // 2
        for _ in range(num_flips):
            idx = random.randint(0, m - 1)
            new_orient[idx] = 1 - new_orient[idx]
        return new_orient

    # Search for smaller n using CSP-guided local search
    n_values = list(range(55, 30, -1))  # 55, 54, ..., 31
    
    for n in n_values:
        if time.time() - start_time > deadline - 3:
            break
        
        # Time budget for this n
        time_budget = min(8.0, max(1.0, (n - 30) * 0.2))
        n_start = time.time()
        
        pop_size = 40
        max_gens = 80
        
        # Initialize population with diverse random orientations
        population = []
        for _ in range(pop_size):
            orient = random_orientation(n)
            in_neigh = build_in_neighbors(n, orient)
            fitness, _ = count_uncovered_and_list(n, in_neigh, max_list=1)
            population.append((fitness, orient, in_neigh))
        
        population.sort(key=lambda x: x[0])
        
        if population[0][0] == 0:
            return {'n': n, 'orientation': population[0][1]}
        
        best_fitness = population[0][0]
        stagnation_count = 0
        
        for g in range(max_gens):
            if time.time() - n_start > time_budget:
                break
            if time.time() - start_time > deadline - 3:
                break
            
            new_population = []
            
            # Elitism: keep top 25%
            elite_count = max(2, pop_size // 4)
            for i in range(elite_count):
                new_population.append(population[i])
            
            # Fill rest with mutated offspring
            while len(new_population) < pop_size:
                # Tournament selection: pick 3 random, take best
                candidates = random.sample(range(pop_size), min(3, pop_size))
                parent_idx = min(candidates, key=lambda i: population[i][0])
                parent_orient = population[parent_idx][1]
                parent_in_neigh = population[parent_idx][2]
                
                # Mutation: flip 1-3 bits
                num_flips = random.randint(1, 3)
                child_orient = flip_bits(parent_orient, n, num_flips)
                
                # Evaluate
                child_in_neigh = build_in_neighbors(n, child_orient)
                fitness, _ = count_uncovered_and_list(n, child_in_neigh, max_list=1)
                
                new_population.append((fitness, child_orient, child_in_neigh))
                
                if fitness == 0:
                    return {'n': n, 'orientation': child_orient}
            
            population = new_population
            population.sort(key=lambda x: x[0])
            
            if population[0][0] == 0:
                return {'n': n, 'orientation': population[0][1]}
            
            # If stuck, apply targeted repair to the best individual
            if population[0][0] == best_fitness:
                stagnation_count += 1
                if stagnation_count >= 5:
                    # Apply CSP-guided repair to top few individuals
                    for i in range(min(3, len(population))):
                        if time.time() - n_start > time_budget:
                            break
                        if time.time() - start_time > deadline - 3:
                            break
                        
                        fitness, orient, in_neigh = population[i]
                        critical_pairs = arc_consistency_prune(n, in_neigh)
                        success, repaired_orient = targeted_repair(n, orient, in_neigh, critical_pairs, max_repair_steps=30)
                        
                        if success:
                            return {'n': n, 'orientation': repaired_orient}
                        
                        # Evaluate repaired orientation
                        new_in_neigh = build_in_neighbors(n, repaired_orient)
                        new_fitness, _ = count_uncovered_and_list(n, new_in_neigh, max_list=1)
                        
                        if new_fitness < fitness:
                            population[i] = (new_fitness, repaired_orient, new_in_neigh)
                            population.sort(key=lambda x: x[0])
                            if population[0][0] == 0:
                                return {'n': n, 'orientation': population[0][1]}
                    
                    stagnation_count = 0
                    best_fitness = population[0][0]
            else:
                stagnation_count = 0
                best_fitness = population[0][0]
            
            # Inject diversity every 15 generations
            if g > 0 and g % 15 == 0:
                for _ in range(pop_size // 5):
                    orient = random_orientation(n)
                    in_neigh = build_in_neighbors(n, orient)
                    fitness, _ = count_uncovered_and_list(n, in_neigh, max_list=1)
                    population.append((fitness, orient, in_neigh))
                population.sort(key=lambda x: x[0])
                population = population[:pop_size]

    # If search didn't find anything better, return the best known (67)
    return best_result

# EVOLVE-BLOCK-END
