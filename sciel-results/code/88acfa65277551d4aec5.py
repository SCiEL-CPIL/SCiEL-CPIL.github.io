import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit

    # Generate all 4-element subsets of [16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    rng = random.Random(42)
    
    # Precompute pairwise intersections for all mask pairs
    inter_matrix = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            ij = mi & all_masks[j]
            inter_matrix[i][j] = ij
            inter_matrix[j][i] = ij
            
    def is_valid_addition_fast(fam_indices, new_idx):
        """Check if all_masks[new_idx] can be added to family (given as indices) without creating a sunflower."""
        n = len(fam_indices)
        if n < 2:
            return True
        
        # Group existing masks by their intersection with new_mask
        groups = defaultdict(list)
        for fam_idx in fam_indices:
            inter = inter_matrix[fam_idx][new_idx]
            groups[inter].append(fam_idx)
        
        # For each group, check if any pair has intersection equal to the group key
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            # Early exit optimization: if group is large, check strategically
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    if inter_matrix[a][b] == inter:
                        return False
        return True

    def greedy_build(order, deadline_check=True):
        """Build a valid family using a specific order of indices."""
        family_indices = []
        for idx in order:
            if deadline_check and time.time() > deadline - 5:
                break
            if is_valid_addition_fast(family_indices, idx):
                family_indices.append(idx)
        return family_indices

    def crossover(parent1, parent2, deadline_check=True):
        """
        Crossover: Merge two families by greedily adding sets from their union.
        The union is ordered to prefer sets present in both (intersection) or randomly.
        """
        # Create a list of unique indices from both parents
        # We want to explore the "space" between them.
        # Strategy: Randomize the order of the union, but maybe bias towards larger intersection?
        # Simpler: Just shuffle the union.
        
        set1 = set(parent1)
        set2 = set(parent2)
        union_indices = list(set1.union(set2))
        
        # To make it more "genetic", we can try to keep the structure.
        # But for sunflowers, any subset of a valid family is valid.
        # So greedy selection from the union is a valid operator.
        
        rng.shuffle(union_indices)
        
        offspring = []
        for idx in union_indices:
            if deadline_check and time.time() > deadline - 5:
                break
            if is_valid_addition_fast(offspring, idx):
                offspring.append(idx)
        
        return offspring

    def mutate(family, deadline_check=True):
        """
        Mutation: Try to add random sets or swap.
        """
        new_family = list(family)
        new_set = set(new_family)
        
        # Try adding random non-family sets
        num_tries = 50
        for _ in range(num_tries):
            if deadline_check and time.time() > deadline - 5:
                break
            idx = rng.randrange(n_total)
            if all_masks[idx] not in new_set:
                if is_valid_addition_fast(new_family, idx):
                    new_family.append(idx)
                    new_set.add(idx)
        
        # Try swapping
        # Pick a random set in family, remove it, try to add a random set
        if new_family:
            remove_pos = rng.randrange(len(new_family))
            removed_idx = new_family.pop(remove_pos)
            new_set.discard(removed_idx)
            
            added = False
            for _ in range(50):
                if deadline_check and time.time() > deadline - 5:
                    break
                idx = rng.randrange(n_total)
                if all_masks[idx] not in new_set:
                    if is_valid_addition_fast(new_family, idx):
                        new_family.append(idx)
                        new_set.add(idx)
                        added = True
                        break
            
            if not added:
                # Restore
                new_family.insert(remove_pos, removed_idx)
                new_set.add(removed_idx)
                
        return new_family

    # Initialize population with diverse greedy builds
    population = []
    pop_size = 20
    
    # First few are deterministic or seeded shuffles
    for i in range(pop_size):
        order = list(range(n_total))
        if i > 0:
            rng.shuffle(order)
        
        fam = greedy_build(order)
        population.append(fam)
        
        if time.time() > deadline - 30:
            break

    # Sort population by size
    population.sort(key=lambda x: len(x), reverse=True)
    
    best_family = list(population[0])
    best_size = len(best_family)
    
    # GA Loop
    max_generations = 20
    for gen in range(max_generations):
        if time.time() > deadline - 10:
            break
            
        new_population = []
        
        # Keep the best few from previous generation (Elitism)
        elite_size = 5
        for i in range(min(elite_size, len(population))):
            new_population.append(list(population[i]))
            
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            # Selection: Tournament selection
            # Pick 3 random parents, choose the best
            p1_idx = rng.randrange(len(population))
            p2_idx = rng.randrange(len(population))
            p3_idx = rng.randrange(len(population))
            
            candidates = [population[p1_idx], population[p2_idx], population[p3_idx]]
            p1 = max(candidates, key=len)
            
            # Crossover
            child = crossover(p1, population[p2_idx])
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
            
            # Update best
            if len(child) > best_size:
                best_size = len(child)
                best_family = list(child)
                
        population = new_population
        population.sort(key=lambda x: len(x), reverse=True)
        
        # Update best if population improved
        if len(population[0]) > best_size:
            best_size = len(population[0])
            best_family = list(population[0])

    # Final Polish: Try to add any remaining sets to the best family found
    if time.time() < deadline - 2:
        family_indices = list(best_family)
        family_set = set(all_masks[i] for i in family_indices)
        
        # Try to add any missing set
        for idx in range(n_total):
            if all_masks[idx] in family_set:
                continue
            if is_valid_addition_fast(family_indices, idx):
                family_indices.append(idx)
                family_set.add(all_masks[idx])
                if len(family_indices) > best_size:
                    best_size = len(family_indices)
                    best_family = list(family_indices)
                if len(family_indices) >= 250:
                    break

    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))

    return result
