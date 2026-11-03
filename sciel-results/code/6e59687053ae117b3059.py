import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets and k-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    t_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    N_k = len(all_k_blocks)
    
    # Map block index to the list of t-subset indices it covers
    block_to_t = []
    for b in all_k_blocks:
        covered_ts = [t_idx[s] for s in itertools.combinations(b, t)]
        block_to_t.append(covered_ts)
        
    # Map t-subset index to list of block indices that cover it
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_to_t):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: Fast Greedy Repair
    # Takes a set of selected blocks (as a set of indices) and ensures validity.
    # Returns the modified set.
    def repair(blocks_set):
        # Calculate coverage counts
        cnt = np.zeros(N_t, dtype=np.int32)
        for bi in blocks_set:
            for ti in block_to_t[bi]:
                cnt[ti] += 1
        
        # Find uncovered t-subsets
        uncovered = np.where(cnt == 0)[0]
        
        while len(uncovered) > 0 and time.time() < deadline - 5.0:
            # Pick an uncovered t-subset. To speed up, pick the one with fewest candidates?
            # Or just pick the first one. First one is O(1).
            # Let's pick the first uncovered one for simplicity, but maybe randomize to avoid bias?
            # Randomizing might help escape local structures in repair.
            # Let's stick to deterministic first for speed, as repair is called very often.
            target_t = uncovered[0]
            
            candidates = t_to_blocks[target_t]
            if not candidates:
                # Should not happen if v,k,t are valid params for a covering design
                break
                
            best_block = -1
            max_gain = -1
            
            for bi in candidates:
                if bi in blocks_set:
                    continue
                # Calculate how many currently uncovered t-subsets this block covers
                gain = 0
                for ti in block_to_t[bi]:
                    if cnt[ti] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = bi
            
            if best_block == -1:
                # Fallback: pick first candidate
                best_block = candidates[0]
                
            blocks_set.add(best_block)
            
            # Update counts
            for ti in block_to_t[best_block]:
                cnt[ti] += 1
                
            # Update uncovered list
            # This is inefficient if done naively. 
            # Since N_t is ~11628 and repair is called often, we need this to be fast.
            # Recalculating np.where is O(N_t). 
            # Alternative: maintain a set of uncovered indices.
            # Let's switch to a set for uncovered to make removal O(1).
            
            # Actually, let's refactor repair to use a set for uncovered.
            # The current loop recalculates np.where every iteration. 
            # For 300 blocks, this might be okay if repair is fast.
            # Let's optimize: instead of np.where, keep a list/set of uncovered.
            
            # Let's restart the repair logic with a more efficient structure if needed.
            # For now, assume np.where is acceptable for small N_t (11k).
            uncovered = np.where(cnt == 0)[0]
            
        return blocks_set

    # Helper: Greedy Initial Solution
    def greedy_init():
        blocks_set = set()
        cnt = np.zeros(N_t, dtype=np.int32)
        
        while np.any(cnt == 0) and time.time() < deadline - 10.0:
            # Find uncovered
            uncovered = np.where(cnt == 0)[0]
            if len(uncovered) == 0:
                break
                
            # Pick a random uncovered t-subset to anchor
            target_t = uncovered[random.randint(0, len(uncovered)-1)]
            
            candidates = t_to_blocks[target_t]
            best_block = -1
            max_gain = -1
            
            for bi in candidates:
                if bi in blocks_set:
                    continue
                gain = 0
                for ti in block_to_t[bi]:
                    if cnt[ti] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = bi
                    
            if best_block == -1:
                best_block = candidates[0]
                
            blocks_set.add(best_block)
            for ti in block_to_t[best_block]:
                cnt[ti] += 1
                
        return blocks_set

    # Helper: Prune Redundant Blocks
    def prune(blocks_set):
        if not blocks_set:
            return blocks_set
        
        cnt = np.zeros(N_t, dtype=np.int32)
        for bi in blocks_set:
            for ti in block_to_t[bi]:
                cnt[ti] += 1
                
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Iterate over a copy because we modify the set
            for bi in list(blocks_set):
                # Check if block is essential
                essential = False
                for ti in block_to_t[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    blocks_set.discard(bi)
                    for ti in block_to_t[bi]:
                        cnt[ti] -= 1
                    changed = True
        return blocks_set

    # 2. Initialize Population
    pop_size = 20
    population = []
    
    # First individual: Greedy
    pop0 = greedy_init()
    pop0 = prune(pop0)
    population.append(pop0)
    
    # Other individuals: Perturbed Greedy
    for i in range(1, pop_size):
        if time.time() > deadline - 20.0:
            break
        # Start with greedy, then add random noise?
        # Or just run greedy with different random seeds?
        # Greedy is deterministic given the random choices.
        # Let's create diverse starts by taking greedy and removing some blocks, then repairing.
        base = set(pop0)
        # Remove 5% of blocks
        n_remove = max(1, int(len(base) * 0.05))
        to_remove = random.sample(list(base), n_remove)
        for bi in to_remove:
            base.discard(bi)
        
        # Repair
        base = repair(base)
        base = prune(base)
        population.append(base)

    best_individual = min(population, key=len)
    best_size = len(best_individual)
    
    # 3. Genetic Algorithm Loop
    # Parameters
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 5
    
    generation = 0
    max_generations = 500 # Safety cap, time limit is real constraint
    
    while time.time() < deadline - 15.0:
        generation += 1
        
        # Sort population by size (cost)
        population.sort(key=len)
        
        # Elitism: Keep top elite_size
        elites = [set(p) for p in population[:elite_size]]
        
        # Create New Population
        new_population = list(elites)
        
        while len(new_population) < pop_size:
            if time.time() > deadline - 15.0:
                break
            
            # Select Parents
            # Tournament Selection: Pick 2 random, take better
            p1 = random.choice(population)
            p2 = random.choice(population)
            if len(p1) < len(p2):
                parent1, parent2 = p1, p2
            else:
                parent1, parent2 = p2, p1
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform Crossover: Union of parents
                # This ensures validity (superset of valid sets is valid)
                child = set(parent1) | set(parent2)
            else:
                # Just copy parent
                child = set(parent1)
            
            # Mutation
            # Remove random blocks to encourage reduction
            # Number of mutations proportional to size
            n_mutations = max(1, int(len(child) * mutation_rate))
            if child:
                # Sample indices to remove
                # We can't sample from set directly, convert to list
                child_list = list(child)
                if len(child_list) > n_mutations:
                    to_remove = random.sample(child_list, n_mutations)
                    for bi in to_remove:
                        child.discard(bi)
                else:
                    # If set is too small, just clear it? No, that's bad.
                    # If n_mutations >= len, we remove all.
                    child.clear()
            
            # Repair
            # This is the most expensive part.
            child = repair(child)
            
            # Prune
            child = prune(child)
            
            new_population.append(child)
            
            # Track best
            if len(child) < best_size:
                best_size = len(child)
                best_individual = set(child)
        
        # Replace population
        population = new_population
        
        # Optional: If no improvement in many generations, restart with new random elites?
        # For now, rely on mutation/crossover.

    # 4. Final Polish on Best Individual
    # Try a few local search moves on the best individual found by GA
    current_best = set(best_individual)
    
    # Simple Local Search: Try to remove blocks one by one if possible
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for bi in list(current_best):
            # Try removing bi
            test_set = set(current_best)
            test_set.discard(bi)
            
            # Check validity fast
            # Instead of full repair, just check if it's valid.
            # If valid, accept.
            # If invalid, reject.
            # This is a "Strict Descent" local search.
            
            # Validity Check
            cnt = np.zeros(N_t, dtype=np.int32)
            for b in test_set:
                for ti in block_to_t[b]:
                    cnt[ti] += 1
            
            if np.min(cnt) >= 1:
                current_best = test_set
                changed = True
                break # Restart loop as set changed
    
    # Final Prune (redundant but safe)
    current_best = prune(current_best)
    
    # Final Verification
    cnt_final = np.zeros(N_t, dtype=np.int32)
    for bi in current_best:
        for ti in block_to_t[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback to a valid greedy solution if GA produced garbage (shouldn't happen)
        fallback = greedy_init()
        fallback = prune(fallback)
        current_best = fallback

    # Convert to list of lists
    result = [list(all_k_blocks[bi]) for bi in current_best]
    
    return result
