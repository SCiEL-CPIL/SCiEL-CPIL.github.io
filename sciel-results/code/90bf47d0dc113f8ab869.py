import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each block which t-subsets it covers (as list of indices)
    block_t_subsets = []
    for block in all_blocks:
        subs = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_subsets.append(subs)
    
    # Build reverse index: for each t-subset, which blocks cover it
    # Using lists for flexibility
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i, subs in enumerate(block_t_subsets):
        for j in subs:
            subset_to_blocks[j].append(i)
    
    def coverage_count(blocks_list):
        """Return array of coverage counts for each t-subset."""
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks_list:
            counts[block_t_subsets[b]] += 1
        return counts
    
    def is_valid(blocks_list):
        """Check if all t-subsets are covered at least once."""
        counts = coverage_count(blocks_list)
        return np.all(counts >= 1)
    
    def greedy_construct(random_seed=None):
        """Randomized greedy set cover: at each step, pick a block that covers
        the most uncovered t-subsets, with random tie-breaking."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random
        
        uncovered = set(range(num_t_subsets))
        selected = []
        
        # Precompute gains for each block
        gains = np.array([len(block_t_subsets[i]) for i in range(num_blocks)], dtype=np.int32)
        
        # We'll maintain which t-subsets are uncovered
        # For speed, use a numpy array
        uncovered_arr = np.ones(num_t_subsets, dtype=np.bool_)
        
        while len(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Find block with max gain among uncovered
            # Recompute gains lazily: for each block, count how many of its t-subsets are uncovered
            # This is expensive, so let's use a smarter approach
            
            # Actually, let's just pick randomly from top candidates for speed
            # First, find approximate best by sampling
            sample_size = min(200, num_blocks)
            candidates = rng.sample(range(num_blocks), sample_size)
            
            best_block = -1
            best_gain = -1
            for b in candidates:
                gain = sum(1 for s in block_t_subsets[b] if s in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block == -1 or best_gain == 0:
                # Try a few more random blocks
                for b in rng.sample(range(num_blocks), min(50, num_blocks)):
                    gain = sum(1 for s in block_t_subsets[b] if s in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
            
            if best_block == -1:
                break
            
            selected.append(best_block)
            
            # Update uncovered
            for s in block_t_subsets[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
        
        return selected
    
    def prune_redundant(blocks_list):
        """Remove redundant blocks (those whose t-subsets are all covered by others)."""
        if not blocks_list:
            return blocks_list
        
        counts = coverage_count(blocks_list)
        changed = True
        while changed:
            changed = False
            new_list = []
            for b in blocks_list:
                # Check if block b is redundant
                redundant = True
                for s in block_t_subsets[b]:
                    if counts[s] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_list.append(b)
                else:
                    # Remove it
                    for s in block_t_subsets[b]:
                        counts[s] -= 1
                    changed = True
            blocks_list = new_list
        return blocks_list
    
    def local_search_2opt(blocks_list):
        """Try swapping one block for another to improve (reduce) coverage count."""
        if not blocks_list:
            return blocks_list
        
        current_counts = coverage_count(blocks_list)
        current_size = len(blocks_list)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            block_set = set(blocks_list)
            
            # For each block in the solution, try replacing it with a block not in the solution
            # that results in a smaller or equal solution with valid coverage
            for i, b in enumerate(list(blocks_list)):
                if time.time() > deadline - 10.0:
                    break
                
                # Temporarily remove block b
                for s in block_t_subsets[b]:
                    current_counts[s] -= 1
                
                # Find blocks that cover the "critical" t-subsets (those with count 0 after removal)
                critical = [s for s in block_t_subsets[b] if current_counts[s] == 0]
                
                if not critical:
                    # Block b was fully redundant
                    blocks_list.pop(i)
                    improved = True
                    break
                
                # Find candidate replacement blocks that cover all critical t-subsets
                candidates = []
                for c in subset_to_blocks[critical[0]]:
                    if c in block_set:
                        continue
                    # Check if c covers all critical t-subsets
                    covers_all = all(s in block_t_subsets[c] for s in critical)
                    if covers_all:
                        candidates.append(c)
                
                if candidates:
                    # Pick a random candidate
                    new_b = random.choice(candidates)
                    blocks_list[i] = new_b
                    block_set.discard(b)
                    block_set.add(new_b)
                    # Update counts
                    for s in block_t_subsets[new_b]:
                        current_counts[s] += 1
                    improved = True
                    break
                else:
                    # Restore
                    for s in block_t_subsets[b]:
                        current_counts[s] += 1
        
        return blocks_list
    
    def iterated_local_search(num_restarts=None):
        """ILS: construct, prune, local search, then perturb and repeat."""
        best_solution = None
        best_size = float('inf')
        
        if num_restarts is None:
            # Estimate number of restarts based on time
            elapsed = time.time() - start_time
            time_per_restart = 5.0  # seconds
            num_restarts = max(3, int((deadline - 5.0 - start_time) / time_per_restart))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Construct initial solution
            seed = restart * 7919 + 13
            sol = greedy_construct(random_seed=seed)
            
            # Prune
            sol = prune_redundant(sol)
            
            # Local search
            sol = local_search_2opt(sol)
            
            # Final prune
            sol = prune_redundant(sol)
            
            # Check validity
            if is_valid(sol):
                if len(sol) < best_size:
                    best_size = len(sol)
                    best_solution = sol[:]
            
            # Perturbation: remove a few random blocks and re-cover
            if best_solution is not None and time.time() < deadline - 10.0:
                # Start from best_solution for next iteration's perturbation
                perturbed = best_solution[:]
                # Remove 1-3 random blocks
                num_remove = random.randint(1, 3)
                for _ in range(min(num_remove, len(perturbed))):
                    idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
                
                # Re-cover the gaps using greedy
                uncovered = set(range(num_t_subsets))
                counts = np.zeros(num_t_subsets, dtype=np.int32)
                for b in perturbed:
                    counts[block_t_subsets[b]] += 1
                for s in range(num_t_subsets):
                    if counts[s] == 0:
                        uncovered.add(s)
                
                while uncovered and time.time() < deadline - 5.0:
                    # Find block covering most uncovered
                    best_b = -1
                    best_gain = -1
                    for b in random.sample(range(num_blocks), min(100, num_blocks)):
                        gain = sum(1 for s in block_t_subsets[b] if s in uncovered)
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    if best_b == -1 or best_gain == 0:
                        break
                    perturbed.append(best_b)
                    for s in block_t_subsets[best_b]:
                        if s in uncovered:
                            uncovered.discard(s)
                
                perturbed = prune_redundant(perturbed)
                perturbed = local_search_2opt(perturbed)
                perturbed = prune_redundant(perturbed)
                
                if is_valid(perturbed) and len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_solution = perturbed[:]
        
        return best_solution if best_solution is not None else []
    
    # Main: run ILS
    solution = iterated_local_search()
    
    # Final safety: if solution is invalid, fall back to a simple greedy
    if not is_valid(solution):
        solution = greedy_construct(random_seed=42)
        solution = prune_redundant(solution)
    
    # Convert to output format
    blocks = [list(all_blocks[b]) for b in solution]
    return blocks
