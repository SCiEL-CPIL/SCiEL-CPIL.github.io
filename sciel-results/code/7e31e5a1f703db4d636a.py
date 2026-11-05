import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute for each k-subset, the list of t-subset indices it covers
    def block_to_indices(block):
        """Return list of t-subset indices covered by a block."""
        return [sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Precompute Gray code sequence for all C(v,k) blocks
    # Consecutive blocks differ by exactly one point
    all_blocks = list(itertools.combinations(range(v), k))
    N_blocks = len(all_blocks)
    
    # Build Gray code ordering: for combinations, use binary reflected Gray code on the index
    # Actually, for combinations we need a different approach.
    # We'll use a simple approach: sort by a Gray-code-like ordering
    # For C(19,7), we can use the fact that we can enumerate systematically
    
    # Better approach: use a recursive Gray code for combinations
    # Or simply precompute the delta for each consecutive pair in a natural ordering
    # and use that for incremental updates
    
    # Precompute coverage indices for each block
    block_covs = [block_to_indices(b) for b in all_blocks]
    
    # For Gray code traversal, we need to know which t-subsets change when going from block i to block i+1
    # In a standard combination enumeration, consecutive blocks may differ by more than one element
    # So we'll use a different strategy: precompute for each block, the set of t-subsets it covers
    # and use incremental updates based on set differences
    
    # Actually, let's implement a proper Gray code for combinations
    # This is complex, so instead we'll use a sliding window over a random permutation
    # with incremental coverage updates
    
    # Precompute point-to-t-subset mapping for faster gain computation
    # For each point x and each (t-1)-subset of other points, find the t-subset index
    t_minus_1_subs = list(itertools.combinations(range(v), t-1))
    point_t1_map = {}  # (point, frozenset_of_t_minus_1) -> t-subset index
    for sub in all_t_subs:
        for i in range(t):
            rest = tuple(p for j, p in enumerate(sub) if j != i)
            point_t1_map[(sub[i], frozenset(rest))] = sub_to_idx[sub]

    def greedy_cover(seed, strategy=0):
        """
        Greedy set cover with multiple strategies.
        strategy 0: random start, greedy extension
        strategy 1: start with 'hard' subsets (points with low degree in uncovered)
        strategy 2: deterministic min-start
        """
        rng = random.Random(seed)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Choose starting t-subset based on strategy
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
                
            if strategy == 0:
                start_sub_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
            elif strategy == 1:
                sample_size = min(50, len(unc_idx))
                sample_idx = rng.sample(list(unc_idx), sample_size)
                point_counts = defaultdict(int)
                for idx in sample_idx:
                    for p in all_t_subs[idx]:
                        point_counts[p] += 1
                best_idx = sample_idx[0]
                best_rarity = float('inf')
                for idx in sample_idx:
                    rarity = min(point_counts[p] for p in all_t_subs[idx])
                    if rarity < best_rarity:
                        best_rarity = rarity
                        best_idx = idx
                start_sub_idx = best_idx
            else:
                start_sub_idx = min(unc_idx)
            
            start_sub = all_t_subs[start_sub_idx]
            block = list(start_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate x using precomputed mapping
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute (t-1)-subsets of current block for fast lookup
                block_t1 = list(itertools.combinations(block, t-1))
                
                for x in candidates:
                    gain = 0
                    for rest in block_t1:
                        key = (x, frozenset(rest))
                        if key in point_t1_map:
                            idx = point_t1_map[key]
                            if uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    best_x = rng.choice(candidates)
                
                block.append(best_x)
            
            block.sort()
            
            # Update uncovered
            for idx in block_to_indices(block):
                uncovered[idx] = False
                
            blocks.append(block)
            
        return blocks

    def remove_redundant(blocks):
        """Remove redundant blocks using iterative single-pass."""
        if not blocks:
            return blocks
            
        # Compute coverage counts
        count = np.zeros(N_t, dtype=int)
        block_covs_local = []
        for b in blocks:
            covs = block_to_indices(b)
            block_covs_local.append(covs)
            for idx in covs:
                count[idx] += 1
        
        # Single pass: keep only critical blocks
        kept = []
        kept_covs = []
        for i, b in enumerate(blocks):
            is_critical = False
            for idx in block_covs_local[i]:
                if count[idx] == 1:
                    is_critical = True
                    break
            if is_critical:
                kept.append(b)
                kept_covs.append(block_covs_local[i])
            else:
                for idx in block_covs_local[i]:
                    count[idx] -= 1
        
        # Second pass on kept blocks
        if len(kept) < len(blocks):
            count2 = np.zeros(N_t, dtype=int)
            for covs in kept_covs:
                for idx in covs:
                    count2[idx] += 1
            kept2 = []
            for i, b in enumerate(kept):
                is_critical = False
                for idx in kept_covs[i]:
                    if count2[idx] == 1:
                        is_critical = True
                        break
                if is_critical:
                    kept2.append(b)
                else:
                    for idx in kept_covs[i]:
                        count2[idx] -= 1
            kept = kept2
        
        return kept

    def genetic_merge(blocks_a, blocks_b, uncovered_mask=None):
        """
        Merge two coverings using greedy selection with coverage tracking.
        """
        # Start with union, then greedily select blocks that cover most uncovered subsets
        all_blocks_merged = list(blocks_a) + list(blocks_b)
        seen = set()
        unique_merged = []
        for b in all_blocks_merged:
            key = tuple(sorted(b))
            if key not in seen:
                seen.add(key)
                unique_merged.append(b)
        
        # Greedy selection: pick blocks that cover most currently uncovered t-subsets
        uncovered = np.ones(N_t, dtype=bool)
        selected = []
        remaining = unique_merged[:]
        
        while uncovered.any() and remaining:
            best_gain = -1
            best_idx = -1
            for i, b in enumerate(remaining):
                covs = block_to_indices(b)
                gain = sum(1 for idx in covs if uncovered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if random.random() < 0.3:
                        best_idx = i
            
            if best_gain <= 0:
                break
            
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            for idx in block_to_indices(chosen):
                uncovered[idx] = False
        
        # If we didn't cover everything, add remaining blocks
        if uncovered.any():
            for b in remaining:
                selected.append(b)
                for idx in block_to_indices(b):
                    uncovered[idx] = False
        
        return remove_redundant(selected)

    def two_exchange(blocks):
        """
        Local search: try to remove blocks and replace with better ones.
        Uses incremental coverage tracking for efficiency.
        """
        if len(blocks) < 2:
            return blocks
        
        # Compute coverage counts
        block_covs = [block_to_indices(b) for b in blocks]
        count = np.zeros(N_t, dtype=int)
        for covs in block_covs:
            for idx in covs:
                count[idx] += 1
        
        # Try removing each block (in random order for diversity)
        order = list(range(len(blocks)))
        random.shuffle(order)
        
        for i in order:
            # Check if block i is removable
            removable = True
            for idx in block_covs[i]:
                if count[idx] == 1:
                    removable = False
                    break
            if removable:
                for idx in block_covs[i]:
                    count[idx] -= 1
                blocks.pop(i)
                block_covs.pop(i)
                return two_exchange(blocks)
        
        return blocks

    def local_search(blocks, rng, max_iter=50):
        """
        Local search: ILS-style perturbation with removal and re-addition.
        """
        if len(blocks) == 0:
            return blocks
        
        best_blocks = blocks[:]
        best_len = len(blocks)
        
        # Compute coverage
        block_covs = [block_to_indices(b) for b in blocks]
        count = np.zeros(N_t, dtype=int)
        for covs in block_covs:
            for idx in covs:
                count[idx] += 1
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturbation: randomly remove 1-3 blocks
            n_remove = rng.randint(1, min(3, len(blocks)))
            removed = []
            indices_to_remove = rng.sample(range(len(blocks)), n_remove)
            
            for i in sorted(indices_to_remove, reverse=True):
                removed.append(blocks[i])
                for idx in block_covs[i]:
                    count[idx] -= 1
                blocks.pop(i)
                block_covs.pop(i)
            
            # Re-add blocks greedily
            for b in removed:
                covs = block_to_indices(b)
                # Check if this block adds coverage
                gain = sum(1 for idx in covs if count[idx] == 0)
                if gain > 0 or rng.random() < 0.3:
                    blocks.append(b)
                    block_covs.append(covs)
                    for idx in covs:
                        count[idx] += 1
            
            # Polish: remove redundant blocks
            blocks = remove_redundant(blocks)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
        
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    initial_solutions = []
    strategies = [0, 1, 2]
    for strategy in strategies:
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            blocks = greedy_cover(seed * 100 + strategy, strategy)
            blocks = remove_redundant(blocks)
            if blocks:
                initial_solutions.append(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks[:]
    
    if not initial_solutions:
        return greedy_cover(0, 0)
    
    # Phase 2: ILS with perturbation and local search
    rng = random.Random(42)
    population = initial_solutions[:]
    
    for gen in range(30):
        if time.time() > deadline - 5:
            break
        
        # Selection: keep best solutions
        population.sort(key=lambda x: len(x))
        population = population[:max(3, len(population)//2)]
        
        # Generate new solutions via merge and local search
        new_population = []
        for i in range(min(len(population), 4)):
            if time.time() > deadline - 5:
                break
            
            # Merge with a random other solution
            if len(population) > 1:
                j = rng.choice([x for x in range(len(population)) if x != i])
                merged = genetic_merge(population[i], population[j])
            else:
                merged = population[i][:]
            
            # Apply local search with perturbation
            polished = local_search(merged, rng, max_iter=30)
            if polished:
                new_population.append(polished)
        
        population.extend(new_population)
        
        # Update best
        for blocks in population:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
        
        if best_len <= 131:
            break
    
    # Final polish on best solution
    if best_blocks and time.time() < deadline - 10:
        final = local_search(best_blocks, rng, max_iter=50)
        if len(final) < best_len:
            best_blocks = final
    
    return best_blocks
