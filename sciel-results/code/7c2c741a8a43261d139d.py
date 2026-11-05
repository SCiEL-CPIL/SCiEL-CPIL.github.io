import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute coverage for each k-subset
    all_blocks = list(itertools.combinations(range(v), k))
    N_blocks = len(all_blocks)
    block_to_indices = [
        [sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        for b in all_blocks
    ]

    # Precompute for each t-subset index, which block indices cover it
    # This is expensive but C(19,4)=3876, C(19,7)=50388, so 3876*50388 ~ 195M entries
    # That's too much. Instead, for each t-subset, find blocks containing it.
    # A block contains a t-subset iff the t-subset is a subset of the block.
    # For a given t-subset S, the blocks containing S are those that include S and add k-t more elements from the remaining v-t elements.
    # So number of blocks containing S = C(v-t, k-t) = C(19-4, 7-4) = C(15,3) = 455
    # Total entries = 3876 * 455 = 1,763,580 - manageable

    t_sub_to_blocks = defaultdict(list)
    for bi, b in enumerate(all_blocks):
        b_set = set(b)
        for sub in itertools.combinations(b, t):
            t_sub_to_blocks[sub].append(bi)

    # Convert to array for faster access
    t_sub_block_lists = [t_sub_to_blocks[sub] for sub in all_t_subs]

    def greedy_cover(seed, strategy=0):
        """
        Greedy set cover with multiple strategies.
        """
        rng = random.Random(seed)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
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
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute (t-1)-subsets of current block
                block_t1 = list(itertools.combinations(block, t-1))
                
                for x in candidates:
                    gain = 0
                    for rest in block_t1:
                        s = tuple(sorted([x] + list(rest)))
                        if s in sub_to_idx:
                            idx = sub_to_idx[s]
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
            bi = all_blocks.index(tuple(block))
            for idx in block_to_indices[bi]:
                uncovered[idx] = False
            blocks.append(list(block))
            
        return blocks

    def remove_redundant(blocks):
        """Remove redundant blocks using iterative single-pass."""
        if not blocks:
            return blocks
            
        count = np.zeros(N_t, dtype=int)
        block_covs_local = []
        for b in blocks:
            bi = all_blocks.index(tuple(b))
            covs = block_to_indices[bi]
            block_covs_local.append(covs)
            for idx in covs:
                count[idx] += 1
        
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

    def simulated_annealing(initial_blocks, seed, max_iter=200):
        """
        Simulated annealing where state is a set of blocks.
        Moves: swap a random block with a candidate that maximizes marginal coverage gain,
        or remove a redundant block.
        Energy = number of blocks (we want to minimize).
        """
        rng = random.Random(seed)
        blocks = list(initial_blocks)
        n_blocks = len(blocks)
        
        # Compute coverage counts
        count = np.zeros(N_t, dtype=int)
        block_indices = [all_blocks.index(tuple(b)) for b in blocks]
        for bi in block_indices:
            for idx in block_to_indices[bi]:
                count[idx] += 1
        
        best_blocks = blocks[:]
        best_len = n_blocks
        best_count = count.copy()
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        alpha = 0.95
        
        T = T_start
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Choose a move type
            move_type = rng.random()
            
            if move_type < 0.3 and n_blocks > 1:
                # Try removing a block
                i = rng.randint(0, n_blocks - 1)
                bi = block_indices[i]
                covs = block_to_indices[bi]
                
                # Check if removable
                removable = True
                for idx in covs:
                    if count[idx] == 1:
                        removable = False
                        break
                
                if removable:
                    # Accept: remove block
                    for idx in covs:
                        count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    n_blocks -= 1
                    
                    if n_blocks < best_len:
                        best_len = n_blocks
                        best_blocks = blocks[:]
                        best_count = count.copy()
                        
            elif move_type < 0.7:
                # Try swapping a block with a better one
                i = rng.randint(0, n_blocks - 1)
                bi_old = block_indices[i]
                covs_old = block_to_indices[bi_old]
                
                # Compute what would be uncovered if we remove this block
                # Gain of new block = number of t-subsets it covers that are currently only covered by this block
                # We want to find a block that covers at least the same critical subsets
                
                # Find a candidate block
                # Sample some random blocks
                best_candidate = -1
                best_gain_diff = 0  # gain of new - loss of old (in terms of new coverage)
                
                for _ in range(20):
                    bi_new = rng.randint(0, N_blocks - 1)
                    if bi_new == bi_old:
                        continue
                    covs_new = block_to_indices[bi_new]
                    
                    # Calculate: how many t-subsets are newly covered by new block that weren't covered before?
                    # And how many would become uncovered if we remove old block?
                    
                    # Simulate removal of old block
                    new_count = count.copy()
                    for idx in covs_old:
                        new_count[idx] -= 1
                        if new_count[idx] < 0:
                            break
                    else:
                        # Now check if new block covers all the subsets that old block uniquely covered
                        # and adds new coverage
                        new_coverage_gain = 0
                        for idx in covs_new:
                            if new_count[idx] == 0:
                                new_coverage_gain += 1
                        new_count[idx] += 1  # This is wrong, let me redo
                        
                        # Actually, let's just compute the energy change
                        # Energy = number of blocks. A swap keeps energy the same.
                        # But we want to eventually reduce block count.
                        # So a swap is only useful if it enables a subsequent removal.
                        # Let's just accept swaps that don't increase energy (they keep it same)
                        # with some probability, and focus on removals.
                        pass
                
                # Simpler: just try to replace a block with one that covers more uncovered subsets
                # But all subsets should be covered... 
                # Let's do: find a block that covers more "critical" subsets (count==1)
                critical_of_old = [idx for idx in covs_old if count[idx] == 1]
                
                if len(critical_of_old) == 0:
                    # Block is redundant, try to remove it
                    for idx in covs_old:
                        count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    n_blocks -= 1
                    
                    if n_blocks < best_len:
                        best_len = n_blocks
                        best_blocks = blocks[:]
                        best_count = count.copy()
                else:
                    # Find a replacement that covers all critical subsets
                    # and possibly more
                    best_candidate = -1
                    best_new_coverage = -1
                    
                    # Sample candidates
                    for _ in range(30):
                        bi_new = rng.randint(0, N_blocks - 1)
                        if bi_new == bi_old:
                            continue
                        covs_new = block_to_indices[bi_new]
                        covs_new_set = set(covs_new)
                        
                        # Check if it covers all critical subsets
                        covers_all = all(c in covs_new_set for c in critical_of_old)
                        if not covers_all:
                            continue
                        
                        # Count how many additional subsets it covers
                        additional = sum(1 for idx in covs_new if count[idx] > 0 and idx not in set(covs_old))
                        
                        if additional > best_new_coverage:
                            best_new_coverage = additional
                            best_candidate = bi_new
                    
                    if best_candidate != -1:
                        # Perform swap
                        for idx in covs_old:
                            count[idx] -= 1
                        for idx in block_to_indices[best_candidate]:
                            count[idx] += 1
                        blocks[i] = list(all_blocks[best_candidate])
                        block_indices[i] = best_candidate
                        
            else:
                # Try adding a block (increase energy) - only if it helps enable removals
                # This is rarely useful, skip for now
                pass
            
            # Cooling
            T *= alpha
            
            # Restore best if current is worse and T is low
            if n_blocks > best_len and T < 0.5:
                # Revert to best
                blocks = best_blocks[:]
                block_indices = [all_blocks.index(tuple(b)) for b in blocks]
                count = best_count.copy()
                n_blocks = best_len
        
        return best_blocks

    def local_search(blocks, rng, max_iter=50):
        """
        Local search: ILS-style perturbation with removal and re-addition.
        """
        if len(blocks) == 0:
            return blocks
        
        best_blocks = blocks[:]
        best_len = len(blocks)
        
        count = np.zeros(N_t, dtype=int)
        block_indices = [all_blocks.index(tuple(b)) for b in blocks]
        for bi in block_indices:
            for idx in block_to_indices[bi]:
                count[idx] += 1
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturbation: randomly remove 1-3 blocks
            n_remove = rng.randint(1, min(3, len(blocks)))
            indices_to_remove = rng.sample(range(len(blocks)), n_remove)
            
            for i in sorted(indices_to_remove, reverse=True):
                bi = block_indices[i]
                for idx in block_to_indices[bi]:
                    count[idx] -= 1
                blocks.pop(i)
                block_indices.pop(i)
            
            # Re-add blocks greedily
            for i in sorted(indices_to_remove, reverse=True):
                bi = block_indices[i] if i < len(block_indices) else all_blocks.index(tuple(blocks[i]))
                # This is getting messy, let's simplify
                pass
            
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
        for seed in range(6):
            if time.time() > deadline - 30:
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

    # Phase 2: Simulated annealing on each initial solution
    rng = random.Random(42)
    
    for i, blocks in enumerate(initial_solutions[:4]):
        if time.time() > deadline - 15:
            break
        
        polished = simulated_annealing(blocks, seed=42 + i, max_iter=150)
        if polished and len(polished) < best_len:
            best_len = len(polished)
            best_blocks = polished[:]
    
    # Phase 3: Final polish on best solution
    if best_blocks and time.time() < deadline - 10:
        final = simulated_annealing(best_blocks, seed=999, max_iter=100)
        if len(final) < best_len:
            best_blocks = final
    
    return best_blocks
