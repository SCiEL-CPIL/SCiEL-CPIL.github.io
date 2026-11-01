"""Combined covering design C(v,k,t) with bitmask greedy + swap local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t): a set of k-subsets (blocks) of {0..v-1}
    such that every t-subset is contained in at least one block.
    Minimize the number of blocks.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # 1. Precompute all t-subsets and assign an index to each
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute block-to-t-subset mapping
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)
    
    # For each block, store the list of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subset_index[ts])
        block_t_indices.append(indices)
    
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
    
    # Precompute degrees
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]
    
    def popcount(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    def run_greedy(seed, top_k=1, use_rarity=False):
        """Greedy set cover with optional rarity weighting."""
        rng = random.Random(seed)
        uncovered = [True] * n_t
        gains = [0] * n_blocks
        
        # Initialize gains
        if use_rarity:
            for b_idx in range(n_blocks):
                gain = 0.0
                for t_idx in block_t_indices[b_idx]:
                    deg = t_subset_degrees[t_idx]
                    if deg > 0:
                        gain += 1.0 / deg
                gains[b_idx] = gain
        else:
            for b_idx in range(n_blocks):
                gains[b_idx] = float(len(block_t_indices[b_idx]))
        
        selected = []
        selected_set = set()
        
        while True:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_indices = []
            
            for b_idx in range(n_blocks):
                if b_idx in selected_set:
                    continue
                if gains[b_idx] <= 0:
                    continue
                if gains[b_idx] > best_gain:
                    best_gain = gains[b_idx]
                    best_indices = [b_idx]
                elif gains[b_idx] == best_gain:
                    best_indices.append(b_idx)
            
            if not best_indices or best_gain <= 0:
                break
            
            if top_k > 1 and len(best_indices) > top_k:
                chosen = rng.choice(best_indices[:top_k])
            else:
                chosen = rng.choice(best_indices)
            
            selected.append(chosen)
            selected_set.add(chosen)
            
            # Update gains
            for t_idx in block_t_indices[chosen]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        if b_idx not in selected_set:
                            if use_rarity:
                                deg = t_subset_degrees[t_idx]
                                if deg > 0:
                                    gains[b_idx] -= 1.0 / deg
                            else:
                                gains[b_idx] -= 1.0
        
        return selected
    
    def minimize(blocks_list, seed, max_rounds=20):
        """Remove redundant blocks through multiple passes."""
        rng = random.Random(seed)
        active = set(blocks_list)
        
        # Compute coverage counts
        cover_count = [0] * n_t
        for b_idx in active:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        changed = True
        rounds = 0
        while changed and rounds < max_rounds and time.time() < deadline - 15.0:
            changed = False
            rounds += 1
            items = list(active)
            rng.shuffle(items)
            for b_idx in items:
                if b_idx not in active:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return list(active), cover_count
    
    def local_search_swap(blocks_list, seed, max_iters=30):
        """Local search with block swaps to find better solutions."""
        rng = random.Random(seed)
        active = set(blocks_list)
        
        # Compute coverage counts
        cover_count = [0] * n_t
        for b_idx in active:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 20.0:
            improved = False
            iters += 1
            current_blocks = list(active)
            rng.shuffle(current_blocks)
            
            for B in current_blocks:
                if B not in active:
                    continue
                
                # Find t-subsets uniquely covered by B
                unique_to_B = []
                for t_idx in block_t_indices[B]:
                    if cover_count[t_idx] == 1:
                        unique_to_B.append(t_idx)
                
                if not unique_to_B:
                    # B is redundant, remove it
                    active.remove(B)
                    for t_idx in block_t_indices[B]:
                        cover_count[t_idx] -= 1
                    improved = True
                    continue
                
                # Find candidates that cover all unique t-subsets
                candidates = None
                for t_idx in unique_to_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter: must not be in active, must be different from B
                valid_candidates = [c for c in candidates if c != B and c not in active]
                
                if not valid_candidates:
                    continue
                
                # Score candidates by how many additional t-subsets they cover
                best_candidate = None
                best_score = -1
                for c in valid_candidates[:20]:  # Limit to top 20 for speed
                    score = 0
                    for t_idx in block_t_indices[c]:
                        if cover_count[t_idx] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                
                if best_candidate is None:
                    continue
                
                # Perform swap
                active.remove(B)
                active.add(best_candidate)
                
                for t_idx in block_t_indices[B]:
                    cover_count[t_idx] -= 1
                for t_idx in block_t_indices[best_candidate]:
                    cover_count[t_idx] += 1
                
                improved = True
        
        return list(active), cover_count
    
    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy runs with different strategies
    greedy_configs = [
        (0, 1, False),
        (1, 1, False),
        (2, 2, False),
        (3, 3, False),
        (4, 1, True),
        (5, 2, True),
        (6, 5, False),
        (7, 1, True),
    ]
    
    for seed, top_k, use_rarity in greedy_configs:
        if time.time() > deadline - 30.0:
            break
        
        res = run_greedy(seed, top_k=top_k, use_rarity=use_rarity)
        if res:
            # Minimize
            minimized, _ = minimize(res, seed + 1000, max_rounds=15)
            
            # Local search with swaps
            searched, _ = local_search_swap(minimized, seed + 2000, max_iters=25)
            
            # Final minimization after swaps
            final, _ = minimize(searched, seed + 3000, max_rounds=10)
            
            if len(final) < best_count:
                best_count = len(final)
                best_blocks = final
    
    # Phase 2: If time remains, do more diverse searches
    if best_blocks is not None and time.time() < deadline - 40.0:
        extra_seeds = list(range(10, 25))
        rng_master = random.Random(42)
        rng_master.shuffle(extra_seeds)
        
        for seed in extra_seeds:
            if time.time() > deadline - 25.0:
                break
            
            top_k = rng_master.choice([1, 2, 3, 5])
            use_rarity = rng_master.choice([True, False])
            
            res = run_greedy(seed, top_k=top_k, use_rarity=use_rarity)
            if res:
                minimized, _ = minimize(res, seed + 1000, max_rounds=12)
                searched, _ = local_search_swap(minimized, seed + 2000, max_iters=20)
                final, _ = minimize(searched, seed + 3000, max_rounds=8)
                
                if len(final) < best_count:
                    best_count = len(final)
                    best_blocks = final
    
    if best_blocks is None:
        return []
    
    return [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
# EVOLVE-BLOCK-END
