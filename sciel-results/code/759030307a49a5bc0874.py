import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    t_subs = list(itertools.combinations(points, t))
    n_t = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    block_combinations = list(itertools.combinations(points, k))
    n_blocks = len(block_combinations)
    
    # Precompute t-subsets for each block
    block_t_subs = []
    for block in block_combinations:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_subs.append(subs)
        
    # Precompute which blocks cover each t-subset
    t_sub_blocks = [[] for _ in range(n_t)]
    for i, subs in enumerate(block_t_subs):
        for s in subs:
            t_sub_blocks[s].append(i)

    # Greedy Initialization
    def greedy_init():
        uncovered = set(range(n_t))
        selected = []
        selected_set = set()
        while uncovered:
            best_block = -1
            best_gain = -1
            for i in range(n_blocks):
                if i in selected_set:
                    continue
                gain = 0
                for s in block_t_subs[i]:
                    if s in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            if best_block == -1:
                break
            selected.append(best_block)
            selected_set.add(best_block)
            for s in block_t_subs[best_block]:
                if s in uncovered:
                    uncovered.remove(s)
        return selected

    initial_blocks = greedy_init()
    
    # Local Search State
    current_blocks = initial_blocks[:]
    current_counts = [0] * n_t
    for b in current_blocks:
        for s in block_t_subs[b]:
            current_counts[s] += 1
            
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    start_time = time.time()
    timeout = 115.0
    rng = random.Random(42)
    
    # Optimization: Cache block_t_subs for fast access
    # Already done.
    
    while time.time() - start_time < timeout:
        # Phase 1: Remove redundant blocks
        improved = True
        while improved:
            improved = False
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                # Check if redundant
                redundant = True
                for s in block_t_subs[b]:
                    if current_counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_t_subs[b]:
                        current_counts[s] -= 1
                    current_blocks.pop(i)
                    improved = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = current_blocks[:]
                    break # Restart loop as indices shifted
                
        # Phase 2: Try swaps to enable further removals
        if not current_blocks:
            break
            
        # Try a few random swaps
        for _ in range(10): # Try 10 random swaps per iteration
            b_rem_idx = rng.randrange(len(current_blocks))
            b_rem = current_blocks[b_rem_idx]
            
            # Calculate newly uncovered
            newly_uncovered = []
            for s in block_t_subs[b_rem]:
                if current_counts[s] == 1:
                    newly_uncovered.append(s)
                    
            if not newly_uncovered:
                # Redundant, should have been caught in Phase 1
                # But maybe Phase 1 missed it due to break?
                # No, Phase 1 breaks and restarts.
                # So this case shouldn't happen if Phase 1 is thorough.
                # But just in case:
                for s in block_t_subs[b_rem]:
                    current_counts[s] -= 1
                current_blocks.pop(b_rem_idx)
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                continue

            # Find best repair
            current_block_set = set(current_blocks)
            scores = {}
            for s in newly_uncovered:
                for b in t_sub_blocks[s]:
                    if b not in current_block_set:
                        scores[b] = scores.get(b, 0) + 1
            
            if not scores:
                continue
                
            # Find best b
            best_b = -1
            max_score = 0
            for b, sc in scores.items():
                if sc > max_score:
                    max_score = sc
                    best_b = b
                    
            if best_b == -1:
                continue
                
            # Check if best_b covers all newly_uncovered
            if max_score == len(newly_uncovered):
                # Valid swap
                # Apply
                for s in block_t_subs[b_rem]:
                    current_counts[s] -= 1
                for s in block_t_subs[best_b]:
                    current_counts[s] += 1
                current_blocks[b_rem_idx] = best_b
                
                # Note: We don't update best_size here because size didn't change.
                # But this new configuration might allow a removal in the next Phase 1.
            
    return [list(block_combinations[idx]) for idx in best_blocks]
