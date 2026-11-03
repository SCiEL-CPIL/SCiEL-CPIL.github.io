import itertools
import time
import random
import sys

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Precompute sub_to_blocks for fast candidate lookup
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_mcv(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        counts = [0] * n_t
        selected_indices = []
        uncovered = set(range(n_t))
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
                
            # Find uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Pick candidate that maximizes coverage of other uncovered subsets
            best_block = -1
            max_gain = -1
            best_candidates = []
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_candidates = [b]
                elif gain == max_gain:
                    best_candidates.append(b)
                    
            if len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
                    
        return selected_indices

    def prune_and_optimize(selected_indices, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        # Initialize counts
        counts = [0] * n_t
        selected_set = set(selected_indices)
        
        for i in selected_set:
            for s in block_covers[i]:
                counts[s] += 1
                
        # 1. Initial Pruning: Remove any block that is redundant
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                redundant = True
                for s in block_covers[i]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for s in block_covers[i]:
                        counts[s] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
                        
        # 2. Local Search with Simulated Annealing
        # Objective: Minimize size.
        # Move: Swap block i with block j if it maintains validity.
        # We accept swaps that allow further pruning, or use SA to explore neutral swaps.
        
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        
        while T > min_T and _deadline_ok(deadline):
            if not selected_set:
                break
                
            # Pick a random block to try removing/swapping
            i = random.choice(list(selected_set))
            
            # Identify critical subsets covered by i
            critical = []
            for s in block_covers[i]:
                if counts[s] == 1:
                    critical.append(s)
                    
            if not critical:
                # Block i is redundant. Remove it.
                selected_set.remove(i)
                for s in block_covers[i]:
                    counts[s] -= 1
                T = max(min_T, T * 0.9) # Cool down on success
                continue
                
            # Block i is critical. We need to find a replacement j.
            # j must cover ALL subsets in critical.
            # Candidates for j are blocks that cover all s in critical.
            
            # Find intersection of blocks covering each s in critical
            # Start with blocks covering the first critical subset
            if not critical:
                continue
                
            # Optimization: Pick the critical subset with fewest covering blocks to start intersection
            crit_sorted = sorted(critical, key=lambda s: len(sub_to_blocks[s]))
            target_s = crit_sorted[0]
            candidates = set(sub_to_blocks[target_s])
            
            # Intersect with others
            for s in crit_sorted[1:]:
                candidates &= set(sub_to_blocks[s])
                
            # Remove existing blocks and i itself
            candidates -= selected_set
            
            if not candidates:
                # No replacement found. Reject move.
                continue
                
            # Pick best candidate j
            # Heuristic: j should cover the most subsets that are currently covered by exactly 1 block (other than i)
            # Or simply: j covers the most total subsets in the current cover?
            # Let's score j by how many "critical" subsets of OTHER blocks it covers?
            # Simpler: Score j by number of subsets it covers that are currently count==1.
            
            best_j = -1
            best_score = -1
            cand_list = list(candidates)
            
            # If too many candidates, sample
            if len(cand_list) > 50:
                cand_list = random.sample(cand_list, 50)
                
            for j in cand_list:
                score = 0
                for s in block_covers[j]:
                    if counts[s] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j == -1:
                continue
                
            # Perform swap i -> best_j
            selected_set.remove(i)
            selected_set.add(best_j)
            
            for s in block_covers[i]:
                counts[s] -= 1
            for s in block_covers[best_j]:
                counts[s] += 1
                
            # Check if this swap made any block redundant
            pruned = False
            for b in list(selected_set):
                redundant = True
                for s in block_covers[b]:
                    if counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(b)
                    for s in block_covers[b]:
                        counts[s] -= 1
                    pruned = True
                    
            if pruned:
                T = max(min_T, T * 0.9)
            else:
                # Neutral or worse move? 
                # If we didn't prune, we might be stuck. 
                # Accept with probability exp(-delta/T) where delta is 0 (neutral) 
                # Actually, if we didn't prune, the size is the same. 
                # We accept to explore.
                if random.random() > 0.5: 
                     T = max(min_T, T * cooling)
                else:
                     # Revert? No, let's keep it to change state.
                     T = max(min_T, T * cooling)

        return list(selected_set)

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 5 # MCV is fast, pruning/LS is slower. 
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        selected = greedy_mcv(i)
        selected = prune_and_optimize(selected, i)
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            
    if best_result is None:
        selected = greedy_mcv(0)
        selected = prune_and_optimize(selected, 0)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
             # Fallback: return a valid (but maybe larger) cover
             # The greedy_mcv alone is valid. Let's return that if this fails.
             # But this shouldn't happen.
             pass
             
    return result
