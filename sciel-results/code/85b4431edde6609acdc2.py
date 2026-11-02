import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    limit = 130.0  # Safety margin before 150s kill
    
    # Precompute t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Precompute all k-blocks and their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # For each block, compute which t-subsets it covers (as a set of indices)
    block_t_sets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index_map[combo])
        block_t_sets.append(covered)
    
    # Build inverse: for each t-subset, which blocks cover it
    t_to_blocks = defaultdict(set)
    for bi, covered in enumerate(block_t_sets):
        for ti in covered:
            t_to_blocks[ti].add(bi)
    
    def greedy_pass(rng, time_budget):
        """Greedy set cover with randomization. Returns list of block indices."""
        uncovered = set(range(num_t))
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() - start_time > time_budget:
                break
            
            best_idx = -1
            best_gain = -1
            best_tiebreak = float('inf')
            
            # Sample a subset of candidates to speed up
            candidates = list(selected_set) if len(selected_set) < num_k else []
            candidates = list(range(num_k))
            
            # For efficiency, iterate all but skip already selected
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = len(uncovered & block_t_sets[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_tiebreak = rng.random()
                elif gain == best_gain and gain > 0:
                    tb = rng.random()
                    if tb < best_tiebreak:
                        best_tiebreak = tb
                        best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered -= block_t_sets[best_idx]
        
        return selected
    
    def remove_redundant(selected, time_budget):
        """Remove redundant blocks from selected list."""
        if not selected:
            return selected
        
        selected_set = set(selected)
        # Build coverage count for each t-subset
        cover_count = defaultdict(int)
        for bi in selected_set:
            for ti in block_t_sets[bi]:
                cover_count[ti] += 1
        
        changed = True
        while changed and (time.time() - start_time < time_budget):
            changed = False
            new_selected = []
            for bi in selected:
                # Check if any t-subset is only covered by this block
                only_covered = False
                for ti in block_t_sets[bi]:
                    if cover_count[ti] == 1:
                        only_covered = True
                        break
                if only_covered:
                    new_selected.append(bi)
                else:
                    # Remove this block
                    for ti in block_t_sets[bi]:
                        cover_count[ti] -= 1
                    changed = True
            if changed:
                selected = new_selected
                selected_set = set(selected)
        
        return selected
    
    def local_search_polish(selected, time_budget):
        """Try to replace blocks to enable further redundancy removal."""
        if not selected:
            return selected
        
        selected_set = set(selected)
        best = list(selected)
        best_len = len(best)
        
        # Build coverage count
        cover_count = defaultdict(int)
        for bi in best:
            for ti in block_t_sets[bi]:
                cover_count[ti] += 1
        
        # Try removing each block and see if we can fix gaps with one addition
        for bi in list(best):
            if time.time() - start_time > time_budget:
                break
            
            # Remove bi
            for ti in block_t_sets[bi]:
                cover_count[ti] -= 1
            
            # Find uncovered t-subsets
            uncovered = {ti for ti in block_t_sets[bi] if cover_count[ti] == 0}
            if not uncovered:
                # bi was redundant, remove it
                best.remove(bi)
                best_len -= 1
                continue
            
            # Try to find a single block that covers all uncovered
            # This is expensive; sample candidates
            found = False
            candidates = [i for i in range(num_k) if i not in best_set]
            # Shuffle for randomness
            random.shuffle(candidates)
            for ci in candidates[:200]:  # Limit search
                if block_t_sets[ci] >= uncovered:
                    # Add ci
                    best.append(ci)
                    for ti in block_t_sets[ci]:
                        cover_count[ti] += 1
                    found = True
                    break
            
            if not found:
                # Restore bi
                for ti in block_t_sets[bi]:
                    cover_count[ti] += 1
        
        return best
    
    # Run multiple randomized restarts
    best_result = None
    best_len = float('inf')
    
    num_restarts = 8
    time_per_restart = (limit - 30.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > limit - 30.0:
            break
        
        rng = random.Random(restart * 7919 + 31)
        selected = greedy_pass(rng, limit - 30.0)
        selected = remove_redundant(selected, limit - 25.0)
        
        remaining_polish = min(10.0, max(2.0, limit - 20.0 - (time.time() - start_time)))
        selected = local_search_polish(selected, limit - 20.0)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
    
    # Final deterministic pass with remaining time
    if time.time() - start_time < limit - 5.0:
        rng = random.Random(42)
        selected = greedy_pass(rng, limit - 5.0)
        selected = remove_redundant(selected, limit - 3.0)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
    
    if best_result is None:
        return []
    
    return [list(k_blocks[i]) for i in best_result]
