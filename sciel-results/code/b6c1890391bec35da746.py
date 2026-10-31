import time
import random
import itertools
import numpy as np

# Global cache for precomputed data
_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    # Map subset tuple to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Build coverage matrix using numpy for speed
    # Cover[i, j] = 1 if subset j is in block i
    Cover = np.zeros((n_blocks, n_subsets), dtype=np.uint8)
    
    for i, b in enumerate(all_blocks):
        for s in itertools.combinations(b, t):
            j = subset_to_idx[s]
            Cover[i, j] = 1
            
    _CACHE[key] = (all_blocks, Cover, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, Cover, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = np.ones(n_subsets, dtype=np.uint8)
    selected = []
    
    while np.sum(uncovered) > 0 and time.time() < deadline:
        # Compute gains: number of uncovered subsets each block covers
        # Using matrix-vector multiplication for speed
        gains = Cover @ uncovered
        
        best_idx = int(np.argmax(gains))
        
        if gains[best_idx] == 0:
            break
            
        selected.append(best_idx)
        # Update uncovered: set to 0 where the selected block covers
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    return selected

def _local_search(selected, Cover, n_subsets, deadline, rng):
    if not selected:
        return selected
    
    # Compute coverage counts
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in selected:
        counts += Cover[b_idx]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        new_selected = []
        for b_idx in selected:
            # Check if any subset in this block has count == 1
            # (i.e., this block is the only one covering it)
            critical_mask = (Cover[b_idx] == 1) & (counts == 1)
            if np.any(critical_mask):
                new_selected.append(b_idx)
            else:
                # Remove this block
                counts -= Cover[b_idx]
                improved = True
        selected = new_selected
        
        if time.time() > deadline:
            break
            
        # Phase 2: Try to replace blocks
        # For each block, see if another block can cover all its critical subsets
        for i in range(len(selected)):
            b_idx = selected[i]
            critical_mask = (Cover[b_idx] == 1) & (counts == 1)
            critical_idx = np.where(critical_mask)[0]
            
            if len(critical_idx) == 0:
                # This block is redundant, remove it
                counts -= Cover[b_idx]
                selected.pop(i)
                improved = True
                i -= 1
                continue
            
            # Find candidate blocks that cover all critical subsets
            # A block j is a candidate if Cover[j, critical_idx] are all 1
            # This is equivalent to: sum(Cover[:, critical_idx], axis=1) == len(critical_idx)
            
            # But we need to exclude the current block and already selected blocks
            # Let's compute scores for all blocks
            # score[j] = number of critical subsets covered by block j
            # We want score[j] == len(critical_idx)
            
            # To speed this up, we can just check a random subset of candidates
            # Or we can use the fact that we only care about blocks that cover
            # at least one critical subset.
            
            # Let's find all blocks that cover at least one critical subset
            # Then filter for those that cover ALL critical subsets
            
            # For efficiency, let's just try a few random replacements
            # We'll pick a random block from the full set and check if it's a valid replacement
            
            # Actually, a better approach:
            # 1. Find all blocks that cover the first critical subset
            # 2. Filter those to find blocks that cover ALL critical subsets
            
            first_crit = critical_idx[0]
            # Find all blocks that cover first_crit
            # This is: np.where(Cover[:, first_crit] == 1)[0]
            # But this is O(n_blocks) which is ~50k. Acceptable for a few iterations.
            
            # However, doing this for every block in every iteration might be slow.
            # Let's limit the number of replacement attempts per iteration.
            
            # Alternative: Just try to remove the block if it's redundant (done above).
            # For replacement, let's use a simpler heuristic:
            # Pick a random block, check if it covers all critical subsets of b_idx.
            # If yes, and it's not in selected, swap.
            
            # To make this more effective, let's try a few random candidates.
            
            for _ in range(3):
                cand_idx = rng.randrange(len(Cover))
                
                # Check if cand_idx covers all critical subsets
                if np.all(Cover[cand_idx, critical_idx] == 1):
                    # Check if cand_idx is not already in selected
                    if cand_idx not in selected:
                        # Perform swap
                        counts -= Cover[b_idx]
                        counts += Cover[cand_idx]
                        selected[i] = cand_idx
                        improved = True
                        break
                        
    return selected

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    
    all_blocks, Cover, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy
    cons_deadline = t0 + (deadline - t0) * 0.5
    selected = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    ls_deadline = deadline - 5.0
    selected = _local_search(selected, Cover, n_subsets, ls_deadline, rng)
    
    # Phase 3: Second Local Search Pass
    if time.time() < deadline - 10.0:
        selected = _local_search(selected, Cover, n_subsets, deadline - 5.0, rng)
        
    result = [list(all_blocks[i]) for i in selected]
    
    return result

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
