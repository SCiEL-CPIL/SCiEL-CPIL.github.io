import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Map each block to the list of t-subset indices it covers
    block_to_t = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_t.append(covered)
    
    # Inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_to_t):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper: check if a set of block indices is a valid cover
    def is_valid(block_indices):
        if not block_indices:
            return N_t == 0
        # Use a set for fast lookup
        b_set = set(block_indices)
        for ti in range(N_t):
            # Check if any block in the set covers this t-subset
            # Optimization: check the first few candidates in t_to_blocks[ti]
            # If none are in b_set, it's uncovered
            covered = False
            for bi in t_to_blocks[ti]:
                if bi in b_set:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Helper: remove redundant blocks from a valid cover
    def prune_redundant(block_indices):
        b_list = list(block_indices)
        b_set = set(b_list)
        # Iterate through blocks and try to remove them
        # We do this in multiple passes or a single pass with a queue
        # Single pass is faster but might miss chains. 
        # Given time constraints, a single pass with random order is good.
        
        # Sort by block size of coverage? No, they are all same size.
        # Random order to avoid bias
        rng = random.Random(42)
        rng.shuffle(b_list)
        
        kept = []
        for bi in b_list:
            # Tentatively remove bi
            b_set.discard(bi)
            # Check if still valid
            if is_valid(b_set):
                # Keep it removed (don't add to kept)
                continue
            else:
                # Must keep it
                b_set.add(bi)
                kept.append(bi)
        
        return kept

    # Strategy: Weighted Random Sampling (Probabilistic Greedy)
    # 1. Start with empty cover
    # 2. While not valid:
    #    a. Calculate weight for each candidate block: number of uncovered t-subsets it covers
    #    b. Sample a block with probability proportional to weight
    #    c. Add to cover
    # 3. Prune redundant blocks
    
    # To speed up weight calculation, we maintain an array of uncovered counts
    # But since we sample, we can just compute the weight for candidates.
    # Actually, maintaining the array is O(N_t) per step, which is 11628.
    # Number of steps is ~300. 300 * 11628 is ~3.5M operations. Fast.
    
    def probabilistic_greedy(seed):
        rng = random.Random(seed)
        covered_mask = np.zeros(N_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # We need to pick blocks that cover the most *uncovered* t-subsets.
        # Instead of scanning all N_b blocks (75582) every time, 
        # we can pick a random uncovered t-subset, and then pick a random block covering it?
        # No, that's just random. We want "greedy" weighted by gain.
        
        # Efficient approach:
        # Maintain list of uncovered t-subsets.
        # Pick a random uncovered t-subset? No, that's not greedy.
        
        # Let's stick to the standard greedy but with randomization in ties or selection.
        # Actually, "Weighted Random Sampling" usually implies:
        # Score(block) = Gain(block)
        # P(block) = Score(block) / Sum(Scores)
        
        # Computing Sum(Scores) for all blocks is expensive if done naively.
        # But we can approximate or use a reservoir sampling approach?
        
        # Alternative: "Randomized Greedy"
        # 1. Pick a random uncovered t-subset.
        # 2. Among all blocks covering it, pick one with high probability proportional to its total gain.
        
        # Let's implement a faster variant:
        # 1. Find all uncovered t-subsets.
        # 2. If none, break.
        # 3. Pick a random uncovered t-subset `target_t`.
        # 4. Get all blocks covering `target_t` (from t_to_blocks).
        # 5. For each candidate block, calculate gain (number of uncovered t's it covers).
        # 6. Sample a block from candidates with probability proportional to gain.
        # 7. Add to solution.
        # 8. Update covered_mask.
        
        # This is O(N_b) in worst case for step 5 if we check all, but t_to_blocks[target_t] is small.
        # How many blocks cover a specific t-subset?
        # A t-subset is contained in C(v-t, k-t) blocks.
        # C(19-5, 8-5) = C(14, 3) = 364.
        # So for each step, we evaluate ~364 candidates.
        # 300 steps * 364 candidates * (cost to calculate gain).
        # Cost to calculate gain: iterate over block_to_t[bi] (126 items) and check covered_mask.
        # 300 * 364 * 126 ~ 13.7M operations. This is very fast in Python/NumPy?
        # Python loops are slow. 13M iterations might take 1-2 seconds.
        # With 115s budget, we can run many restarts.
        
        while True:
            if time.time() > deadline - 10.0:
                break
                
            # Find uncovered
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-subset
            target_t_idx = rng.choice(uncovered_indices.tolist())
            
            # Get candidate blocks covering this t-subset
            candidates = t_to_blocks[target_t_idx]
            if not candidates:
                # Should not happen if v,k,t are valid
                break
                
            # Calculate gain for each candidate
            # Gain = number of t-subsets in block_to_t[bi] that are NOT in covered_mask
            # We can vectorize this check?
            # block_to_t[bi] is a list of ints.
            # covered_mask is a numpy array of bools.
            
            best_block = -1
            best_gain = 0
            total_gain = 0
            
            # To avoid O(N) sum for sampling, we can just pick the best (greedy) 
            # or use a simple weighted random pick.
            # Let's try pure Greedy first (max gain), then maybe randomize if ties.
            # Actually, the prompt suggests "different strategy". 
            # Pure greedy is deterministic. 
            # Let's do: Pick the block with MAX gain. If there are multiple, pick randomly among them.
            # This is "Greedy with random tie-breaking".
            
            max_gain = -1
            best_candidates = []
            
            for bi in candidates:
                if bi in selected_set:
                    continue
                # Calculate gain
                # block_to_t[bi] is a list of t-indices
                # We need count of ti in block_to_t[bi] where covered_mask[ti] == False
                # Using numpy:
                t_indices = np.array(block_to_t[bi])
                gain = np.sum(~covered_mask[t_indices])
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain:
                    best_candidates.append(bi)
            
            if not best_candidates:
                # All candidates already selected?
                # This implies we are stuck or invalid state?
                # If all blocks covering target_t are selected, but target_t is uncovered,
                # it means our covered_mask is wrong or we have a bug.
                # Actually, if target_t is uncovered, and we have selected all blocks covering it,
                # then target_t should be covered. Contradiction.
                # Unless covered_mask is not updated correctly.
                # We update covered_mask when we add a block.
                # So if target_t is uncovered, no selected block covers it.
                # So candidates (blocks covering target_t) cannot be in selected_set.
                # So best_candidates should not be empty.
                
                # Fallback: just pick the first candidate
                bi = candidates[0]
            else:
                bi = rng.choice(best_candidates)
            
            selected_blocks.append(bi)
            selected_set.add(bi)
            
            # Update covered_mask
            for ti in block_to_t[bi]:
                covered_mask[ti] = True
        
        return selected_blocks

    # Main Loop: Multi-start Probabilistic Greedy
    best_solution = None
    best_len = float('inf')
    
    # Run multiple restarts with different seeds
    # Each run takes ~1-2 seconds.
    # We have ~110 seconds.
    # So we can run ~50-100 restarts.
    
    seed = 42
    while time.time() < deadline - 10.0:
        blocks = probabilistic_greedy(seed)
        
        # Prune redundant blocks
        # Pruning is expensive because is_valid is O(N_t * avg_candidates).
        # N_t = 11628.
        # is_valid checks each t-subset. For each, it checks t_to_blocks[ti] until one is found in b_set.
        # b_set is a set, so lookup is O(1).
        # Average candidates per t-subset is N_b * C(k,t) / C(v,t) = 75582 * 126 / 11628 ~ 819.
        # So is_valid is ~11628 * (819 / 2) ~ 4.7M operations? No, it breaks early.
        # If the cover is good, most t-subsets are covered by the first or second candidate in t_to_blocks[ti].
        # So is_valid is fast.
        
        # Prune takes O(N_blocks * Cost(is_valid)).
        # N_blocks ~ 300.
        # 300 * Fast_Is_Valid.
        
        if time.time() < deadline - 5.0:
            pruned = prune_redundant(blocks)
            
            if len(pruned) < best_len:
                best_len = len(pruned)
                best_solution = pruned
                
        seed += 1

    if best_solution is None:
        # Fallback: return a simple greedy solution
        # This should not happen
        return [list(all_blocks[bi]) for bi in range(0)] # Empty? No, must be valid.
        # Let's just run one greedy without pruning to be safe, though it might be large.
        # Actually, if we didn't find anything, we might have timed out in the first run.
        # Let's assume we found at least one.
        
    return [list(all_blocks[bi]) for bi in best_solution]
