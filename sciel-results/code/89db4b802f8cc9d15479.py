import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # All t-subsets (triples)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Represent each block as a set of triple indices
    # For v=24, k=7, t=3: n_subsets = 2024. 
    # We can fit 2024 bits into an array of 32 uint64s.
    # However, numpy bit_count on large arrays is tricky.
    # Alternative: Use a sparse representation or list of lists for the "masks" but vectorize the intersection?
    # Actually, for v=24, C(24,3)=2024. 
    # Let's use a list of sets for the blocks' covered triples.
    # To vectorize "gain", we need to know which triples are currently uncovered.
    # Let `uncovered` be a boolean array of size n_subsets.
    # For each block, we want count of (block_triples & uncovered).
    
    # Precompute block_triples as a list of lists of triple indices.
    # This is memory heavy but fast for iteration?
    # 346k blocks * 35 triples = 12M integers. ~100MB. Acceptable.
    
    block_triples = []
    for block in all_blocks:
        subs = [subset_index[sub] for sub in itertools.combinations(block, t)]
        block_triples.append(subs)
    
    # Convert to a numpy array? 
    # If we use a 2D numpy array, we need to pad to max length (35).
    # block_triples_np = np.full((n_blocks, 35), -1, dtype=np.int32)
    # for i, subs in enumerate(block_triples):
    #     block_triples_np[i, :len(subs)] = subs
    
    # A faster approach for "Gain":
    # Maintain `uncovered_mask` as a numpy boolean array (or int array).
    # For each step, we need to find the block with max sum(uncovered_mask[block_triples[i]]).
    # This is essentially a matrix-vector product if we had a binary matrix.
    # 346k x 2024 matrix is huge (700M entries). Too big for dense.
    
    # Let's stick to the list of lists but optimize the loop.
    # Or, use the "Bitmask" approach with Python integers?
    # 2024 bits fits in ~32 64-bit words.
    # Python int can handle 2024 bits natively and `bit_count` is fast.
    # But `uncovered_mask & block_mask` for 346k blocks in a Python loop is still ~0.1-0.2s per step.
    # 70 steps * 0.2s = 14s. This matches the current performance.
    
    # To go faster, we need to parallelize or reduce the number of blocks considered.
    # Strategy: "Diverse Greedy" with NumPy vectorization on a subset?
    # No, let's try a different structural change: **Simulated Annealing on the Complement**?
    # Or **Local Search with Efficient Swap Evaluation**?
    
    # Let's try the **Vectorized Gain** using NumPy on a *subset* of blocks?
    # No, we must check all to be sure.
    
    # Let's try **Integer Bitmasks** but with a **Reduced Candidate Set** per step.
    # Observation: In the early stages of greedy, many blocks have similar high gains.
    # We can pre-compute the "potential" of blocks.
    
    # Actually, let's look at the "Inspiration 3" which used NumPy bitmasks.
    # It scored 1.0909 (worse than 1.1020). Why? Maybe overhead.
    
    # Let's try a **Hybrid**: 
    # 1. Fast Greedy to get a solution (~80 blocks).
    # 2. Aggressive Local Search (Simulated Annealing) using **Precomputed Overlaps**.
    
    # Key Insight for Local Search:
    # To check if removing block B is valid, we need to know if any triple in B has count 1.
    # We can maintain `coverage_counts` (array of size 2024).
    # To check if swapping B_out with B_in is valid:
    #   - Triples in B_out but not B_in: count must be > 1.
    #   - Triples in B_in but not B_out: count increases (always valid).
    #   - Triples in both: unchanged.
    # So we only need to check `min(coverage_counts[t] for t in B_out if t not in B_in) > 1`.
    # This can be done quickly if we have the list of triples for B_out and B_in.
    
    # Let's implement a robust SA with this check.
    
    # 1. Initial Solution: Greedy with Random Tie-breaking
    def get_greedy_solution(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_subsets) - 1
        selected = []
        used = set()
        
        # Precompute masks for all blocks
        # This takes time but is done once.
        # 346k blocks * 35 combos.
        # Let's do it efficiently.
        
        # Actually, let's just use the `block_triples` list.
        # For gain, we iterate.
        
        while uncovered_mask:
            best_gain = -1
            best_candidates = []
            
            # Optimization: Skip blocks that are subsets of already selected? 
            # No, greedy doesn't work that way.
            
            for i, subs in enumerate(block_triples):
                if i in used:
                    continue
                # Calculate gain: number of bits in (uncovered_mask & mask_of_block_i)
                # We don't have mask_of_block_i as an int precomputed.
                # Let's precompute `block_masks_int` once.
                pass
            
            # This loop is the bottleneck.
            # Let's precompute `block_masks_int` as a list of Python ints.
            # 346k ints of ~250 bytes each = ~85MB. OK.
            
            if not hasattr(get_greedy_solution, 'masks'):
                masks = []
                for subs in block_triples:
                    m = 0
                    for s in subs:
                        m |= (1 << s)
                    masks.append(m)
                get_greedy_solution.masks = masks
            
            masks = get_greedy_solution.masks
            
            for i in range(n_blocks):
                if i in used:
                    continue
                gain = (uncovered_mask & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            
            # Random tie-breaking
            chosen = rng.choice(best_candidates)
            selected.append(chosen)
            used.add(chosen)
            uncovered_mask &= ~masks[chosen]
            
        return selected

    # 2. Local Search (SA)
    def local_search(init_blocks, seed, time_limit):
        rng = random.Random(seed)
        current = list(init_blocks)
        current_set = set(current)
        
        # Precompute masks for quick access
        masks = get_greedy_solution.masks
        triples_list = block_triples # list of lists of triple indices
        
        # Compute coverage counts
        counts = np.zeros(n_subsets, dtype=np.int32)
        for b in current:
            for t_idx in triples_list[b]:
                counts[t_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        T = 50.0
        T_min = 1.0
        cooling = 0.99
        
        while T > T_min and time.time() < time_limit:
            # Move: Swap
            # Pick random block to remove
            b_out = rng.choice(current)
            out_triples = triples_list[b_out]
            
            # Check if removal is valid (all counts > 1 for out_triples)
            # If valid, we can just remove (size -1). This is a "Delete" move.
            # If not valid, we must add a block that covers the "critical" triples.
            
            # Let's try a "Swap" move: Remove b_out, Add b_in.
            # We want b_in to cover the critical triples of b_out.
            
            # Identify critical triples in b_out (count == 1)
            critical = [t_idx for t_idx in out_triples if counts[t_idx] == 1]
            
            if not critical:
                # Can just delete
                # Remove b_out
                current.remove(b_out)
                current_set.remove(b_out)
                for t_idx in out_triples:
                    counts[t_idx] -= 1
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = list(current)
            else:
                # Need to find a block b_in that covers all critical triples
                # and is not in current_set.
                # To be efficient, pick a random critical triple and find blocks covering it?
                # Or just pick a random block from all_blocks and check?
                # Random pick is O(1) but low probability of covering all criticals if many.
                # Let's pick a random block and check.
                
                # To improve acceptance, we can bias b_in selection.
                # But for simplicity, let's try random swaps first.
                # If we find a valid swap, accept with prob exp(-delta/T) where delta=0 (size same).
                # Actually, if size is same, we accept to explore.
                
                b_in = rng.randrange(n_blocks)
                if b_in in current_set:
                    continue
                    
                in_triples = triples_list[b_in]
                
                # Check validity of swap:
                # 1. All critical triples of b_out must be covered by b_in.
                #    i.e., critical is a subset of in_triples.
                # 2. All triples in b_out that are NOT in b_in must have count > 1.
                #    (Because they lose coverage from b_out and don't gain from b_in)
                #    Actually, if a triple is in b_out and b_in, count is unchanged.
                #    If in b_out but not b_in, count decreases by 1. Must be > 1 originally.
                #    If in b_in but not b_out, count increases by 1. Always valid.
                
                # Check 1: Critical subset
                # Convert critical to a set for fast intersection?
                # critical is small (usually 1-3).
                in_set = set(in_triples)
                if not all(t in in_set for t in critical):
                    continue # b_in doesn't cover the criticals
                
                # Check 2: Non-critical removals
                # Triples in b_out that are NOT in b_in.
                # These are out_triples - in_triples.
                # For each such t, counts[t] must be > 1.
                valid = True
                for t_idx in out_triples:
                    if t_idx not in in_set:
                        if counts[t_idx] <= 1:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Valid Swap!
                # Perform swap
                current.remove(b_out)
                current_set.remove(b_out)
                for t_idx in out_triples:
                    counts[t_idx] -= 1
                
                current.append(b_in)
                current_set.add(b_in)
                for t_idx in in_triples:
                    counts[t_idx] += 1
                
                # Acceptance: Size is same. Accept with probability 0.5 to avoid cycles?
                # Or always accept to explore?
                if rng.random() < 0.5:
                    pass # Keep
                else:
                    # Revert
                    current.remove(b_in)
                    current_set.remove(b_in)
                    for t_idx in in_triples:
                        counts[t_idx] -= 1
                    
                    current.append(b_out)
                    current_set.add(b_out)
                    for t_idx in out_triples:
                        counts[t_idx] += 1
                
            T *= cooling
            
        return best_solution

    # Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Precompute masks (lazy init in get_greedy_solution)
    # Let's force init once
    _ = get_greedy_solution(0) # This will populate .masks if not exists? 
    # Wait, the lazy init inside the function might be slow on first call.
    # Let's do it here explicitly to be safe.
    if not hasattr(get_greedy_solution, 'masks'):
        masks = []
        for subs in block_triples:
            m = 0
            for s in subs:
                m |= (1 << s)
            masks.append(m)
        get_greedy_solution.masks = masks

    restart_id = 0
    while time.time() < deadline - 5.0:
        # 1. Greedy
        init = get_greedy_solution(restart_id)
        
        # 2. Local Search
        ls_limit = min(time.time() + 2.0, deadline - 2.0)
        refined = local_search(init, restart_id, ls_limit)
        
        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = [all_blocks[i] for i in refined]
        
        restart_id += 1
        
        # If we find 76, we can try to push for 75.
        # If we find 70, we are in good shape.
        
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
