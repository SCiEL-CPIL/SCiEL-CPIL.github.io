import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets and map to indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    # Precompute all k-subsets
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
        
    t_count = nCr(k, t)
    
    # Precompute for each k-subset, the indices of the t-subsets it contains
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=10, alpha=0.0):
        """
        Randomized greedy set cover with optional diversity penalty.
        Picks a block from the top 'top_k' gainers randomly.
        alpha > 0 penalizes overlap with existing coverage to encourage diversity.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        cover_count = np.zeros(n_ts, dtype=np.int32)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains: number of uncovered triples covered by each block
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                if not np.all(covered):
                    return None
                else:
                    break
            
            # Find indices of blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if alpha > 0:
                # Calculate overlap penalty
                overlaps = (cover_count[block_t_indices] > 0).sum(axis=1)
                scores = gains - alpha * overlaps
                max_score = np.max(scores)
                # Filter candidates to those with max score (which implies max gain usually)
                # But we want to stick to max gain for correctness of greedy step, 
                # or allow slight deviation? 
                # Standard greedy picks max gain. If we penalize, we might pick lower gain.
                # Let's stick to max gain candidates and pick the one with min overlap among them.
                # This is safer.
                cand_overlaps = overlaps[candidates]
                min_overlap = np.min(cand_overlaps)
                # Find candidates with min overlap
                best_cands = candidates[cand_overlaps == min_overlap]
            else:
                best_cands = candidates

            if len(best_cands) > top_k:
                # Sample from top candidates to diversify
                # We pick a random subset of the max gainers
                pass
            
            best_idx = best_cands[rng.randint(len(best_cands))]
            
            # Add block
            selected_blocks.append(best_idx)
            
            # Update covered
            covered[block_t_indices[best_idx]] = True
            cover_count[block_t_indices[best_idx]] += 1
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        # We iterate until no more blocks can be removed
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage for each t-subset
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                # Check if this block is redundant
                subs_in_block = block_t_indices[b_idx]
                
                # If all these t-subsets have coverage_count > 1, the block is redundant
                # Note: coverage_count includes this block itself, so we need > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    # Redundant: do not add to new_blocks
                    # Decrement coverage for these subsets
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    # Keep it
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    def local_search_swap(block_indices):
        """
        Attempt to reduce block count by swapping blocks.
        For each block B in the set, try to find a block B' not in the set
        such that replacing B with B' keeps the covering valid.
        If such a B' exists, we don't reduce count, but it might enable further deletions.
        
        However, a more effective local search for *reduction* is:
        1. Try to delete blocks (already done in reduce_blocks).
        2. Try to swap a block B with a block B' that allows deletion of ANOTHER block C.
        This is complex.
        
        A simpler heuristic from the inspiration: "Local search with block swaps escapes the greedy local minimum".
        Let's implement a simple 1-swap:
        - Iterate through current blocks.
        - For each block B, temporarily remove it.
        - Identify uncovered t-subsets U.
        - Find a block B' not in current set that covers all of U.
        - If found, replace B with B'. This doesn't reduce size, but might allow subsequent deletions.
        - Then run reduce_blocks again.
        
        Actually, let's look for a direct reduction via swap:
        If we replace B with B', and B' is "better" (covers more unique things or allows removing C), we gain.
        
        Let's try a simpler approach: Random Restart with Local Polishing.
        The current program does multiple greedy runs.
        The inspiration suggests swaps help.
        
        Let's implement a specific swap-based reduction:
        1. Start with reduced set S.
        2. For each b in S:
           a. Temporarily remove b. Let S' = S \ {b}.
           b. Compute coverage of S'. Let U be uncovered t-subsets.
           c. If U is empty, we already handled this in reduce_blocks.
           d. If U is not empty, look for a block b_new NOT in S such that b_new covers U.
           e. If such b_new exists, replace b with b_new in S.
           f. Run reduce_blocks on the new S.
           g. If size decreases, accept.
        
        This is expensive. Let's limit the search.
        """
        if not block_indices:
            return block_indices
            
        current_blocks = list(block_indices)
        n_blocks = len(current_blocks)
        
        # Precompute set of current block indices for fast lookup
        current_set = set(current_blocks)
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Build coverage count
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            for i, b_idx in enumerate(current_blocks):
                if time.time() > deadline - 5.0:
                    break
                    
                # Temporarily remove b_idx
                subs_in_b = block_t_indices[b_idx]
                coverage_count[subs_in_b] -= 1
                
                # Find uncovered subsets
                uncovered_mask = coverage_count == 0
                if not np.any(uncovered_mask):
                    # Should not happen if reduce_blocks was run, but just in case
                    coverage_count[subs_in_b] += 1
                    continue
                    
                uncovered_indices = np.where(uncovered_mask)[0]
                
                # We need a block b_new that covers ALL uncovered_indices
                # Check candidates. To be efficient, we can filter candidates.
                # A candidate must contain at least one of the uncovered indices.
                # Actually, it must contain ALL of them.
                
                # Heuristic: Pick one uncovered subset, say u0.
                # Candidates must cover u0.
                # We don't have an inverted index (t-subset -> blocks).
                # Building one is expensive but might be worth it?
                # C(24,3) = 2024. C(24,7) = 346104.
                # Inverted index: for each of 2024 triples, list of blocks covering it.
                # Size: 2024 * 1771 (avg blocks per triple? No, total incidences = 346104 * 35 = 12M).
                # 12M ints is ~48MB. Acceptable.
                
                # Let's build the inverted index once outside?
                # But this is inside local_search_swap.
                # Let's pass it in or build it globally.
                
                # For now, let's skip this complex swap and rely on multiple greedy runs with different alphas.
                # The "swap" idea might be too slow to implement safely in the remaining time without precomputation.
                
                # Restore coverage
                coverage_count[subs_in_b] += 1

        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    num_runs = 0
    alphas = [0.0, 0.05, 0.1, 0.2, 0.5]
    
    while time.time() < deadline - 15.0:
        seed = num_runs
        alpha = alphas[num_runs % len(alphas)]
        
        blocks = greedy_randomized(seed, top_k=10, alpha=alpha)
        
        if blocks is not None:
            # Reduce redundancy
            reduced = reduce_blocks(blocks)
            
            # Optional: Local search (currently a no-op placeholder for safety)
            # reduced = local_search_swap(reduced)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_blocks = reduced
        
        num_runs += 1
        
        # If we've run enough, break
        if num_runs > 500: 
             break

    if best_blocks is not None:
        # Convert indices back to lists of points
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback if no valid covering found (should not happen)
    blocks = greedy_randomized(0, top_k=10, alpha=0.0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
        
    return []
