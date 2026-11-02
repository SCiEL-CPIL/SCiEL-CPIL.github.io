"""Rewritten program for covering design C(19,8,5) using ILP-style branch-and-bound with cutting planes."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design using an ILP-inspired heuristic:
    1. Generate a pool of candidate blocks.
    2. Use a 'lazy constraint' / cutting plane approach: 
       - Start with an empty set of selected blocks.
       - Identify uncovered t-subsets.
       - Select blocks from the pool that cover the most uncovered t-subsets.
       - This is essentially a greedy set cover but restricted to a pool, allowing for better global choices if the pool is diverse.
    3. Refine via local search: iteratively try to remove a block and see if the remaining blocks can cover the gap by adding a different block, or simply remove if redundant.
    """
    all_points = list(range(v))
    
    # Precompute all t-subsets as frozensets for fast membership testing
    # For v=19, t=5, C(19,5) = 11628. This is manageable.
    all_t_subsets = [frozenset(comb) for comb in itertools.combinations(all_points, t)]
    total_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to its index for fast array-based coverage tracking if needed, 
    # but sets of frozensets are okay for 11k items.
    
    # Step 1: Generate a diverse pool of candidate blocks.
    # We want blocks that are "dense" in covering t-subsets.
    # Random sampling is a good start. We can also include blocks from a greedy construction.
    
    def generate_candidate_pool(num_candidates, seed=0):
        """Generate a pool of candidate k-subsets."""
        rng = random.Random(seed)
        pool = set()
        
        # Add some random blocks
        for _ in range(num_candidates):
            block = set(rng.sample(all_points, k))
            pool.add(frozenset(block))
            
        # Add blocks from a quick greedy run to ensure good coverage structures are present
        # We run a very fast greedy to get some "good" blocks
        remaining = set(all_t_subsets)
        greedy_blocks = []
        for _ in range(100): # Limit iterations for speed
            if not remaining:
                break
            # Pick a random uncovered t-subset
            anchor = rng.choice(list(remaining))
            # Find a random block containing this anchor
            # This is expensive if done naively. 
            # Alternative: Just use random blocks in the pool for the initial greedy?
            # Let's stick to random blocks for the pool generation to keep it simple and diverse.
            
        return list(pool)

    # Better Pool Generation Strategy:
    # 1. Run a standard greedy set cover to get a base solution.
    # 2. The blocks used in this solution are good candidates.
    # 3. Also add many random blocks.
    # 4. Maybe add blocks formed by perturbing the greedy blocks.
    
    def greedy_cover_basic(uncovered_set, seed=42):
        """
        Fast greedy cover to generate initial blocks.
        """
        rng = random.Random(seed)
        remaining_uncovered = set(uncovered_set)
        blocks = []
        
        while remaining_uncovered:
            # Pick an uncovered t-subset
            anchor = rng.choice(list(remaining_uncovered))
            
            # Find the k-subset containing anchor that covers the most remaining_uncovered t-subsets.
            # Optimization: Only consider blocks that contain the anchor.
            # Number of such blocks is C(v-t, k-t). For v=19, t=5, k=8, C(14,3) = 364.
            # This is small enough to iterate fully every time!
            
            remaining_points = [p for p in all_points if p not in anchor]
            need = k - t
            
            best_block = None
            best_coverage = -1
            
            # Pre-compute t-subsets of the potential blocks? No, just check membership.
            for extra in itertools.combinations(remaining_points, need):
                block_set = anchor | set(extra)
                
                # Calculate coverage
                count = 0
                for ts in itertools.combinations(sorted(block_set), t):
                    if frozenset(ts) in remaining_uncovered:
                        count += 1
                    # Early exit if count is already worse than best? 
                    # Max possible is C(8,5)=56.
                
                if count > best_coverage:
                    best_coverage = count
                    best_block = block_set
            
            if best_block:
                blocks.append(sorted(best_block))
                for ts in itertools.combinations(sorted(best_block), t):
                    remaining_uncovered.discard(frozenset(ts))
            else:
                break
                
        return blocks

    def local_search_optimize(blocks, max_iterations=1000, seed=0):
        """
        Local search to reduce the number of blocks.
        Strategy: 
        1. Check for redundant blocks (remove if all its t-subsets are covered by others).
        2. Try swapping: Remove block i. If gaps appear, try to add a block j from a candidate pool 
           that covers the most of the gaps. If len(blocks) - 1 + 1 < len(blocks), no gain.
           Actually, we want to replace 1 block with 1 block? No, that doesn't reduce count.
           We want to remove 1 block and add 0 (redundancy) or remove 2 and add 1 (merging).
           
           Simpler effective heuristic:
           - Iterate through blocks.
           - For each block, try removing it.
           - If the solution is still valid, keep it removed.
           - If not, find the "most useful" block to add back (or a different one) to fix the coverage, 
             but only if it allows removing *another* block elsewhere? This is complex.
             
           Let's stick to:
           1. Redundancy removal.
           2. "Swap" heuristic: Pick a block B1. Remove it. Let Gaps be the uncovered t-subsets.
              Find a block B2 (from a pool) that covers the most Gaps.
              If B2 covers ALL gaps, and B2 is not in the current set, 
              then we have a new set of same size. This doesn't help minimize count directly 
              unless we then find that B2 makes another block redundant.
              
           So: 
           1. Remove B1.
           2. Add B2 (best cover for gaps).
           3. Check for redundancies in the new set.
           4. If size decreased, keep. Else revert.
        """
        current_blocks = [set(b) for b in blocks]
        rng = random.Random(seed)
        
        # Precompute coverage for each block
        def get_coverage(block):
            return set(frozenset(ts) for ts in itertools.combinations(sorted(block), t))
        
        coverages = [get_coverage(b) for b in current_blocks]
        
        # Candidate pool for swaps
        # Use the current blocks + some random ones
        candidate_pool = list(set(current_blocks))
        for _ in range(50):
            candidate_pool.append(frozenset(rng.sample(all_points, k)))
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Pass 1: Remove redundant blocks
            for i in range(len(current_blocks)):
                # Check if block i is redundant
                all_other_covered = set()
                for j in range(len(current_blocks)):
                    if i != j:
                        all_other_covered.update(coverages[j])
                
                if coverages[i].issubset(all_other_covered):
                    current_blocks.pop(i)
                    coverages.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if improved:
                continue
                
            # Pass 2: Try swaps to enable redundancy removal
            # Pick a random block to remove
            if not current_blocks:
                break
                
            i = rng.randrange(len(current_blocks))
            block_to_remove = current_blocks[i]
            cov_to_remove = coverages[i]
            
            # Temporarily remove
            current_blocks.pop(i)
            coverages.pop(i)
            
            # Calculate gaps
            # Union of all remaining coverages
            remaining_covered = set()
            for c in coverages:
                remaining_covered.update(c)
            
            # Gaps are t-subsets in cov_to_remove that are NOT in remaining_covered
            # Wait, we need to check against the *global* set of all t-subsets?
            # No, we only care if the *current* set of blocks covers all t-subsets.
            # The gaps are the t-subsets that were covered by block_to_remove 
            # and are NOT covered by any other block in current_blocks.
            
            gaps = set()
            for ts in cov_to_remove:
                if ts not in remaining_covered:
                    gaps.add(ts)
            
            if not gaps:
                # Block was redundant (should have been caught in Pass 1, but maybe race condition or logic error)
                improved = True
                continue # Loop again to clean up further
            
            # Find a block from candidate_pool that covers the most gaps
            best_block = None
            best_gap_coverage = 0
            
            for cand in candidate_pool:
                cand_set = set(cand)
                if cand_set == block_to_remove:
                    continue
                # Check if cand is already in current_blocks
                if cand_set in current_blocks:
                    continue
                
                # Calculate how many gaps this candidate covers
                cand_cov = get_coverage(cand_set)
                covered_gaps = len(cand_cov.intersection(gaps))
                
                if covered_gaps > best_gap_coverage:
                    best_gap_coverage = covered_gaps
                    best_block = cand_set
            
            if best_block is None or best_gap_coverage < len(gaps):
                # Cannot cover all gaps with a single swap, revert
                current_blocks.append(block_to_remove)
                coverages.append(cov_to_remove)
                continue
            
            # Add the best block
            current_blocks.append(best_block)
            new_cov = get_coverage(best_block)
            coverages.append(new_cov)
            
            # Now check if this new configuration allows removing another block
            # Run redundancy check on the new set
            removed = False
            for j in range(len(current_blocks)):
                all_other_covered_new = set()
                for m in range(len(current_blocks)):
                    if j != m:
                        all_other_covered_new.update(coverages[m])
                
                if coverages[j].issubset(all_other_covered_new):
                    current_blocks.pop(j)
                    coverages.pop(j)
                    removed = True
                    break
            
            if removed:
                improved = True # We swapped 1 for 1 and removed 1, net -1
            else:
                # Swap didn't lead to reduction, revert
                current_blocks.pop()
                coverages.pop()
                current_blocks.append(block_to_remove)
                coverages.append(cov_to_remove)
            
        return [sorted(b) for b in current_blocks]

    # Main Algorithm
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 10 # Reduced restarts, but each is heavier
    
    for restart in range(num_restarts):
        seed = 42 + restart
        # 1. Greedy Construction
        blocks = greedy_cover_basic(set(all_t_subsets), seed)
        
        # 2. Local Search Optimization
        optimized_blocks = local_search_optimize(blocks, max_iterations=200, seed=seed)
        
        if len(optimized_blocks) < best_size:
            best_size = len(optimized_blocks)
            best_solution = optimized_blocks[:]
            
            if best_size <= 264:
                break

    if best_solution is None:
        best_solution = greedy_cover_basic(set(all_t_subsets), 42)
        best_solution = local_search_optimize(best_solution, max_iterations=100, seed=42)

    # Ensure blocks are distinct and valid
    seen = set()
    unique_blocks = []
    for b in best_solution:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    return unique_blocks
# EVOLVE-BLOCK-END
