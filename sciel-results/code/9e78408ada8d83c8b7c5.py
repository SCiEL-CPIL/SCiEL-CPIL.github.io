import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute for each block:
    # 1. The list of triple indices it covers
    # 2. A mapping from point to list of blocks containing that point (for efficient neighbor search)
    block_triple_indices = [[] for _ in range(num_blocks)]
    point_to_blocks = [[] for _ in range(v)]
    
    for i, b in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
        block_triple_indices[i] = indices
        for p in b:
            point_to_blocks[p].append(i)

    # Helper to calculate gain of adding a block given current coverage counts
    # Returns number of new triples covered
    def calc_gain(block_idx, coverage_counts):
        gain = 0
        for tri_idx in block_triple_indices[block_idx]:
            if coverage_counts[tri_idx] == 0:
                gain += 1
        return gain

    # Helper to update coverage counts when adding/removing a block
    def update_coverage(coverage_counts, block_idx, sign):
        for tri_idx in block_triple_indices[block_idx]:
            coverage_counts[tri_idx] += sign

    # 1. Initial Construction: Greedy
    # We use a fast greedy to get a valid starting point.
    # Since we are changing the paradigm, we use a simple randomized greedy to get diverse starts.
    
    def greedy_construct(random_seed):
        rng = random.Random(random_seed)
        # Shuffle blocks to break ties and introduce randomness in selection order
        # But standard greedy picks max gain. To make it faster and simpler for init:
        # We'll pick the block with max gain, breaking ties randomly.
        
        coverage_counts = [0] * num_triples
        selected = []
        
        # Optimization: Pre-compute initial gains? No, dynamic.
        # To speed up, we can just iterate.
        
        while True:
            # Check if all covered
            # min(coverage_counts) > 0?
            # This is O(num_triples). 2024 items. Fast.
            if min(coverage_counts) > 0:
                break
            
            if time.time() > deadline - 5:
                break # Time out, return what we have (might be invalid, but greedy usually finishes fast)
                
            max_gain = -1
            candidates = []
            
            # Scan all blocks
            for i in range(num_blocks):
                gain = calc_gain(i, coverage_counts)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break # Should not happen if problem is solvable, but safety
            
            # Pick random candidate
            b_idx = rng.choice(candidates)
            selected.append(b_idx)
            update_coverage(coverage_counts, b_idx, 1)
            
        return selected

    def is_valid(coverage_counts):
        return all(c > 0 for c in coverage_counts)

    def prune_solution(block_indices):
        """
        Removes redundant blocks.
        """
        if not block_indices:
            return []
        
        coverage_counts = [0] * num_triples
        for b_idx in block_indices:
            update_coverage(coverage_counts, b_idx, 1)
        
        current_set = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_set)):
                b_idx = current_set[i]
                redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if coverage_counts[tri_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    current_set.pop(i)
                    update_coverage(coverage_counts, b_idx, -1)
                    changed = True
                    if time.time() > deadline - 10:
                        return current_set
                    break
        
        return current_set

    best_solution = None
    best_size = float('inf')
    
    # 2. Simulated Annealing
    # Objective: Minimize number of blocks.
    # State: Set of block indices.
    # Move: Remove a block B1, Add a block B2.
    # Constraint: The resulting set must still be a valid covering.
    # Cost: |Set|. Delta = +1 (if size increases) or -1 (if size decreases, which is impossible for a single swap unless we allow multi-swap, but usually we remove 1 add 1 so size stays same? 
    # Wait. If we remove B1 and add B2, size stays same. 
    # To reduce size, we must find a move that allows removal without replacement? 
    # Or, we can define the move as: 
    # 1. Remove B1. If still valid, accept (size -1).
    # 2. If not valid, we need to add B2 to fix it. If we add B2, size is same. 
    # This doesn't reduce size.
    
    # Correct SA formulation for Set Cover:
    # State: Current set S.
    # Move 1 (Removal): Pick b in S. Remove b. If valid, accept (cost -1). If invalid, reject (or use in swap).
    # Move 2 (Swap): Pick b in S to remove. Let U be the set of triples uncovered by removing b. 
    #                Pick b' not in S such that b' covers all triples in U. 
    #                Replace b with b'. Size unchanged.
    #                This allows moving to a different local minimum.
    
    # We want to minimize size. 
    # If we only do swaps, we never reduce size. 
    # We must try removals.
    
    # Strategy:
    # 1. Start with greedy solution.
    # 2. Loop:
    #    a. Try random removal. If valid, accept. (Size decreases)
    #    b. If invalid, try swap. Pick a random block to remove. Find a block that covers the "holes". 
    #       If found, accept with probability based on temperature? 
    #       Actually, swaps keep size same. Why do we need them? To escape local minima where no single removal works, but a sequence of swaps leads to a state where a removal works.
    
    # Let's implement this.
    
    def simulated_annealing(initial_blocks, seed):
        rng = random.Random(seed)
        current_blocks = set(initial_blocks)
        coverage_counts = [0] * num_triples
        for b in current_blocks:
            update_coverage(coverage_counts, b, 1)
        
        # Ensure valid start
        if not is_valid(coverage_counts):
            return None, float('inf')
            
        current_size = len(current_blocks)
        best_size_sa = current_size
        best_blocks_sa = set(current_blocks)
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        # Time budget per SA run
        run_deadline = time.time() + 10 # 10 seconds per run
        
        while time.time() < run_deadline:
            # Try Removal
            if current_blocks:
                # Pick random block
                b_remove = rng.choice(list(current_blocks))
                # Check if removal is valid
                # Count how many triples would become 0
                holes = []
                for tri_idx in block_triple_indices[b_remove]:
                    if coverage_counts[tri_idx] == 1:
                        holes.append(tri_idx)
                
                if not holes:
                    # Valid removal
                    current_blocks.remove(b_remove)
                    update_coverage(coverage_counts, b_remove, -1)
                    current_size -= 1
                    if current_size < best_size_sa:
                        best_size_sa = current_size
                        best_blocks_sa = set(current_blocks)
                else:
                    # Invalid removal. Try Swap.
                    # We need to find a block b_add that covers all 'holes'.
                    # 'holes' are triples that are ONLY covered by b_remove.
                    # b_add must contain all points in these triples? No, b_add must be a block that covers these specific triple indices.
                    
                    # Optimization: 
                    # If len(holes) > C(k, t), impossible.
                    # Also, if len(holes) is large, it's rare.
                    
                    # Find candidate b_add.
                    # Iterate through blocks not in current_blocks.
                    # To speed up, we can restrict candidates to blocks that intersect the points in holes?
                    # Actually, b_add must cover specific triples.
                    
                    # Simple approach: Scan all blocks not in current_blocks.
                    # This is O(N_blocks * C(k,t)).
                    # 346k * 35 ~ 12M ops. Might be slow if done every step.
                    # We can limit swaps to every X steps or only if T is high?
                    # Or, we can just try removals for a while, and if stuck, do swaps.
                    
                    # Let's try a swap only if we are stuck (no removals for a while) or randomly with prob 0.1.
                    if rng.random() < 0.2:
                        # Find a block that covers all holes
                        found = False
                        # To make it faster, we can sample a few blocks
                        # Or just scan. Let's scan but break early if time is tight.
                        # Actually, scanning 346k blocks for every swap is expensive.
                        # Let's restrict candidates to blocks that contain at least one point from the first hole triple.
                        # This reduces candidates significantly.
                        
                        first_hole = holes[0]
                        # Get the triple
                        tri = all_triples[first_hole]
                        # Candidates must contain all points in tri? No, just cover the triple.
                        # So b_add must be a superset of tri? No, b_add is a k-subset. It covers tri if tri is a subset of b_add.
                        # So b_add must contain all points in tri.
                        
                        # Find blocks containing tri.
                        # We can precompute triple_to_blocks?
                        # C(24,3) = 2024 triples.
                        # Each block covers C(7,3)=35 triples.
                        # Total triples covered = 346k * 35 ~ 12M.
                        # Average blocks per triple = 12M / 2024 ~ 6000.
                        # So for each hole, there are ~6000 candidate blocks.
                        # Intersection of candidate sets for all holes.
                        
                        # Let's build a list of blocks covering the first hole.
                        # We don't have triple_to_blocks precomputed. Let's do it on the fly or precompute.
                        # Precomputing triple_to_blocks:
                        # triple_to_blocks = [set() for _ in range(num_triples)]
                        # for i, b in enumerate(all_blocks):
                        #   for tri_idx in block_triple_indices[i]:
                        #     triple_to_blocks[tri_idx].add(i)
                        # Memory: 2024 sets of ~6000 ints. 12M ints. ~100MB? Maybe too much.
                        # Let's not precompute.
                        
                        # Instead, let's just try a few random blocks from the whole set?
                        # No, that's unlikely to cover specific holes.
                        
                        # Alternative: 
                        # If we can't do swaps efficiently, we rely on multiple restarts of Greedy + Removal.
                        # But the prompt asks for a structural change. SA with swaps is the structural change.
                        # Let's implement the swap search efficiently.
                        
                        # Get points in the first hole
                        p1, p2, p3 = tri
                        # Candidates must contain p1, p2, p3.
                        # We can intersect point_to_blocks[p1], point_to_blocks[p2], point_to_blocks[p3].
                        # These lists are size C(23, 6) ~ 100k?
                        # C(23, 6) = 100947.
                        # Intersection of 3 lists of 100k.
                        # Convert to sets? 
                        # set(point_to_blocks[p1]) & set(point_to_blocks[p2]) & set(point_to_blocks[p3])
                        # This is fast in Python if lists are not too huge.
                        
                        candidates = set(point_to_blocks[p1])
                        candidates.intersection_update(point_to_blocks[p2])
                        candidates.intersection_update(point_to_blocks[p3])
                        
                        # Remove current blocks
                        candidates -= current_blocks
                        
                        # Now check which of these candidates cover ALL holes
                        # Iterate through candidates
                        for b_add in candidates:
                            # Check if b_add covers all holes
                            valid_swap = True
                            for h_idx in holes:
                                if coverage_counts[h_idx] == 1: # It's a hole
                                    # Check if b_add covers this triple
                                    # We need to know if triple h_idx is in block_triple_indices[b_add]
                                    # This is O(C(k,t)) check.
                                    # We can precompute a set of triple indices for each block?
                                    # block_triple_sets = [set(indices) for indices in block_triple_indices]
                                    # Memory: 346k sets of 35 ints. ~12M ints. ~100MB.
                                    # Acceptable? Maybe.
                                    # Let's check memory. 12M * 28 bytes (Python int) ~ 336MB. Too much.
                                    # Let's use the list and do a binary search or just linear scan?
                                    # Linear scan of 35 items is fast.
                                    
                                    if h_idx not in block_triple_indices[b_add]:
                                        valid_swap = False
                                        break
                            
                            if valid_swap:
                                # Perform swap
                                current_blocks.remove(b_remove)
                                current_blocks.add(b_add)
                                update_coverage(coverage_counts, b_remove, -1)
                                update_coverage(coverage_counts, b_add, 1)
                                # Size unchanged
                                found = True
                                break
                        
                        if not found:
                            # No swap found, do nothing (or reject)
                            pass
            
            # Cool down
            T *= cooling
            if T < min_T:
                break
        
        return list(best_blocks_sa), best_size_sa

    # Run multiple SA instances
    num_starts = 10
    for seed in range(num_starts):
        if time.time() > deadline - 15:
            break
            
        # Get initial solution
        init_blocks = greedy_construct(seed)
        if not init_blocks:
            continue
            
        # Prune initial
        init_blocks = prune_solution(init_blocks)
        
        # Run SA
        sa_blocks, sa_size = simulated_annealing(init_blocks, seed)
        
        if sa_blocks and sa_size < best_size:
            best_solution = sa_blocks
            best_size = sa_size
            
            if best_size <= 69:
                break

    if best_solution is None:
        # Fallback: Return the best greedy result we found during the loop? 
        # We didn't store it. Let's just run one greedy and prune.
        init_blocks = greedy_construct(0)
        best_solution = prune_solution(init_blocks)
        if not best_solution:
             # Absolute fallback
             return [all_blocks[i] for i in range(num_blocks)]

    # Convert indices to blocks
    result_blocks = [all_blocks[i] for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    pass
