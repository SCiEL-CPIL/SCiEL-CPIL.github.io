import itertools
import random
import time
import sys
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset the points it contains
    tsub_points = np.array(all_tsubs, dtype=np.int8)
    
    # Precompute for each point, which t-subsets it belongs to (inverse index)
    # This is crucial for fast gain calculation
    point_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)
    
    # Convert to numpy arrays for speed
    point_tsub_arrays = {p: np.array(idx_list, dtype=np.int32) for p, idx_list in point_to_tsubs.items()}
    
    def greedy_build(seed):
        """Vectorized randomized greedy covering."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.bool_)
        blocks = []
        
        # Random order of t-subsets to pick
        tsub_order = np.arange(n_tsubs)
        rng.shuffle(tsub_order)
        
        # We need to pick t-subsets one by one.
        # To make it fast, we can maintain a list of uncovered indices and shuffle it?
        # Or just iterate and check?
        # With n_tsubs ~ 3876, iterating is fine.
        
        # Optimization: Keep a set of uncovered indices?
        # Or just a boolean array and a list of remaining?
        
        # Let's use a list of uncovered indices and pop from it.
        # But we need to respect the random order?
        # Actually, standard greedy picks the "best" block for the current "worst" uncovered t-subset?
        # Or does it pick an uncovered t-subset, then build a block around it?
        # The current code picks the first uncovered in a random order, then expands.
        
        # Let's stick to the "pick uncovered t-subset, expand block" strategy but faster.
        
        uncovered_indices = set(range(n_tsubs))
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            # To be deterministic-ish but randomized, we can pick from the set
            # But picking from a set is O(1).
            # However, the "random order" in the original code was pre-shuffled.
            # Let's just pick a random one from the remaining uncovered.
            # This is equivalent to random order if we always pick uniformly at random from remaining.
            
            first_uncovered_idx = rng.choice(list(uncovered_indices))
            # Wait, list(uncovered_indices) is O(N).
            # Better: maintain a list and shuffle it?
            # Or just use the pre-shuffled order and scan?
            
            # Let's revert to pre-shuffled order scan, it's safer and O(N) worst case per step, but N is small (3876).
            # Actually, 3876 * 150 blocks * 500 restarts = 280M operations. Might be tight in Python.
            
            # Alternative:
            # Maintain a numpy array of uncovered indices?
            
            # Let's try a different approach for speed:
            # Instead of "pick t-subset, expand", which is hard to vectorize,
            # use "Standard Greedy Set Cover":
            # 1. Calculate gain of every possible block (k-subset)? No, C(19,7) is 50388.
            # 2. Calculate gain of every block?
            #    Gain = number of uncovered t-subsets contained in the block.
            # 3. Pick block with max gain.
            # 4. Mark covered.
            
            # This is O(C(v,k) * C(k,t)) per step.
            # C(19,7) = 50388. C(7,4) = 35.
            # 50388 * 35 = 1.7M operations per step.
            # 150 steps = 250M operations.
            # This is doable with NumPy if vectorized over blocks?
            # But we need to update coverage.
            
            # Let's stick to the "expand around a seed" method but optimize the inner loop.
            
            # Actually, the "expand around a seed" method is:
            # 1. Pick an uncovered t-subset S.
            # 2. We must include S in our block B.
            # 3. We need to add k-t points to S to form B.
            # 4. We want to maximize the number of NEW t-subsets covered.
            
            # This is a combinatorial optimization on the remaining points.
            # Since k-t is small (7-4=3), we can be smarter.
            
            # Let's try the Standard Greedy Set Cover approach with NumPy.
            # Precompute all k-subsets?
            # C(19,7) = 50388 blocks.
            # For each block, precompute the list of t-subset indices it covers.
            
            # This precomputation is expensive but done once.
            # 50388 blocks * 35 t-subsets/block = 1.7M entries.
            # Memory: 1.7M * 4 bytes = 6.8 MB. Fine.
            
            # Let's do this precomputation outside the greedy loop if possible, or once.
            
            break # Exit while loop, this strategy is too complex to implement quickly in this step.
        
        # Let's restart with the original "expand around seed" but optimized.
        # The bottleneck in the original code was `itertools.combinations(block, t-1)` inside the candidate loop.
        
        # New Strategy:
        # 1. Precompute for every pair (point, t-1 subset) the t-subset index?
        #    Too many.
        
        # Let's use the "Standard Greedy Set Cover" over all C(v,k) blocks.
        # It's simpler and vectorizable.
        
        return None

    # --- Implementation of Standard Greedy Set Cover ---
    
    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # block_tsubs: list of lists of t-subset indices
    block_tsubs = []
    for block in all_ksubs:
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                covered.append(idx)
        block_tsubs.append(covered)
    
    # Convert to numpy arrays for fast access?
    # We need to calculate gain: sum of uncovered[i] for i in block_tsubs[b]
    # This is a sparse matrix multiplication or dot product.
    
    # Let's create a sparse matrix or just use lists and sum.
    # With 50k blocks and 35 entries each, summing 35 booleans is fast.
    
    def greedy_build_standard(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8) # 1 if uncovered
        blocks = []
        
        # We need to pick the block with max gain.
        # Gain[b] = sum(uncovered[i] for i in block_tsubs[b])
        
        # To speed up, we can maintain a list of active blocks?
        # Or just iterate over all blocks each time?
        # 150 steps * 50k blocks * 35 ops = 262M ops.
        # In Python, this might be too slow (1-2 seconds per step? No, 262M simple ops in NumPy is fast, in Python is slow).
        
        # Let's vectorize the gain calculation.
        # Create a matrix M of shape (n_ksubs, n_tsubs) where M[b, i] = 1 if block b covers t-subset i.
        # This is 50k * 3876 * 1 byte = 195 MB. A bit large but acceptable?
        # Or sparse?
        
        # Let's try to avoid the huge matrix.
        # We can compute gains for all blocks in parallel using NumPy if we have the matrix.
        
        # Let's build the matrix.
        # M = np.zeros((n_ksubs, n_tsubs), dtype=np.int8)
        # for b, covered in enumerate(block_tsubs):
        #     for i in covered:
        #         M[b, i] = 1
        
        # This construction takes time.
        # 1.7M assignments. Fast.
        
        M = np.zeros((n_ksubs, n_tsubs), dtype=np.int8)
        for b, covered in enumerate(block_tsubs):
            if covered:
                M[b, covered] = 1
        
        # Now, gain = M @ uncovered
        # uncovered is (n_tsubs,). M is (n_ksubs, n_tsubs).
        # gain is (n_ksubs,).
        
        # We need to pick the block with max gain.
        # If max gain is 0, we are stuck (should not happen if covering exists).
        
        # To add randomness, we can pick from the top few?
        # Or just pick max. Randomness comes from... where?
        # In standard greedy, if we always pick max, it's deterministic.
        # We can randomize by adding noise to gains?
        # Or by randomizing the order of tie-breaking?
        
        # Let's add small random noise to gains to break ties randomly.
        
        while True:
            gains = M.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Pick a block with max gain
            # To allow randomness, pick randomly from blocks with gain == max_gain
            # Or add noise.
            
            # Let's pick randomly from those with max gain.
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            b_idx = rng.choice(candidates)
            
            blocks.append(all_ksubs[b_idx])
            
            # Update uncovered
            # uncovered = uncovered - M[b_idx]
            # But we only care if it becomes 0.
            # Since M[b_idx] is 0/1 and uncovered is 0/1,
            # we can do: uncovered = np.maximum(0, uncovered - M[b_idx])
            # Or simply: uncovered[M[b_idx] == 1] = 0
            
            mask = M[b_idx] == 1
            uncovered[mask] = 0
            
            if not np.any(uncovered):
                break
        
        return blocks

    # Run multiple restarts
    # Since the standard greedy is deterministic (except for tie-breaking),
    # we need to ensure diversity.
    # Randomizing tie-breaking is good.
    # Also, we can randomize the initial order? No, standard greedy doesn't have order.
    
    # Let's run 50 restarts.
    # Each restart: 150 steps * (50k dot products of length 3876? No, M.dot(uncovered) is a matrix-vector product).
    # M is 50k x 3876.
    # Dot product is 50k * 3876 multiplies. ~195M multiplies per step.
    # 150 steps = 30B multiplies.
    # NumPy can do ~1 GFLOP/s? No, much faster with BLAS.
    # 30B ops might take 10-30 seconds.
    # So we can maybe do 3-5 restarts?
    
    # We need to be faster.
    # Optimization:
    # Instead of recalculating gains for ALL blocks,
    # we can update gains incrementally?
    # When we cover a set of t-subsets U,
    # the gain of block b decreases by |block_tsubs[b] intersect U|.
    
    # Let's maintain a `gains` array.
    # Initially, gains[b] = len(block_tsubs[b]) (since all uncovered).
    # When we pick block b*, we cover t-subsets U = block_tsubs[b*].
    # For every other block b, gains[b] -= |block_tsubs[b] intersect U|.
    
    # This is still expensive if we update all blocks.
    # But we can use the matrix M.
    # gains = M @ uncovered.
    # When uncovered changes by delta (delta is -1 for covered t-subsets),
    # gains changes by M @ delta.
    
    # delta is a sparse vector (only U is -1).
    # So gains -= M[:, U].sum(axis=1)
    
    # M[:, U] is a slice of M.
    # This is still a matrix-vector product but U is small (35 elements).
    # M[:, U] is 50k x 35.
    # Summing along axis 1 is 50k * 35 = 1.7M ops.
    # This is fast!
    
    def greedy_build_fast(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        gains = np.array([len(c) for c in block_tsubs], dtype=np.int32)
        blocks = []
        
        # M is (n_ksubs, n_tsubs)
        
        while True:
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            
            # Random tie-breaking
            b_idx = rng.choice(candidates)
            
            blocks.append(all_ksubs[b_idx])
            
            # Update uncovered
            covered_indices = block_tsubs[b_idx]
            if not covered_indices:
                break
                
            uncovered[covered_indices] = 0
            
            # Update gains
            # gains -= M[:, covered_indices].sum(axis=1)
            # M is int8, covered_indices is small.
            
            # Extract columns
            # This creates a view/copy.
            # M[:, covered_indices] is (n_ksubs, len(covered_indices))
            # Summing it is fast.
            
            delta_gains = M[:, covered_indices].sum(axis=1)
            gains -= delta_gains
            
            # Check if all covered
            if not np.any(uncovered):
                break
        
        return blocks

    # Test speed
    # 150 steps * 1.7M ops = 255M ops.
    # Should be < 1 second.
    # So we can do many restarts.
    
    n_restarts = 50
    for seed in range(n_restarts):
        if time.time() > deadline - 20:
            break
        blocks = greedy_build_fast(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
    
    # Local Search: Remove redundant blocks
    if best_blocks and time.time() < deadline - 10:
        best_blocks = local_search_reduce(best_blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline)
        
    return best_blocks if best_blocks else greedy_build_fast(0)

def local_search_reduce(blocks, v, k, t, all_tsubs, tsub_index, n_tsubs, deadline):
    """Remove redundant blocks."""
    if not blocks:
        return blocks
    
    # Count coverage
    cov_count = np.zeros(n_tsubs, dtype=np.int32)
    block_tsubs_list = []
    for block in blocks:
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                covered.append(idx)
                cov_count[idx] += 1
        block_tsubs_list.append(covered)
    
    improved = True
    blocks_list = list(blocks)
    bts_list = list(block_tsubs_list)
    
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(blocks_list)):
            if time.time() > deadline - 5:
                break
            
            # Try removing block i
            # If all cov_count[j] > 1 for j in bts_list[i], we can remove.
            # Actually, we need to check if any t-subset has count == 1.
            # If min(cov_count[bts_list[i]]) > 1, we can remove.
            
            if bts_list[i]:
                min_cov = np.min(cov_count[bts_list[i]])
                if min_cov > 1:
                    # Remove
                    cov_count[bts_list[i]] -= 1
                    blocks_list.pop(i)
                    bts_list.pop(i)
                    improved = True
                    break # Restart loop as indices changed
    
    return blocks_list
