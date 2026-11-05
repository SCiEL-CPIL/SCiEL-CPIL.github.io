import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    
    # Precompute block coverage indices for fast access
    block_indices = [M[b].indices for b in range(num_k)]
    
    # Precompute for each t-subset, which blocks cover it
    # This is useful for identifying candidates that cover specific exposed subsets
    t_subset_to_blocks = [set() for _ in range(num_t)]
    for b_idx in range(num_k):
        for s_idx in block_indices[b_idx]:
            t_subset_to_blocks[s_idx].add(b_idx)

    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_indices[b_idx]
                # Check if all subsets covered by this block are covered by >1 block
                if np.all(coverage[idxs] > 1):
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed, temp=0.1):
        """
        Greedy construction with temperature-based selection.
        """
        random.seed(seed)
        np.random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            mask = (coverage == 0)
            gains = M.dot(mask.astype(np.int8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Temperature-based selection:
            # If temp is high, we might pick a slightly suboptimal block to escape local optima
            # Simple approach: Add small noise to gains of top candidates
            if len(candidates) > 1 and temp > 0:
                # Create a small window of noise
                noise = np.random.random(len(candidates)) * temp
                # We don't want to pick a block with 0 gain if max_gain > 0
                # So we only perturb among the top gainers
                scores = max_gain + noise
                b_idx = candidates[np.argmax(scores)]
            else:
                b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            idxs = block_indices[b_idx]
            coverage[idxs] += 1
            uncovered_count = np.sum(coverage == 0)
            
        return blocks_idx, coverage, None

    def local_search(blocks_idx, coverage, time_limit):
        """
        Vectorized local search using swaps.
        """
        ls_start = time.time()
        improved = True
        blocks_set = set(blocks_idx)
        
        while improved and (time.time() - ls_start) < time_limit:
            improved = False
            
            # Iterate through blocks to try removing
            # We shuffle the order to avoid bias
            order = list(range(len(blocks_idx)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2.0:
                    return blocks_idx, coverage, False
                
                b_remove = blocks_idx[i]
                idxs_remove = block_indices[b_remove]
                
                # Find exposed subsets (covered only by b_remove)
                exposed_mask = (coverage == 1)
                exposed = idxs_remove[exposed_mask[idxs_remove]]
                
                if len(exposed) == 0:
                    # Can remove without adding
                    coverage[idxs_remove] -= 1
                    blocks_idx.pop(i)
                    blocks_set.discard(b_remove)
                    improved = True
                    # Reduce after removal
                    blocks_idx = reduce_blocks(blocks_idx, coverage)
                    blocks_set = set(blocks_idx)
                    break
                else:
                    # Need to find a block that covers all exposed subsets
                    # Identify candidate blocks that cover the first exposed subset
                    first_exposed = exposed[0]
                    candidate_blocks = t_subset_to_blocks[first_exposed]
                    
                    best_swap = -1
                    best_gain = -1
                    
                    # Vectorized scoring for candidates
                    # We need to check if candidate covers ALL exposed
                    # And calculate how many NEW subsets it covers
                    
                    # To speed this up, we can filter candidates first
                    # A candidate must cover all exposed subsets.
                    # We can use the intersection of sets of blocks covering each exposed subset
                    # But that's expensive. Instead, we iterate over candidates from the first exposed subset
                    # and check the rest.
                    
                    # Optimization: If there are many exposed, the number of candidates drops quickly
                    # We'll stick to iterating candidates of the first exposed, but check others vectorized if possible?
                    # For now, simple loop but optimized inner check.
                    
                    for b_candidate in candidate_blocks:
                        if b_candidate == b_remove or b_candidate in blocks_set:
                            continue
                        
                        # Check if b_candidate covers all exposed
                        # Using numpy: check if all exposed indices are in block_indices[b_candidate]
                        # This is a set operation.
                        # To make it faster, we can precompute a boolean matrix or use sparse structure?
                        # Let's try a quick check:
                        # If the size of the intersection of (exposed) and (coverage of b_candidate) is not len(exposed), skip.
                        
                        # Efficient check:
                        # We know b_candidate covers first_exposed.
                        # We need to check the rest of exposed.
                        
                        # Let's use a simple set check first, it's often fast enough for small exposed sets
                        # exposed is a numpy array
                        # block_indices[b_candidate] is a numpy array
                        # We can use np.in1d or set intersection
                        # Set intersection is generally faster for small sets
                        cand_set = set(block_indices[b_candidate])
                        if not all(e in cand_set for e in exposed):
                            continue
                        
                        # If it covers all exposed, calculate gain
                        # Gain = number of subsets covered by b_candidate that are currently uncovered (coverage == 0)
                        # Note: exposed subsets are currently covered by b_remove, so coverage is 1.
                        # After removing b_remove, they become 0.
                        # So we want to know how many of b_candidate's subsets are currently 0 in coverage,
                        # PLUS the exposed ones (which will become 0).
                        # Actually, the "new" coverage gain is:
                        # 1. The exposed ones (which b_remove was covering, so they become uncovered, then b_candidate covers them -> net 0 change in coverage count for them, but they are "saved")
                        # 2. Any other subsets that b_candidate covers that are currently 0.
                        
                        # Let's define the score as:
                        # Number of subsets in b_candidate that are currently coverage == 0.
                        # This represents how many "new" holes we fill.
                        # The exposed ones are "saved" from becoming holes.
                        
                        cand_idxs = block_indices[b_candidate]
                        # Count how many of cand_idxs have coverage == 0
                        new_covers = np.sum(coverage[cand_idxs] == 0)
                        
                        if new_covers > best_gain:
                            best_gain = new_covers
                            best_swap = b_candidate
                    
                    if best_swap >= 0:
                        # Perform swap
                        coverage[idxs_remove] -= 1
                        blocks_idx.pop(i)
                        blocks_set.discard(b_remove)
                        
                        idxs_add = block_indices[best_swap]
                        coverage[idxs_add] += 1
                        blocks_idx.append(best_swap)
                        blocks_set.add(best_swap)
                        
                        improved = True
                        # Reduce after swap
                        blocks_idx = reduce_blocks(blocks_idx, coverage)
                        blocks_set = set(blocks_idx)
                        break
        
        return blocks_idx, coverage, improved

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        # Vary temperature to explore different solution spaces
        temp = 0.05 + 0.1 * (num_restarts % 5)
        res = greedy_construct(num_restarts, temp)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            # Local Search
            # Allocate time based on how good this solution is
            # If it's significantly better, spend more time
            time_left = deadline - time.time()
            ls_time = min(15.0, time_left * 0.2)
            if ls_time > 1.0:
                blocks_idx, coverage, _ = local_search(blocks_idx, coverage, ls_time)
                
                # Final reduction
                blocks_idx = reduce_blocks(blocks_idx, coverage)
                current_score = len(blocks_idx)
                if current_score < best_score:
                    best_score = current_score
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
                    
    return best_blocks
