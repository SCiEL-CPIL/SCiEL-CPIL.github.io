"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block to its t-subset indices
    block_tsubs = [None] * n_blocks
    # Map t-subset index to list of block indices covering it
    tsub_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(all_blocks):
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs[i] = idxs
        for idx in idxs:
            tsub_blocks[idx].append(i)

    best_blocks = None
    best_count = float('inf')

    # We will run the heap-based greedy multiple times with different randomizations?
    # The heap greedy is deterministic if we break ties consistently.
    # To introduce diversity, we can randomize the initial order or tie-breaking.
    # However, the "global" nature of the heap greedy often makes it robust.
    # Let's try a few restarts with slight perturbations or just one strong run.
    # Given the speed, we can afford multiple restarts.
    
    n_restarts = 10

    for restart in range(n_restarts):
        if not time_ok():
            break
            
        random.seed(12345 + restart * 7919)
        
        # Lazy Heap Greedy
        gains = [len(block_tsubs[i]) for i in range(n_blocks)]
        covered = [False] * n_t
        covered_count = 0
        selected_blocks = []
        
        # Heap stores (-gain, block_id)
        # We use a lazy deletion approach: when popping, check if gain is still valid
        heap = [(-gains[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        while covered_count < n_t and time_ok():
            # Pop the best block
            neg_gain, b_idx = heapq.heappop(heap)
            current_gain = -neg_gain
            
            # Check if this gain is still valid (lazy deletion)
            if current_gain != gains[b_idx]:
                continue
            
            if current_gain == 0:
                # No more gains to be had, but we might not be covered?
                # If greedy gets stuck, we might need to fill randomly, but 
                # with C(19,7,4) and 50k blocks, it should cover everything.
                break
                
            # Select this block
            selected_blocks.append(b_idx)
            
            # Update coverage
            for idx in block_tsubs[b_idx]:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
                    # Update gains of all blocks covering this t-subset
                    for other_b_idx in tsub_blocks[idx]:
                        if gains[other_b_idx] > 0:
                            gains[other_b_idx] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-gains[other_b_idx], other_b_idx))

        if covered_count == n_t:
            # Convert indices to blocks
            current_blocks = [all_blocks[i] for i in selected_blocks]
            
            # Deletion phase: remove redundant blocks
            if time_ok():
                tsub_block_count = [0] * n_t
                for b_idx in selected_blocks:
                    for idx in block_tsubs[b_idx]:
                        tsub_block_count[idx] += 1
                        
                changed = True
                while changed and time_ok():
                    changed = False
                    for i in range(len(selected_blocks)):
                        if not time_ok():
                            break
                        removable = True
                        for idx in block_tsubs[selected_blocks[i]]:
                            if tsub_block_count[idx] <= 1:
                                removable = False
                                break
                        if removable:
                            for idx in block_tsubs[selected_blocks[i]]:
                                tsub_block_count[idx] -= 1
                            selected_blocks.pop(i)
                            changed = True
                            break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [all_blocks[i] for i in selected_blocks]

            if best_count <= 140:
                break

    if best_blocks is None:
        # Fallback: should not happen
        return []

    # MILP polish
    if _HAS_MILP and time_ok():
        best_blocks = _milp_polish(v, k, t, best_blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks)

    return best_blocks


def _milp_polish(v, k, t, blocks, t_sub_to_idx, all_t_subsets, block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks):
    n_t = len(all_t_subsets)
    n_blocks_in = len(blocks)
    if n_blocks_in == 0:
        return blocks

    # We want to solve the Set Cover problem on the set of candidate blocks
    # to find the minimum subset that covers all t-subsets.
    
    # Build the constraint matrix A (n_t x n_blocks_in)
    # A[i, j] = 1 if block j covers t-subset i
    
    # To speed up MILP, we can try to reduce the number of candidate blocks.
    # However, the greedy output is already relatively small (~150-200 blocks).
    # So we can use all of them.
    
    from scipy.sparse import lil_matrix
    A = lil_matrix((n_t, n_blocks_in), dtype=np.int8)
    for j in range(n_blocks_in):
        b = blocks[j]
        # Find index of b in all_blocks? No, we need the t-subset indices directly.
        # We can recompute or store. Let's recompute for safety as block_tsubs is for all_blocks indices.
        # Actually, we have `blocks` as list of tuples.
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            A[idx, j] = 1
            
    A = A.tocsr()

    c = np.ones(n_blocks_in)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    try:
        res = milp(c, integrality=np.ones(n_blocks_in), constraints=constraint,
                   bounds=bounds, options={'time_limit': 40, 'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [j for j in range(n_blocks_in) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]
        
        # Verify coverage
        if len(new_blocks) < len(blocks):
            cov = [False] * n_t
            for b in new_blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    cov[idx] = True
            if all(cov):
                return new_blocks

    return blocks
# EVOLVE-BLOCK-END
