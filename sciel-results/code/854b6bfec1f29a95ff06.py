import itertools
import time
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Map each block to the list of t-subset indices it covers
    block_t_indices = []
    for block in k_blocks:
        covered = []
        for combo in itertools.combinations(block, t):
            covered.append(t_index[combo])
        block_t_indices.append(covered)

    # Map each t-subset to the list of block indices that cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_indices):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # 2. Heap-based Greedy
    # current_gain[i] = number of currently uncovered t-subsets covered by block i
    current_gain = np.array([len(lst) for lst in block_t_indices], dtype=np.int32)
    selected = set()
    solution = []
    
    # Max-heap using negative gains
    heap = [(-current_gain[i], i) for i in range(num_k)]
    heapq.heapify(heap)
    
    # uncovered[s] = True if t-subset s is not yet covered
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t

    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Pop from heap until we find a valid (non-stale, non-selected) entry
        while heap:
            neg_gain, i = heapq.heappop(heap)
            if i in selected:
                continue
            if -neg_gain != current_gain[i]:
                continue
            # Valid candidate
            break
        else:
            break # Heap exhausted
            
        if -neg_gain <= 0:
            break # No gain, cannot cover more
            
        # Select block i
        selected.add(i)
        solution.append(i)
        
        # Find newly covered t-subsets
        # block_t_indices[i] contains all t-subsets covered by i
        # We only care about those that are currently 'uncovered'
        newly_covered = []
        for s_idx in block_t_indices[i]:
            if uncovered[s_idx]:
                newly_covered.append(s_idx)
                
        if not newly_covered:
            continue # Should not happen if gain > 0, but safety
            
        # Mark them as covered
        for s_idx in newly_covered:
            uncovered[s_idx] = False
            uncovered_count -= 1
            
        # Update gains for all blocks that share these t-subsets
        for s_idx in newly_covered:
            for j in t_to_blocks[s_idx]:
                if j in selected:
                    continue
                current_gain[j] -= 1
                # Push updated gain to heap
                heapq.heappush(heap, (-current_gain[j], j))

    # 3. Simple Redundancy Removal (if time permits)
    if time.time() < deadline - 5:
        # Rebuild cover counts
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in solution:
            for s_idx in block_t_indices[i]:
                cover_count[s_idx] += 1
                
        new_solution = []
        for i in solution:
            # Check if block i is essential (covers any t-subset with count 1)
            essential = False
            for s_idx in block_t_indices[i]:
                if cover_count[s_idx] == 1:
                    essential = True
                    break
            if essential:
                new_solution.append(i)
            else:
                # Remove it
                for s_idx in block_t_indices[i]:
                    cover_count[s_idx] -= 1
                    
        solution = new_solution

    # Convert indices back to blocks
    result = [list(k_blocks[i]) for i in solution]
    return result
