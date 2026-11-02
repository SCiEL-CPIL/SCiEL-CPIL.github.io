import itertools
import time
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Step 3: Build coverage structure using bitmasks
    # For each block, compute which t-subsets it covers
    # Use numpy for efficient operations
    block_t_subsets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)
    
    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)
    
    # Precompute block sizes (number of t-subsets each block covers)
    block_sizes = np.array([len(s) for s in block_t_subsets], dtype=np.int32)

    if elapsed() > deadline - 80:
        # Fallback: simple greedy without optimization
        uncovered = np.ones(num_t, dtype=np.bool_)
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        while uncovered.any() and elapsed() < deadline - 5:
            best_idx = -1
            best_gain = -1
            for i in range(num_k):
                if selected_mask[i]:
                    continue
                gain = np.sum(uncovered[list(block_t_subsets[i])])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for idx in block_t_subsets[best_idx]:
                uncovered[idx] = False
        return [list(k_blocks[i]) for i in selected]

    # Main greedy with heap and efficient updates
    def greedy_cover(seed_weights=None):
        """Greedy set cover with optional weight perturbation for diversity."""
        if seed_weights is None:
            # Standard weights: inverse of cover count
            t_cover_count = np.zeros(num_t, dtype=np.float64)
            for i in range(num_k):
                for idx in block_t_subsets[i]:
                    t_cover_count[idx] += 1
            safe_counts = np.maximum(t_cover_count, 1.0)
            t_weights = 1.0 / safe_counts
        else:
            t_weights = seed_weights
        
        # Initialize gain array
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_subsets[i]:
                gain[i] = np.sum(t_weights[list(block_t_subsets[i])])
        
        # Use a max-heap (negate for min-heap)
        heap = [(-gain[i], i) for i in range(num_k) if gain[i] > 0]
        heapq.heapify(heap)
        
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        
        # For efficient gain updates, track which t-subsets each block covers as numpy arrays
        block_cover_arrays = [np.array(list(s), dtype=np.int32) if s else np.array([], dtype=np.int32) 
                              for s in block_t_subsets]
        
        while uncovered_count > 0 and elapsed() < deadline - 35:
            # Pop from heap until we find an unselected block
            while heap:
                neg_gain, i = heapq.heappop(heap)
                if not selected_mask[i]:
                    break
            else:
                break
            
            gain_val = -neg_gain
            if gain_val <= 0:
                break
            
            # Select this block
            selected.append(i)
            selected_mask[i] = True
            
            # Update uncovered and gains
            cover_arr = block_cover_arrays[i]
            newly_covered = []
            for idx in cover_arr:
                if uncovered[idx]:
                    uncovered[idx] = False
                    newly_covered.append(idx)
            
            uncovered_count -= len(newly_covered)
            
            # Update gains for affected blocks
            for idx in newly_covered:
                w = t_weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w
                        # Push updated gain to heap
                        heapq.heappush(heap, (-gain[b], b))
        
        return selected

    # Run multiple greedy passes with different weight perturbations
    best_result = None
    best_count = float('inf')
    
    # First pass: standard greedy
    selected = greedy_cover()
    
    # Post-processing: remove redundant blocks
    def remove_redundant(selected):
        if not selected:
            return selected
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in selected:
            for idx in block_t_subsets[i]:
                cover_count[idx] += 1
        
        selected_set = set(selected)
        # Try removing each block multiple times
        for _pass in range(3):
            if elapsed() > deadline - 20:
                break
            changed = False
            for idx_remove in range(len(selected) - 1, -1, -1):
                if elapsed() > deadline - 20:
                    break
                block_idx = selected[idx_remove]
                # Check if removing this block leaves any t-subset uncovered
                can_remove = True
                for idx in block_t_subsets[block_idx]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in block_t_subsets[block_idx]:
                        cover_count[idx] -= 1
                    selected.pop(idx_remove)
                    selected_set.discard(block_idx)
                    changed = True
            if not changed:
                break
        
        return selected
    
    selected = remove_redundant(selected)
    if selected:
        best_count = len(selected)
        best_result = selected
    
    # Additional passes with perturbed weights for diversity
    num_extra_passes = 3
    for pass_num in range(num_extra_passes):
        if elapsed() > deadline - 30:
            break
        
        # Create perturbed weights
        base_weights = np.zeros(num_t, dtype=np.float64)
        t_cover_count = np.zeros(num_t, dtype=np.float64)
        for i in range(num_k):
            for idx in block_t_subsets[i]:
                t_cover_count[idx] += 1
        safe_counts = np.maximum(t_cover_count, 1.0)
        base_weights = 1.0 / safe_counts
        
        # Add random perturbation
        rng = np.random.RandomState(42 + pass_num)
        perturbation = rng.random(num_t) * 0.1
        perturbed_weights = base_weights * (1.0 + perturbation)
        
        selected = greedy_cover(seed_weights=perturbed_weights)
        selected = remove_redundant(selected)
        
        if selected and len(selected) < best_count:
            best_count = len(selected)
            best_result = selected
    
    if best_result is None:
        # Fallback
        uncovered = np.ones(num_t, dtype=np.bool_)
        best_result = []
        for i in range(num_k):
            if elapsed() > deadline - 5:
                break
            for idx in block_t_subsets[i]:
                if uncovered[idx]:
                    best_result.append(i)
                    uncovered[idx] = False
            if not uncovered.any():
                break
    
    return [list(k_blocks[i]) for i in best_result]
