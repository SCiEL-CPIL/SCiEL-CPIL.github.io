import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Generate all t-subsets and index them
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Step 2: Generate all k-blocks and compute which t-subsets each covers
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each k-block, compute the set of t-subset indices it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    if elapsed() > deadline - 60:
        # Fallback: simple unweighted greedy
        uncovered = set(range(num_t))
        selected = []
        while uncovered and elapsed() < deadline - 10:
            best_idx = -1
            best_gain = -1
            for i in range(num_k):
                if i in selected:
                    continue
                gain = len(block_t_subsets[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered -= block_t_subsets[best_idx]
        result = [list(k_blocks[i]) for i in selected]
        # Remove redundant blocks
        if elapsed() < deadline - 5:
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
            # Redo properly
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
            selected = new_selected
            result = [list(k_blocks[i]) for i in selected]
        return result

    # Step 3: Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # Compute initial coverage count per t-subset
    t_cover_count = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_subsets[i]:
            t_cover_count[idx] += 1

    # Step 4: Lagrangian relaxation guided greedy set cover
    # We iteratively adjust penalty weights for uncovered t-subsets
    # to guide the greedy selection process toward global optimization
    
    # Initialize Lagrangian multipliers (penalties) for each t-subset
    # Start with weights based on rarity (inverse frequency)
    safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
    lagrangian_weights = 1.0 / safe_counts
    
    # We'll do multiple passes with different weight adjustments
    best_selected = None
    best_count = float('inf')
    
    # Multiple Lagrangian relaxation iterations with different weight strategies
    max_iterations = 5
    iteration_time_budget = (deadline - 40) / max_iterations
    
    for iteration in range(max_iterations):
        if elapsed() > deadline - 40:
            break
        
        # Adjust weights based on previous iteration's uncovered sets
        if iteration == 0:
            # First iteration: use rarity-based weights
            current_weights = lagrangian_weights.copy()
        else:
            # Subsequent iterations: boost weights for previously uncovered t-subsets
            # This encourages the greedy to focus on hard-to-cover t-subsets
            current_weights = lagrangian_weights.copy()
            if best_selected is not None:
                # Compute which t-subsets were under-covered in best solution
                cover_count_prev = np.zeros(num_t, dtype=np.int32)
                for i in best_selected:
                    for idx in block_t_subsets[i]:
                        cover_count_prev[idx] += 1
                # Boost weights for t-subsets with low coverage
                for idx in range(num_t):
                    if cover_count_prev[idx] == 1:
                        current_weights[idx] *= 2.0
                    elif cover_count_prev[idx] == 0:
                        current_weights[idx] *= 5.0
        
        # Initialize gain for each block: sum of weights of t-subsets it covers
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_subsets[i]:
                gain[i] = np.sum(current_weights[list(block_t_subsets[i])])

        # Track uncovered
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        # Greedy selection with current Lagrangian weights
        while uncovered_count > 0 and elapsed() < deadline - 30:
            # Find best block using numpy for speed
            # Set selected blocks' gains to -1 to exclude them
            gain_with_mask = gain.copy()
            gain_with_mask[selected_mask] = -1.0
            
            best_idx = int(np.argmax(gain_with_mask))
            best_gain = gain_with_mask[best_idx]

            if best_gain <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            # Update uncovered and gains
            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = current_weights[idx]
                    # Subtract weight from all blocks covering this t-subset
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        # Post-processing: remove redundant blocks
        if elapsed() < deadline - 15:
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1

            changed = True
            while changed and elapsed() < deadline - 10:
                changed = False
                new_selected = []
                for i in selected:
                    needed = False
                    for idx in block_t_subsets[i]:
                        if cover_count[idx] == 1:
                            needed = True
                            break
                    if needed:
                        new_selected.append(i)
                    else:
                        changed = True
                        for idx in block_t_subsets[i]:
                            cover_count[idx] -= 1
                if changed:
                    selected = new_selected

            # Verify coverage is complete
            cover_count_final = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count_final[idx] += 1
            
            if np.all(cover_count_final > 0):
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = selected[:]
        
        if elapsed() > deadline - 35:
            break

    # If we have a valid best solution, use it; otherwise fall back to first iteration
    if best_selected is not None:
        result = [list(k_blocks[i]) for i in best_selected]
    else:
        # Fallback: run simple weighted greedy with rarity weights
        safe_counts = np.maximum(t_cover_count, 1).astype(np.float64)
        t_weights = 1.0 / safe_counts
        
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_subsets[i]:
                gain[i] = np.sum(t_weights[list(block_t_subsets[i])])

        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)

        while uncovered_count > 0 and elapsed() < deadline - 5:
            gain_with_mask = gain.copy()
            gain_with_mask[selected_mask] = -1.0
            best_idx = int(np.argmax(gain_with_mask))
            best_gain = gain_with_mask[best_idx]

            if best_gain <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        # Remove redundant blocks
        if elapsed() < deadline - 2:
            cover_count = np.zeros(num_t, dtype=np.int32)
            for i in selected:
                for idx in block_t_subsets[i]:
                    cover_count[idx] += 1

            changed = True
            while changed and elapsed() < deadline - 1:
                changed = False
                new_selected = []
                for i in selected:
                    needed = False
                    for idx in block_t_subsets[i]:
                        if cover_count[idx] == 1:
                            needed = True
                            break
                    if needed:
                        new_selected.append(i)
                    else:
                        changed = True
                        for idx in block_t_subsets[i]:
                            cover_count[idx] -= 1
                if changed:
                    selected = new_selected

        result = [list(k_blocks[i]) for i in selected]

    return result
