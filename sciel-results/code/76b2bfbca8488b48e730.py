import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Step 1: Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Map each block to the set of t-subset indices it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    # Map each t-subset to the list of block indices that cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    if elapsed() > deadline - 60:
        return []

    # Step 2: Greedy Initialization
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)
    
    # Initial gain: number of t-subsets covered
    gain = np.array([len(s) for s in block_t_subsets], dtype=np.int32)

    while uncovered_count > 0 and elapsed() < deadline - 40:
        # Find best block
        # Mask out selected blocks
        gain_masked = gain.copy()
        gain_masked[selected_mask] = -1
        
        best_idx = int(np.argmax(gain_masked))
        if gain_masked[best_idx] <= 0:
            break
            
        selected.append(best_idx)
        selected_mask[best_idx] = True
        
        # Update uncovered and gains
        for idx in block_t_subsets[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                # Update gains for all blocks covering this t-subset
                # Vectorized subtraction
                candidates = t_to_blocks[idx]
                if candidates:
                    gain[candidates] -= 1

    # Step 3: Local Search (Prune and Swap)
    cover_count = np.zeros(num_t, dtype=np.int32)
    for i in selected:
        for idx in block_t_subsets[i]:
            cover_count[idx] += 1
            
    in_solution = set(selected)
    
    def is_redundant(b_idx):
        for idx in block_t_subsets[b_idx]:
            if cover_count[idx] == 1:
                return False
        return True

    def remove_block(b_idx):
        nonlocal selected, in_solution, cover_count
        in_solution.discard(b_idx)
        try:
            selected.remove(b_idx)
        except ValueError:
            pass
        for idx in block_t_subsets[b_idx]:
            cover_count[idx] -= 1

    def add_block(b_idx):
        nonlocal selected, in_solution, cover_count
        in_solution.add(b_idx)
        selected.append(b_idx)
        for idx in block_t_subsets[b_idx]:
            cover_count[idx] += 1

    best_solution = selected[:]
    best_size = len(selected)
    
    # Local Search Loop
    while elapsed() < deadline - 10:
        # Phase 1: Prune
        pruned = True
        while pruned:
            pruned = False
            for b_idx in selected[:]:
                if is_redundant(b_idx):
                    remove_block(b_idx)
                    pruned = True
            if pruned:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected[:]

        # Phase 2: Swap
        # Adaptive number of swap attempts
        if elapsed() < deadline - 60:
            num_swap_attempts = 200
        elif elapsed() < deadline - 30:
            num_swap_attempts = 50
        else:
            num_swap_attempts = 5

        for _ in range(num_swap_attempts):
            if elapsed() > deadline - 10:
                break
            if not selected:
                break
                
            b_out = random.choice(selected)
            
            unique_ts = []
            for idx in block_t_subsets[b_out]:
                if cover_count[idx] == 1:
                    unique_ts.append(idx)
            
            if not unique_ts:
                remove_block(b_out)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected[:]
                break

            # Find replacement
            # Heuristic: pick t-subset with fewest covering blocks
            target_t = min(unique_ts, key=lambda x: len(t_to_blocks[x]))
            
            found = False
            for b_in in t_to_blocks[target_t]:
                if b_in == b_out or b_in in in_solution:
                    continue
                
                # Check if b_in covers all unique_ts
                valid = True
                for t_idx in unique_ts:
                    if t_idx not in block_t_subsets[b_in]:
                        valid = False
                        break
                
                if valid:
                    remove_block(b_out)
                    add_block(b_in)
                    found = True
                    break
            
            if found:
                break

    result = [list(k_blocks[i]) for i in best_solution]
    return result
