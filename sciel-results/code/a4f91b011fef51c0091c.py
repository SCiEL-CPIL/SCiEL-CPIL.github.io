import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        # Since all blocks cover exactly C(k,t) subsets, and initially all are uncovered,
        # initial gain is constant. But as we pick blocks, gains drop.
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        # Optimization: Instead of checking all blocks every time, we could use a heap,
        # but updating a heap is O(log N) and checking all is O(N).
        # N is ~75532. Iterations ~300. 300 * 75k = 22.5M ops. Acceptable in Python? 
        # NumPy argmax is fast.
        
        # We need to handle the case where gains become 0.
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            
            # Find best block with optional randomization
            if seed > 0:
                # Add small noise to break ties - use numpy random for vectorized operation
                scores = gains.astype(np.float64).copy()
                # Add uniform noise in [0, 0.01) to break ties stochastically
                scores += rng.random() * 0.01
            else:
                scores = gains.astype(np.float64)
            
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] < 0.5: # Gain is integer, so < 0.5 means 0
                break
                
            selected.append(best_idx)
            
            # Update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def simulated_annealing(initial_blocks, max_time):
        """
        Simulated Annealing on block selection with improved move diversity.
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        current_energy = len(current_blocks) + 1000.0 * np.sum(cover_count == 0)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # SA Parameters
        T = 30.0
        cooling_rate = 0.985
        min_T = 0.05
        penalty = 1000.0
        
        rng = random.Random(42)
        no_improve_count = 0
        patience = 200
        
        while time.time() < max_time:
            if T <= min_T:
                break
            if no_improve_count > patience:
                break
                
            # Choose move type with weighted probability
            move_type = rng.random()
            
            if move_type < 0.4:
                # --- Move 1: Try to Remove (40%) ---
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[i]
                    
                    valid_removal = True
                    new_uncovered = 0
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            valid_removal = False
                            new_uncovered += 1
                    
                    if valid_removal:
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy -= 1.0
                        no_improve_count = 0
                    else:
                        delta_E = 1.0 + penalty * new_uncovered
                        if T > 0 and rng.random() < np.exp(-delta_E / T):
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(i)
                            current_energy += delta_E
                            no_improve_count += 1
                        else:
                            no_improve_count += 1

            elif move_type < 0.8:
                # --- Move 2: Try to Swap (40%) ---
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    old_b = current_blocks[i]
                    
                    unique_ts = []
                    for t_idx in k_sub_t_idxs[old_b]:
                        if cover_count[t_idx] == 1:
                            unique_ts.append(t_idx)
                    
                    if not unique_ts:
                        new_b = rng.randint(0, n_k - 1)
                        if new_b not in current_blocks:
                            for t_idx in k_sub_t_idxs[old_b]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[new_b]:
                                cover_count[t_idx] += 1
                            current_blocks[i] = new_b
                            no_improve_count += 1
                    else:
                        # Find candidates that cover all unique subsets
                        candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                        for t_idx in unique_ts[1:]:
                            candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                        candidates.discard(old_b)
                        candidates -= set(current_blocks)
                        
                        if candidates:
                            new_b = rng.choice(list(candidates))
                            for t_idx in k_sub_t_idxs[old_b]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[new_b]:
                                cover_count[t_idx] += 1
                            current_blocks[i] = new_b
                            no_improve_count += 1
                        else:
                            no_improve_count += 1

            else:
                # --- Move 3: Double Swap (20%) - swap two blocks with two others ---
                if len(current_blocks) >= 2:
                    i1, i2 = rng.sample(range(len(current_blocks)), 2)
                    old_b1, old_b2 = current_blocks[i1], current_blocks[i2]
                    
                    # Find unique subsets for each
                    unique_ts1 = [t_idx for t_idx in k_sub_t_idxs[old_b1] if cover_count[t_idx] == 1]
                    unique_ts2 = [t_idx for t_idx in k_sub_t_idxs[old_b2] if cover_count[t_idx] == 1]
                    
                    if not unique_ts1 and not unique_ts2:
                        # Both have no unique subsets - can swap freely
                        new_b1 = rng.randint(0, n_k - 1)
                        new_b2 = rng.randint(0, n_k - 1)
                        if new_b1 not in current_blocks and new_b2 not in current_blocks and new_b1 != new_b2:
                            for t_idx in k_sub_t_idxs[old_b1]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[old_b2]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[new_b1]:
                                cover_count[t_idx] += 1
                            for t_idx in k_sub_t_idxs[new_b2]:
                                cover_count[t_idx] += 1
                            current_blocks[i1] = new_b1
                            current_blocks[i2] = new_b2
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

            # --- Periodic Deletion Pass ---
            if no_improve_count % 10 == 0 or len(current_blocks) > best_len:
                changed = True
                while changed:
                    changed = False
                    rng.shuffle(current_blocks)
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        is_essential = False
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1:
                                is_essential = True
                                break
                        if not is_essential:
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(j)
                            current_energy -= 1.0
                            changed = True
                            break

            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Cool down
            T *= cooling_rate

        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        for seed in seeds:
            if time.time() >= deadline - 10.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        sa_result = simulated_annealing(best_blocks, deadline - 5.0)
        if len(sa_result) < best_len:
            best_blocks = sa_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)
    
    # Exhaustive deletion polish: try removing each block one at a time
    if best_blocks and time.time() < deadline - 1.0:
        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        # Try removing each block
        i = 0
        while i < len(best_blocks):
            b_idx = best_blocks[i]
            is_essential = False
            for t_idx in k_sub_t_idxs[b_idx]:
                if cover_count[t_idx] == 1:
                    is_essential = True
                    break
            if not is_essential:
                for t_idx in k_sub_t_idxs[b_idx]:
                    cover_count[t_idx] -= 1
                best_blocks.pop(i)
                # Don't increment i, check same position again
            else:
                i += 1
        
        # One more pass
        changed = True
        while changed:
            changed = False
            for i in range(len(best_blocks)):
                b_idx = best_blocks[i]
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    best_blocks.pop(i)
                    changed = True
                    break

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
