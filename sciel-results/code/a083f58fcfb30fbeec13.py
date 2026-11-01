import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    n_k = len(k_blocks)

    # Precompute coverage
    block_cover = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_idx[sub] for sub in subs], dtype=np.int32)
        block_cover.append(indices)
        
    # Inverse mapping
    tsub_to_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_cover):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def verify_solution(selected_indices):
        if not selected_indices:
            return n_t == 0
        check_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_indices:
            check_counts[block_cover[b_idx]] += 1
        return not np.any(check_counts == 0)

    def greedy_set_cover(randomized=False, rng=None, alpha=0.5, top_k=10):
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.full(n_k, len(block_cover[0]), dtype=np.int32)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if randomized and rng is not None:
                positive_mask = gains > 0
                if not np.any(positive_mask):
                    break
                valid_blocks = np.where(positive_mask)[0]
                if len(valid_blocks) > top_k:
                    valid_gains = gains[valid_blocks]
                    weights = np.exp(valid_gains * alpha / 10.0)
                    weights /= weights.sum()
                    top_idx = np.argsort(weights)[-top_k:]
                    top_weights = weights[top_idx]
                    top_weights /= top_weights.sum()
                    best_block = valid_blocks[top_idx[rng.choice(len(top_idx), p=top_weights)]]
                else:
                    best_block = valid_blocks[rng.integers(0, len(valid_blocks))]
            else:
                best_block = np.argmax(gains)
                if gains[best_block] == 0:
                    break
            
            selected.append(best_block)
            
            for s_idx in block_cover[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        return selected

    def local_search(selected_indices, max_rounds=30):
        if not selected_indices:
            return selected_indices
            
        selected_set = set(selected_indices)
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_cover[b_idx]] += 1
            
        improved = True
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            # Redundancy Removal
            new_selected = []
            for b_idx in selected_indices:
                subs = block_cover[b_idx]
                if np.min(counts[subs]) > 1:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            selected_indices = new_selected
            selected_set = set(selected_indices)
            
            # Swap Moves
            if time.time() > deadline:
                break
            for i in range(len(selected_indices)):
                if time.time() > deadline:
                    break
                b_idx = selected_indices[i]
                subs = block_cover[b_idx]
                unique_mask = counts[subs] == 1
                unique_subs = subs[unique_mask]
                
                if len(unique_subs) == 0:
                    continue
                    
                first_s = unique_subs[0]
                candidates = tsub_to_blocks_np[first_s]
                
                best_replacement = -1
                best_score = -1
                
                # Limit candidates for speed
                for c in candidates[:30]:
                    if c in selected_set:
                        continue
                    c_subs_set = set(block_cover[c].tolist())
                    if not all(s in c_subs_set for s in unique_subs):
                        continue
                    c_subs = block_cover[c]
                    score = np.sum(counts[c_subs] > 1)
                    if score > best_score:
                        best_score = score
                        best_replacement = c
                        
                if best_replacement != -1:
                    counts[block_cover[b_idx]] -= 1
                    counts[block_cover[best_replacement]] += 1
                    selected_indices[i] = best_replacement
                    selected_set.discard(b_idx)
                    selected_set.add(best_replacement)
                    improved = True
                    
        return selected_indices

    def simulated_annealing(initial_solution, max_iters=5000):
        if not initial_solution:
            return initial_solution
            
        current_solution = list(initial_solution)
        current_set = set(current_solution)
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_solution:
            counts[block_cover[b_idx]] += 1
            
        # Ensure valid start
        if not verify_solution(current_solution):
            # Fill gaps greedily
            while True:
                missing = np.where(counts == 0)[0]
                if len(missing) == 0:
                    break
                if time.time() > deadline:
                    break
                # Find block covering most missing
                best_b = -1
                best_gain = -1
                for s_idx in missing[:10]: # Sample missing
                    for b_idx in tsub_to_blocks_np[s_idx]:
                        if b_idx in current_set:
                            continue
                        gain = np.sum(counts[block_cover[b_idx]] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b_idx
                if best_b == -1:
                    break
                current_solution.append(best_b)
                current_set.add(best_b)
                counts[block_cover[best_b]] += 1
                
        if not verify_solution(current_solution):
            return current_solution

        best_solution = list(current_solution)
        best_len = len(current_solution)
        
        T_start = 1.0
        T_end = 0.01
        alpha_sa = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Random move: Remove a random block
            if len(current_solution) > 0:
                idx = np.random.randint(0, len(current_solution))
                b_idx = current_solution[idx]
                subs = block_cover[b_idx]
                
                # Check if removal is valid
                if np.min(counts[subs]) > 1:
                    # Valid move: always accept
                    counts[subs] -= 1
                    current_solution.pop(idx)
                    current_set.discard(b_idx)
                    
                    if len(current_solution) < best_len:
                        best_solution = list(current_solution)
                        best_len = len(current_solution)
                else:
                    # Invalid move: accept with probability exp(-delta/T)
                    # Delta is number of newly uncovered subsets
                    delta = np.sum(counts[subs] == 1)
                    if delta == 0:
                        continue
                    prob = np.exp(-delta / (T * 10.0))
                    if np.random.random() < prob:
                        counts[subs] -= 1
                        current_solution.pop(idx)
                        current_set.discard(b_idx)
                        
                        # Try to repair immediately with a swap or add
                        # Find a block that covers the most of the now-uncovered sets
                        missing = np.where(counts == 0)[0]
                        if len(missing) > 0:
                            best_add = -1
                            best_gain = -1
                            for s_idx in missing[:5]:
                                for c in tsub_to_blocks_np[s_idx]:
                                    if c in current_set:
                                        continue
                                    gain = np.sum(counts[block_cover[c]] == 0)
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_add = c
                            if best_add != -1:
                                current_solution.append(best_add)
                                current_set.add(best_add)
                                counts[block_cover[best_add]] += 1
                                if len(current_solution) < best_len:
                                    best_solution = list(current_solution)
                                    best_len = len(current_solution)
                            else:
                                # Revert
                                counts[subs] += 1
                                current_solution.insert(idx, b_idx)
                                current_set.add(b_idx)
            else:
                # Add a random block
                b_idx = np.random.randint(0, n_k)
                if b_idx not in current_set:
                    current_solution.append(b_idx)
                    current_set.add(b_idx)
                    counts[block_cover[b_idx]] += 1
                    
        return best_solution

    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Greedy + Local Search
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        if i % 2 == 0:
            selected_indices = greedy_set_cover(randomized=False)
        else:
            selected_indices = greedy_set_cover(randomized=True, rng=rng, alpha=0.5, top_k=10)
            
        if not selected_indices:
            continue
            
        refined_indices = local_search(selected_indices, max_rounds=20)
        
        if not verify_solution(refined_indices):
            continue
            
        if len(refined_indices) < best_len:
            best_len = len(refined_indices)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]

    # Phase 2: Simulated Annealing from best solution
    if best_blocks is not None:
        # Reconstruct indices from best_blocks
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        initial_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        
        # Run SA for a few iterations
        if time.time() < deadline - 5:
            sa_solution = simulated_annealing(initial_indices, max_iters=2000)
            if verify_solution(sa_solution) and len(sa_solution) < best_len:
                best_len = len(sa_solution)
                best_blocks = [list(k_blocks[i]) for i in sa_solution]

    if best_blocks is None:
        selected_indices = greedy_set_cover(randomized=False)
        if selected_indices:
            refined_indices = local_search(selected_indices, max_rounds=10)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
        else:
            best_blocks = []
            
    return best_blocks
