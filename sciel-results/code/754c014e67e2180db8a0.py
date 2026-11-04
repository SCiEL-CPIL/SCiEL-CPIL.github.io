import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_t_indices = []
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Precompute candidate lists as numpy arrays for faster access
    t_subset_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in t_subset_to_blocks]
    
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    while True:
        if time.time() > deadline - 15.0:
            break
            
        random.seed(restart * 7919 + 13)  # Better seed mixing
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_count = num_t_subsets
        
        # Maintain a list of uncovered indices for efficient random selection
        uncovered_list = list(range(num_t_subsets))
        
        # Greedy Phase with adaptive candidate pool
        while uncovered_list:
            if time.time() > deadline - 5.0:
                # Critical: must finish this restart to have a valid covering
                # If we can't finish, we'll have to return what we have (invalid)
                # But greedy should be fast enough. Force completion.
                pass
            
            # Adaptive K: larger when many uncovered, smaller when few
            frac_uncovered = len(uncovered_list) / num_t_subsets
            if frac_uncovered > 0.5:
                K = min(30, max(10, int(30 * frac_uncovered)))
            elif frac_uncovered > 0.1:
                K = 15
            else:
                K = 5
            
            # Pick K random uncovered subsets and collect candidate blocks
            candidate_set = set()
            sampled = random.sample(uncovered_list, min(K, len(uncovered_list)))
            
            for u_idx in sampled:
                for b_idx in t_subset_to_blocks_arr[u_idx]:
                    candidate_set.add(b_idx)
            
            if not candidate_set:
                break
            
            cand_list = list(candidate_set)
            
            # Vectorized gain calculation
            # Stack all t-indices for candidates
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            
            # Gain = number of currently uncovered t-subsets covered by this block
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            # Find which of these were uncovered (coverage == 0)
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            
            # Update uncovered list: remove newly covered
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                # Filter uncovered_list to remove newly covered
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
            
            uncovered_count = len(uncovered_list)
            
            if uncovered_count == 0:
                break
        
        # Reduction Phase: remove non-essential blocks
        changed = True
        current_indices = block_indices[:]
        
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                # Essential if it covers at least one t-subset with coverage exactly 1
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    # Remove it
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
        
        # Second reduction pass (sometimes catches more)
        changed = True
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
        
        # Local search: try swapping blocks to reduce count
        # For each block, try to replace it with a different block that maintains coverage
        # and potentially allows removing another block
        if len(current_indices) < best_score:
            # Try improvement via swap
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for i in range(len(current_indices)):
                    b_idx = current_indices[i]
                    t_idxs = block_t_indices[b_idx]
                    # Find t-subsets that would become uncovered if we remove this block
                    at_risk = t_idxs[coverage[t_idxs] == 1]
                    if len(at_risk) == 0:
                        # Can remove entirely
                        coverage[t_idxs] -= 1
                        current_indices.pop(i)
                        improved = True
                        break
                    
                    # Try to find a replacement block that covers all at_risk subsets
                    # and doesn't add too many new blocks
                    # For each at_risk subset, find blocks that cover it
                    # We need a block that covers ALL at_risk subsets
                    # This is expensive, so only do it if at_risk is small
                    if len(at_risk) <= 3:
                        # Find candidate replacement blocks
                        candidate_replacements = set()
                        for risk_idx in at_risk:
                            for rep_b in t_subset_to_blocks_arr[risk_idx]:
                                candidate_replacements.add(rep_b)
                        
                        # Remove the current block from candidates
                        candidate_replacements.discard(b_idx)
                        
                        if candidate_replacements:
                            cand_repl_list = list(candidate_replacements)
                            # Evaluate each candidate: does it cover all at_risk?
                            best_repl = -1
                            best_gain = -1
                            for rep_b in cand_repl_list:
                                rep_t_idxs = block_t_indices[rep_b]
                                # Check if it covers all at_risk
                                covers_all = all(
                                    np.any(rep_t_idxs == r) for r in at_risk
                                )
                                if covers_all:
                                    # Gain: how many NEW subsets does it cover that aren't already covered?
                                    # Actually, we want to minimize the "cost"
                                    # Cost = number of subsets that go from coverage 1 to 2 (wasted)
                                    # Better: just check if replacement allows removing other blocks
                                    # For now, just take the first valid replacement
                                    best_repl = rep_b
                                    break
                            
                            if best_repl >= 0:
                                # Remove old block
                                coverage[t_idxs] -= 1
                                # Add new block
                                new_t_idxs = block_t_indices[best_repl]
                                coverage[new_t_idxs] += 1
                                current_indices[i] = best_repl
                                improved = True
                                break
        
        # Update best
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
        # If we've done several restarts and time is running out, stop
        if restart > 200:
            break
    
    if best_block_indices is None:
        # Fallback: run one more greedy to ensure we have a valid covering
        # This shouldn't happen if the first restart completes
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_list = list(range(num_t_subsets))
        
        while uncovered_list:
            u_idx = random.choice(uncovered_list)
            candidates = t_subset_to_blocks_arr[u_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            block_indices.append(best_block_idx)
            idxs = block_t_indices[best_block_idx]
            was_uncovered = coverage[idxs] == 0
            coverage[idxs] += 1
            newly_covered = idxs[was_uncovered]
            if len(newly_covered) > 0:
                newly_covered_set = set(newly_covered.tolist())
                uncovered_list = [x for x in uncovered_list if x not in newly_covered_set]
        
        best_block_indices = block_indices
    
    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_block_indices]
