import numpy as np
import itertools
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Map each block to its covered triple indices
    block_triples = []
    row_indices = []
    col_indices = []
    data = []
    
    for b_idx, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        triple_idxs = []
        for sub in subs:
            t_idx = idx_map[sub]
            triple_idxs.append(t_idx)
            row_indices.append(b_idx)
            col_indices.append(t_idx)
            data.append(1)
        block_triples.append(triple_idxs)
        
    # Build sparse matrix M (n_ks x n_ts)
    # M[i, j] = 1 if block i covers triple j
    M = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_ks, n_ts))
    
    best_solution = None
    best_size = float('inf')
    
    # Precompute triple rarity (number of blocks covering each triple)
    triple_counts_total = np.array(M.sum(axis=0)).flatten()
    triple_rarity = 1.0 / (triple_counts_total + 1e-9)
    
    # Precompute for each block: sum of rarity of all triples it covers
    block_rarity_score = np.zeros(n_ks)
    for b_idx, subs in enumerate(block_triples):
        block_rarity_score[b_idx] = np.sum(triple_rarity[subs])
    
    # Precompute block sizes for gain computation
    # M is sparse; we'll use it for gains
    
    def run_greedy_with_rarity(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        counts = np.zeros(n_ts, dtype=np.int32)
        solution = []
        
        # Smooth surrogate: use log-sum-exp of rarity for uncovered triples in each block
        # This creates a smooth preference for blocks covering rare triples
        alpha = 0.5  # weight for rarity smooth score
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5.0:
                return None, None
                
            gains = M.dot(uncovered)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 0: break
            
            # Compute smooth score: for each candidate, compute log-sum-exp of rarity
            # over its uncovered triples. This is a smooth surrogate that prefers
            # blocks covering multiple rare triples.
            if len(candidates) > 300:
                # Sample to keep computation bounded
                sample_idx = random.sample(range(len(candidates)), 300)
                candidates = candidates[sample_idx]
                
            # For each candidate block, compute smooth rarity score over uncovered triples
            scores = np.zeros(len(candidates))
            for ci, b_idx in enumerate(candidates):
                subs = block_triples[b_idx]
                # Only consider uncovered triples in this block
                unc_sub_mask = np.array([uncovered[s] for s in subs])
                if not np.any(unc_sub_mask):
                    scores[ci] = 0
                    continue
                rarities = triple_rarity[subs][unc_sub_mask]
                # log-sum-exp for numerical stability
                max_r = np.max(rarities)
                scores[ci] = max_r + np.log(np.sum(np.exp(rarities - max_r)))
            
            # Combine gain (already equal for candidates) with smooth rarity score
            # Add small random perturbation for diversity
            scores = scores + np.random.random(len(scores)) * 0.001
            
            best_candidate = candidates[np.argmax(scores)]
            
            solution.append(best_candidate)
            
            subs = block_triples[best_candidate]
            uncovered[subs] = False
            counts[subs] += 1
            
        return solution, len(solution)

    def local_search(solution, time_budget):
        # Returns improved solution
        improved = True
        while improved and time.time() - start_time < time_limit - 2.0:
            improved = False
            
            # Rebuild counts
            counts = np.zeros(n_ts, dtype=np.int32)
            for b in solution:
                counts[block_triples[b]] += 1
            
            # Deletion pass: remove blocks that are redundant
            new_solution = []
            for b_idx in solution:
                subs = block_triples[b_idx]
                if np.min(counts[subs]) >= 2:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_solution.append(b_idx)
            solution = new_solution
            
            if len(solution) == 0:
                break
                
            # Try to replace blocks with better alternatives
            # For each block, check if there's a block that covers all its unique triples
            # and also covers some additional uncovered triples
            for i in range(len(solution)):
                if time.time() - start_time > time_limit - 3.0:
                    break
                    
                b_idx = solution[i]
                subs = block_triples[b_idx]
                
                # Find triples that are uniquely covered by this block
                unique_subs = [s for s in subs if counts[s] == 1]
                
                if len(unique_subs) == 0:
                    # This block is redundant, remove it
                    counts[subs] -= 1
                    solution.pop(i)
                    i -= 1
                    improved = True
                    continue
                
                # Try to find a replacement block that covers all unique_subs
                # and has better rarity score
                if i % 5 == 0:  # Try every 5th block to save time
                    # Find all blocks that cover all unique_subs
                    # Use M to check: M[:, unique_subs] must be all 1s
                    if len(unique_subs) <= 5:  # Only try if few unique triples
                        candidate_blocks = []
                        for ub in all_ks:
                            if all(s in ub for s in unique_subs):
                                if ub != tuple(all_ks[b_idx]):
                                    candidate_blocks.append(ub)
                        
                        if candidate_blocks:
                            # Score candidates by how many additional triples they cover
                            best_score = -1
                            best_block = None
                            for cb in candidate_blocks[:50]:  # Limit to 50 candidates
                                cb_set = set(cb)
                                additional = 0
                                for s in cb_set:
                                    if counts[s] == 0:
                                        additional += 1
                                if additional > best_score:
                                    best_score = additional
                                    best_block = cb
                            
                            if best_block is not None and best_score > 0:
                                # Replace the block
                                old_subs = block_triples[b_idx]
                                counts[old_subs] -= 1
                                new_subs = list(itertools.combinations(best_block, t))
                                for s in new_subs:
                                    counts[idx_map[s]] += 1
                                # Update solution
                                solution[i] = all_ks.index(best_block)
                                improved = True
        
        return solution

    def polish(solution):
        # Final polish: try to remove redundant blocks and replace with better ones
        counts = np.zeros(n_ts, dtype=np.int32)
        for b in solution:
            counts[block_triples[b]] += 1
        
        # Multiple deletion passes
        for _ in range(5):
            new_solution = []
            for b_idx in solution:
                subs = block_triples[b_idx]
                if np.min(counts[subs]) >= 2:
                    counts[subs] -= 1
                else:
                    new_solution.append(b_idx)
            if len(new_solution) == len(solution):
                break
            solution = new_solution
        
        # Final replacement pass
        if time.time() - start_time < time_limit - 1.0:
            counts = np.zeros(n_ts, dtype=np.int32)
            for b in solution:
                counts[block_triples[b]] += 1
            
            for i in range(len(solution)):
                if time.time() - start_time > time_limit - 0.5:
                    break
                b_idx = solution[i]
                subs = block_triples[b_idx]
                unique_subs = [s for s in subs if counts[s] == 1]
                
                if len(unique_subs) == 0:
                    counts[subs] -= 1
                    solution.pop(i)
                    i -= 1
                    continue
                
                if len(unique_subs) <= 4:
                    # Try to find a replacement
                    best_score = -1
                    best_block_idx = None
                    for other_idx in range(n_ks):
                        if other_idx == b_idx:
                            continue
                        other_subs = block_triples[other_idx]
                        if all(counts[s] >= 1 or s in other_subs for s in unique_subs):
                            # Check if this block covers all unique triples
                            if all(s in other_subs for s in unique_subs):
                                additional = sum(1 for s in other_subs if counts[s] == 0)
                                if additional > best_score:
                                    best_score = additional
                                    best_block_idx = other_idx
                    
                    if best_block_idx is not None and best_score > 0:
                        old_subs = block_triples[b_idx]
                        counts[old_subs] -= 1
                        new_subs = block_triples[best_block_idx]
                        counts[new_subs] += 1
                        solution[i] = best_block_idx
        
        return solution

    # Main Loop
    restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restarts
        sol, size = run_greedy_with_rarity(seed)
        if sol is None:
            break
            
        if size < best_size:
            best_size = size
            best_solution = sol[:]
            
        # Local search on the current solution
        if size <= best_size + 5:
            sol_ls = local_search(sol[:], time_limit - 5.0)
            if len(sol_ls) < best_size:
                best_size = len(sol_ls)
                best_solution = sol_ls[:]
                
        restarts += 1
    
    # Final Polish on best_solution
    if best_solution is not None:
        best_solution = polish(best_solution)
        best_size = len(best_solution)

    return [list(all_ks[i]) for i in best_solution]
