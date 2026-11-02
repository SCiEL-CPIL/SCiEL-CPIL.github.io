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
    
    def get_gain(uncovered):
        # uncovered is a numpy array of bools
        # M is CSR, so M @ uncovered is efficient
        gains = M.dot(uncovered)
        return gains

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        counts = np.zeros(n_ts, dtype=np.int32)
        solution = []
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5.0:
                return None, None
                
            gains = get_gain(uncovered)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break # Should not happen if uncovered exists and M is valid
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Tie breaking:
            # 1. Random
            # 2. Prefer blocks covering triples with lower 'potential' (fewer total blocks covering them)
            # Precompute 'rarity' of each triple: len(ts_to_blocks[t])
            # We can approximate rarity by using a static weight.
            
            # For speed, just pick random from candidates if many, else check rarity
            if len(candidates) > 100:
                # Sample
                sample_idx = random.sample(range(len(candidates)), 100)
                candidates = candidates[sample_idx]
                
            # Calculate rarity score for candidates
            # Rarity of a block = sum of (1 / count_of_blocks_covering_triple[t]) for t in block
            # Precompute triple_rarity
            # triple_rarity[t] = 1.0 / len([b for b in all_ks if t in b])
            # This is expensive to compute for all t?
            # len(ts_to_blocks[t]) is the number of blocks covering t.
            # We can compute this from M: M.sum(axis=0)
            
            # Let's precompute triple_rarity once outside
            # But we are inside run_greedy.
            # We can pass it in.
            
            # For now, random pick
            best_candidate = random.choice(candidates)
            
            solution.append(best_candidate)
            
            # Update uncovered and counts
            subs = block_triples[best_candidate]
            uncovered[subs] = False
            counts[subs] += 1
            
        return solution, len(solution)

    # Precompute triple rarity (number of blocks covering each triple)
    # M.sum(axis=0) gives the count for each triple
    triple_counts_total = np.array(M.sum(axis=0)).flatten()
    triple_rarity = 1.0 / (triple_counts_total + 1e-9) # Avoid div by 0
    
    def run_greedy_with_rarity(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        counts = np.zeros(n_ts, dtype=np.int32)
        solution = []
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5.0:
                return None, None
                
            gains = get_gain(uncovered)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 0: break
            
            # Score candidates by rarity
            # Score = sum(triple_rarity[subs] for subs in block_triples[cand])
            # Vectorize this?
            # block_triples is a list of lists.
            # We can create a matrix of rarities?
            # Or just compute for top K candidates.
            
            if len(candidates) > 50:
                # Sample 50
                sample_idx = random.sample(range(len(candidates)), 50)
                candidates = candidates[sample_idx]
                
            best_candidate = candidates[0]
            best_score = -1
            for cand in candidates:
                subs = block_triples[cand]
                # Only consider the newly covered ones? Or all?
                # Gains are equal, so the number of new ones is the same.
                # But the *quality* of those new ones might differ.
                # Use rarity of the triples covered by this block.
                score = np.sum(triple_rarity[subs])
                if score > best_score:
                    best_score = score
                    best_candidate = cand
                    
            solution.append(best_candidate)
            
            subs = block_triples[best_candidate]
            uncovered[subs] = False
            counts[subs] += 1
            
        return solution, len(solution)

    def local_search(solution, counts, time_limit):
        # solution: list of block indices
        # counts: numpy array of current coverage counts
        # Returns improved solution and counts
        improved = True
        while improved and time.time() - start_time < time_limit - 2.0:
            improved = False
            
            # Deletion pass
            new_solution = []
            for b_idx in solution:
                subs = block_triples[b_idx]
                # Check if all subs are covered by others
                # counts[subs] includes b_idx
                # So we need min(counts[subs]) >= 2
                if np.min(counts[subs]) >= 2:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_solution.append(b_idx)
            solution = new_solution
            
            # Perturbation
            if time.time() - start_time > time_limit - 5.0:
                break
                
            if len(solution) == 0:
                break
                
            # Pick a random block to remove
            rem_idx = random.randint(0, len(solution) - 1)
            rem_block = solution[rem_idx]
            solution.pop(rem_idx)
            counts[block_triples[rem_block]] -= 1
            
            # Re-cover
            uncovered = counts == 0
            if np.any(uncovered):
                # Greedy add
                while np.any(uncovered):
                    if time.time() - start_time > time_limit - 3.0:
                        break
                    gains = get_gain(uncovered)
                    max_gain = int(np.max(gains))
                    if max_gain == 0: break
                    candidates = np.where(gains == max_gain)[0]
                    if len(candidates) > 50:
                        sample_idx = random.sample(range(len(candidates)), 50)
                        candidates = candidates[sample_idx]
                    
                    best_candidate = candidates[0]
                    best_score = -1
                    for cand in candidates:
                        subs = block_triples[cand]
                        score = np.sum(triple_rarity[subs])
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                            
                    solution.append(best_candidate)
                    subs = block_triples[best_candidate]
                    uncovered[subs] = False
                    counts[subs] += 1
                    
        return solution, counts

    # Main Loop
    restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restarts
        sol, size = run_greedy_with_rarity(seed)
        if sol is None:
            break
            
        if size < best_size:
            best_size = size
            # Reconstruct counts for local search
            counts = np.zeros(n_ts, dtype=np.int32)
            for b in sol:
                counts[block_triples[b]] += 1
            best_solution = sol[:]
            
        # Local search on the best solution found so far?
        # Or on this one?
        # Let's do local search on the current one if it's good
        if size <= best_size + 2:
            sol_ls, counts_ls = local_search(sol, counts, time.time() + 5.0) # 5s budget per LS?
            if len(sol_ls) < best_size:
                best_size = len(sol_ls)
                best_solution = sol_ls[:]
                
        restarts += 1
        
    # Final Polish on best_solution
    if best_solution is not None:
        counts = np.zeros(n_ts, dtype=np.int32)
        for b in best_solution:
            counts[block_triples[b]] += 1
        # Run deletion one last time
        new_sol = []
        for b in best_solution:
            subs = block_triples[b]
            if np.min(counts[subs]) >= 2:
                counts[subs] -= 1
            else:
                new_sol.append(b)
        best_solution = new_sol

    return [list(all_ks[i]) for i in best_solution]
