import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block -> triple indices
    # n_ks = 346,104. t_count = 35.
    # Memory: 346104 * 35 * 4 bytes ~ 48 MB. Acceptable.
    block_t_indices = np.zeros((n_ks, 35), dtype=np.int32)
    # We need to know the actual count per block, which is always C(k,t) = 35 for k=7, t=3.
    # So fixed size is fine.
    
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Precompute triple -> block indices
    # This is a list of lists.
    triple_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for t_idx in block_t_indices[b_idx]:
            triple_blocks[t_idx].append(b_idx)
            
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return None
                
            # Find block with max gain
            # Gain = number of uncovered triples in block
            # We can compute this quickly.
            
            # Optimization: Instead of checking all n_ks blocks every time,
            # we can maintain a priority queue or just scan.
            # Scanning 346k blocks * 35 triples is ~12M ops.
            # If we do this 100 times, 1.2B ops. Too slow in Python.
            
            # Better: Precompute gains? No, they change.
            # Use NumPy vectorization.
            # covered[block_t_indices] is (n_ks, 35) boolean matrix.
            # This is 346k * 35 * 1 byte = 12 MB.
            # .sum(axis=1) gives gains.
            
            # This is fast in C via NumPy.
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find indices with max gain
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            # Randomly pick one to introduce diversity
            best_candidate = random.choice(max_gain_blocks.tolist())
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        return blocks_indices

    def local_search(blocks_indices):
        if not blocks_indices:
            return blocks_indices
            
        S = set(blocks_indices)
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in S:
            coverage_count[block_t_indices[b]] += 1
            
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            improved = False
            
            # Pass 1: Deletion
            new_S = set()
            for b in S:
                subs = block_t_indices[b]
                # If all subs are covered by >1 block (including b), we can remove b
                # i.e., coverage_count[sub] > 1 for all sub in subs
                if np.all(coverage_count[subs] > 1):
                    # Remove
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_S.add(b)
            
            if new_S != S:
                S = new_S
                continue # Restart loop to check for more deletions
                
            # Pass 2: Swap
            # For each block b in S, find unique triples
            # Try to replace b with b' that covers all unique triples
            swapped = False
            for b in list(S):
                subs = block_t_indices[b]
                unique_mask = (coverage_count[subs] == 1)
                unique_subs = subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # Should have been caught in deletion, but just in case
                    coverage_count[subs] -= 1
                    S.remove(b)
                    improved = True
                    swapped = True
                    break
                    
                # Find candidate b' that covers all unique_subs
                # Intersection of triple_blocks[u] for u in unique_subs
                # Since unique_subs is small, we can intersect lists
                candidates = set(triple_blocks[unique_subs[0]])
                for u in unique_subs[1:]:
                    candidates.intersection_update(triple_blocks[u])
                
                # Remove current set S from candidates
                candidates -= S
                
                if candidates:
                    # Pick a candidate. 
                    # Heuristic: Pick one that covers the most *other* uncovered triples?
                    # Or just pick randomly to escape local minima.
                    # Let's pick the one with max gain (covers most uncovered triples globally)
                    # But we only care about covering unique_subs.
                    # Actually, any valid swap maintains validity.
                    # To improve "redundancy", we might prefer b' that has few unique triples itself.
                    # For now, pick random.
                    b_new = random.choice(list(candidates))
                    
                    # Perform swap
                    S.remove(b)
                    coverage_count[block_t_indices[b]] -= 1
                    
                    S.add(b_new)
                    coverage_count[block_t_indices[b_new]] += 1
                    
                    improved = True
                    swapped = True
                    break # Restart loop after one swap to recalculate unique triples
                    
            if not swapped:
                # No deletion, no swap
                break
                
        return list(S)

    best_solution = None
    best_score = float('inf')
    
    # Run multiple restarts
    restarts = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restarts
        greedy_blocks = run_greedy(seed)
        
        if greedy_blocks is None:
            continue
            
        # Local Search
        polished_blocks = local_search(greedy_blocks)
        
        # Validate
        if polished_blocks:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b in polished_blocks:
                coverage_count[block_t_indices[b]] += 1
            if np.all(coverage_count > 0):
                score = len(polished_blocks)
                if score < best_score:
                    best_score = score
                    best_solution = polished_blocks
        else:
            # Greedy failed? Should not happen
            pass
            
        restarts += 1
        
    if best_solution is None:
        # Fallback to greedy without local search if time ran out or error
        greedy_blocks = run_greedy(0)
        if greedy_blocks:
            best_solution = greedy_blocks
        else:
            return []
            
    return [list(all_ks[i]) for i in best_solution]
