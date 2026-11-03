"""Randomized Greedy with Local Search Removal for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random

def construct(v, k, t):
    # All candidate k-subsets of {0..v-1}
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for blk in all_blocks:
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(blk, t)]
        block_tsubs.append(idxs)

    def greedy_with_randomness(seed):
        random.seed(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        used = np.zeros(n_blocks, dtype=bool)
        solution_blocks = []
        
        while uncovered.any():
            best_score = -1
            best_idx = -1
            # Collect candidates with max score to allow random tie-breaking
            max_score = -1
            candidates = []
            
            for b in range(n_blocks):
                if used[b]:
                    continue
                idxs = block_tsubs[b]
                score = np.sum(uncovered[idxs])
                if score > max_score:
                    max_score = score
                    candidates = [b]
                elif score == max_score:
                    candidates.append(b)
            
            if max_score <= 0:
                break

            best_idx = random.choice(candidates)
            used[best_idx] = True
            solution_blocks.append(best_idx)
            idxs = block_tsubs[best_idx]
            uncovered[idxs] = False

        return solution_blocks

    def local_search_removal(solution_blocks):
        # Initialize cover_count
        cover_count = np.zeros(n_tsubs, dtype=np.int64)
        in_solution = np.zeros(n_blocks, dtype=bool)
        
        for b_idx in solution_blocks:
            in_solution[b_idx] = True
            for t_idx in block_tsubs[b_idx]:
                cover_count[t_idx] += 1
        
        removed = True
        while removed:
            removed = False
            # Shuffle to find different removals
            random.shuffle(solution_blocks)
            for b_idx in solution_blocks:
                idxs = block_tsubs[b_idx]
                if np.min(cover_count[idxs]) > 1:
                    for t_idx in idxs:
                        cover_count[t_idx] -= 1
                    in_solution[b_idx] = False
                    solution_blocks.remove(b_idx)
                    removed = True
                    break
        
        return solution_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple times with different seeds
    # Time limit 120s. 
    # Greedy is ~150 steps * 50k blocks * numpy_sum.
    # numpy_sum on 162 elements is fast.
    # 150 * 50k = 7.5M iterations.
    # In Python, 7.5M loops is ~0.5-1s? Maybe 2s.
    # So 20 runs = 40s. 
    # Local search is fast.
    # Let's try 10-20 runs.
    
    for i in range(20):
        sol = greedy_with_randomness(i)
        sol = local_search_removal(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    if best_solution is None:
        # Fallback to deterministic greedy
        sol = greedy_with_randomness(0)
        sol = local_search_removal(sol)
        best_solution = sol

    final_blocks = [list(all_blocks[i]) for i in best_solution]
    return final_blocks
# EVOLVE-BLOCK-END
