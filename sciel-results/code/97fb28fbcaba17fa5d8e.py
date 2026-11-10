import itertools
import time
import random
import numpy as np
from collections import defaultdict

DEADLINE_SECONDS = 115.0

def _generic_greedy(v, k, t, start_time):
    """Generic randomized greedy for small instances."""
    points = list(range(v))
    all_t = list(itertools.combinations(points, t))
    n_t = len(all_t)
    idx_map = {s: i for i, s in enumerate(all_t)}
    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)

    # coverage[b][s] = 1 if block b covers t-subset s
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    for b, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            cov[b, idx_map[sub]] = 1

    uncovered = np.ones(n_t, dtype=bool)
    chosen = []
    deadline = start_time + DEADLINE_SECONDS

    while uncovered.any() and time.time() < deadline:
        gains = cov[:, uncovered].sum(axis=1)
        if gains.max() <= 0:
            break
        b = int(np.argmax(gains))
        chosen.append(b)
        uncovered[cov[b] == 1] = False

    return [list(all_blocks[i]) for i in chosen]

def construct(v, k, t):
    start_time = time.time()

    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precompute for v=19, k=8, t=5
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)  # 11628
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    n_b = len(all_blocks)  # 75582

    # Build coverage matrix: cov[b, s] = 1 if block b covers t-subset s
    # Each block of size 8 covers C(8,5)=56 t-subsets
    cov = np.zeros((n_b, n_t), dtype=np.int8)
    # Store subset indices for each block for fast access
    block_subsets = []
    for b, blk in enumerate(all_blocks):
        subs = []
        for sub in itertools.combinations(blk, t):
            idx = idx_map[sub]
            cov[b, idx] = 1
            subs.append(idx)
        block_subsets.append(subs)
    
    # Convert block_subsets to numpy arrays for vectorized operations
    block_subsets_np = [np.array(s, dtype=np.int32) for s in block_subsets]

    # For each t-subset, list of blocks that cover it
    blocks_for_subset = [np.where(cov[:, s] == 1)[0] for s in range(n_t)]

    best_blocks = None
    best_count = float('inf')
    deadline = start_time + DEADLINE_SECONDS

    rng = random.Random(42)

    # Helper function to check if a set of blocks is a valid covering
    # Returns True if valid, False otherwise
    def is_valid_covering(chosen_indices):
        if not chosen_indices:
            return n_t == 0
        covered = np.zeros(n_t, dtype=np.int8)
        for b in chosen_indices:
            covered += cov[b]
        return np.all(covered > 0)

    # Helper function to prune redundant blocks
    # Returns a new list of indices with redundant blocks removed
    def prune(chosen_list, current_counts=None):
        if not chosen_list:
            return []
        
        if current_counts is None:
            counts = np.zeros(n_t, dtype=np.int16)
            for b in chosen_list:
                counts += cov[b]
        else:
            counts = current_counts.copy()
            
        new_chosen = []
        # Shuffle to avoid bias in removal order
        indices = list(range(len(chosen_list)))
        rng.shuffle(indices)
        
        for i in indices:
            b = chosen_list[i]
            subs = block_subsets_np[b]
            if len(subs) == 0:
                new_chosen.append(b)
                continue
                
            # A block is redundant if all subsets it covers are covered by at least one other block
            # i.e., counts[s] > 1 for all s in subs
            if np.all(counts[subs] > 1):
                # Remove b
                counts -= cov[b]
            else:
                # Keep b
                new_chosen.append(b)
                
        return new_chosen, counts

    # Main loop: Multiple restarts of greedy + local search
    while time.time() < deadline - 5.0:
        # 1. Randomized Greedy Construction
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []
        
        while uncovered.any() and time.time() < deadline - 10.0:
            # Pick a random uncovered t-subset to guide the choice
            # This helps diversify the search
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            s = int(rng.choice(unc_idx))
            
            # Candidate blocks covering this subset
            cands = blocks_for_subset[s]
            if len(cands) == 0:
                break

            # Compute gain for each candidate
            # gains = number of uncovered subsets covered by block
            # cov[cands] is shape (len(cands), n_t)
            # uncovered is shape (n_t,)
            gains = np.einsum('ij,j->i', cov[cands], uncovered)
            
            # Add small random perturbation to break ties
            gains_perturbed = gains + rng.random() * 0.5
            best_local = int(np.argmax(gains_perturbed))
            b = int(cands[best_local])
            
            chosen.append(b)
            uncovered[cov[b] == 1] = False
            
            # If we have exceeded the best known, we might want to stop early in this iteration
            # but we need a valid covering to prune. So we finish the greedy phase.
            
        if not uncovered.any():
            # We have a valid covering
            # 2. Pruning
            pruned, counts = prune(chosen)
            chosen = pruned
            
            # 3. Local Search (2-Exchange)
            # Try to improve by swapping one block for another
            # We iterate several times
            improved = True
            local_iterations = 0
            max_local_iterations = 200 # Limit local search depth
            
            while improved and local_iterations < max_local_iterations and time.time() < deadline - 5.0:
                improved = False
                local_iterations += 1
                
                # Re-compute counts if not passed
                if local_iterations == 1:
                    counts = np.zeros(n_t, dtype=np.int16)
                    for b in chosen:
                        counts += cov[b]
                else:
                    # counts should be up to date from previous step
                    pass
                    
                # Shuffle chosen to try different swaps
                indices = list(range(len(chosen)))
                rng.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    b_old = chosen[i]
                    subs_old = block_subsets_np[b_old]
                    
                    # Find essential subsets for b_old (covered only by b_old)
                    # essential_subs = [s for s in subs_old if counts[s] == 1]
                    essential_mask = counts[subs_old] == 1
                    if not np.any(essential_mask):
                        # b_old is redundant, remove it
                        chosen.pop(i)
                        counts -= cov[b_old]
                        improved = True
                        break # Restart search as indices changed
                    
                    essential_subs = subs_old[essential_mask]
                    
                    # We need to find a block b_new such that:
                    # 1. b_new covers all essential_subs
                    # 2. b_new != b_old
                    # 3. Ideally, b_new covers other subsets that allow further pruning
                    #    Or simply, the swap doesn't break validity (which is guaranteed if it covers essential subs)
                    #    and allows us to remove b_old.
                    
                    # To find candidates for b_new:
                    # Any block that covers all essential_subs is a candidate.
                    # Intersection of blocks_for_subset for each essential sub.
                    # Since essential_subs can be many, this intersection might be small.
                    
                    if len(essential_subs) == 0:
                        continue
                        
                    # Start with the first essential subset's candidates
                    cand_set = set(blocks_for_subset[essential_subs[0]].tolist())
                    cand_set.discard(b_old)
                    
                    # Intersect with candidates for other essential subsets
                    for s in essential_subs[1:]:
                        if not cand_set:
                            break
                        cand_set &= set(blocks_for_subset[s].tolist())
                    
                    if not cand_set:
                        continue
                        
                    # Pick a random candidate from the intersection
                    # Prefer candidates that cover more "singly covered" subsets?
                    # Or just random to explore.
                    cands_list = list(cand_set)
                    b_new = rng.choice(cands_list)
                    
                    # Perform the swap
                    # Remove b_old, add b_new
                    chosen[i] = b_new
                    counts -= cov[b_old]
                    counts += cov[b_new]
                    improved = True
                    break # Restart loop after a swap to re-evaluate pruning opportunities
            
            # Final prune after local search
            pruned, _ = prune(chosen)
            chosen = pruned
            
            if len(chosen) < best_count:
                best_count = len(chosen)
                best_blocks = chosen[:]
                
                # If we hit the theoretical lower bound or best known, we can stop or continue
                if best_count <= 335:
                    # Continue searching for 334
                    pass

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = np.ones(n_t, dtype=bool)
        chosen = []
        while uncovered.any() and time.time() < deadline:
            gains = np.einsum('ij,j->i', cov, uncovered)
            b = int(np.argmax(gains))
            if gains[b] <= 0:
                break
            chosen.append(b)
            uncovered[cov[b] == 1] = False
        best_blocks = chosen
        pruned, _ = prune(best_blocks)
        best_blocks = pruned

    return [list(all_blocks[i]) for i in best_blocks]
