import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute all blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # c_k_t is the number of t-subsets in a k-block
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i

    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Using lists of ints for simplicity, but we will also build a "covers" map
    block_triples = [None] * n_blocks
    for i, block in enumerate(blocks):
        t_list = []
        for triple in itertools.combinations(block, t):
            t_list.append(triple_to_idx[triple])
        block_triples[i] = t_list

    # For each triple, precompute which blocks cover it
    # This is crucial for local search
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for tj in block_triples[i]:
            triple_to_blocks[tj].append(i)

    # Triple coverage frequency (for initial greedy weighting)
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for tj in block_triples[i]:
            triple_freq[tj] += 1

    # Weight for each triple: 1/freq (rarer triples get higher weight)
    # We will use this for the greedy step
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)

    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, tie_temp=0.0):
        """Run greedy set cover."""
        uncovered = uncovered_init.copy()
        selected = []
        # To speed up gain calculation, we could maintain a heap, but for v=24, 
        # scanning all blocks might be okay if we do it smartly. 
        # However, 346k blocks * ~76 steps * ~140 triples/block is ~3.6M ops per step? 
        # No, gain is sum of weights of uncovered triples in block.
        # Let's just compute gains for all blocks each step. 346k * 140 is ~50M ops. 
        # 76 steps * 50M = 3.8B ops. Too slow in Python.
        
        # Optimization: Only consider blocks that cover at least one uncovered triple.
        # But maintaining that set is hard.
        
        # Alternative: Since n_blocks is 346k, and we only pick ~76, 
        # we can use a vectorized approach with numpy if we store block_triples as a matrix?
        # block_triples is sparse. 
        # Let's stick to a simpler greedy: 
        # 1. Calculate gain for all blocks.
        # 2. Pick best.
        # To make this fast, we precompute the list of triples for each block.
        # We can use numpy for the sum if we represent coverage as a boolean matrix? 
        # 346k x 2024 matrix is 700MB. Too big.
        
        # Let's use a different greedy strategy: 
        # "Rare triple first". 
        # Find the uncovered triple with the fewest remaining candidate blocks.
        # Pick a block that covers it.
        
        # This is more complex to implement efficiently in Python without C extensions.
        
        # Let's try the standard greedy but optimized.
        # We can precompute the "potential" of each block? No, it changes.
        
        # Given the time limit, let's use a simpler, robust greedy:
        # 1. Sort triples by frequency (ascending).
        # 2. For the most constrained uncovered triple, pick the block that covers the most *other* uncovered triples.
        
        # Actually, the previous program did:
        # gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
        # This requires block_triples to be a 2D numpy array. 
        # We can create a sparse-like structure or just a list of lists and use a loop? 
        # A loop over 346k blocks in Python is slow (~0.1s per pass?). 
        # 76 passes * 0.1s = 7.6s. That's acceptable!
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break

            best_gain = -1
            best_idx = -1
            
            # Iterate through blocks
            # We can skip blocks that don't cover any uncovered triple? 
            # Checking that is O(140). 
            # So 346k * 140 = 48M checks per step. 
            # 76 steps * 48M = 3.6B checks. Too slow in Python.
            
            # We need a better way.
            # Let's maintain a list of "active" blocks?
            # Or, let's use the "rare triple" heuristic.
            
            # Find the uncovered triple with the minimum number of covering blocks that are NOT yet selected?
            # This is also hard to update.
            
            # Let's go back to the numpy approach if we can fit the matrix.
            # 346,104 blocks * 2024 triples * 1 byte = 700 MB. 
            # This might be too much memory or slow to allocate.
            # But we can use a sparse representation.
            
            # Let's try a hybrid:
            # 1. Pick a random uncovered triple (or the one with min freq).
            # 2. Among the blocks covering it, pick the one with max gain.
            # This reduces the search space from 346k to ~freq(triple) ~ 100-200 blocks.
            # 76 steps * 200 blocks * 140 triples = 2M ops. Very fast.
            
            # Which uncovered triple to pick? 
            # The one with the fewest available blocks? 
            # We can approximate this by picking the one with the lowest global frequency.
            
            # Get indices of uncovered triples
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
                
            # Sort uncovered triples by frequency (ascending)
            # We have triple_freq precomputed.
            # We want the triple with min freq.
            # To avoid sorting every time, we can just pick the one with min freq among uncovered.
            # np.argmin(triple_freq[unc_idx])
            
            min_freq_idx_local = np.argmin(triple_freq[unc_idx])
            target_triple = unc_idx[min_freq_idx_local]
            
            # Get blocks covering this triple
            candidate_blocks = triple_to_blocks[target_triple]
            
            # Among these, find the one with max gain
            max_gain = -1
            best_block = -1
            
            for b_idx in candidate_blocks:
                # Calculate gain: sum of weights of uncovered triples in this block
                # We can do this by iterating block_triples[b_idx]
                gain = 0
                for tj in block_triples[b_idx]:
                    if uncovered[tj]:
                        gain += triple_weights[tj]
                        
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                # Should not happen if target_triple is uncovered
                break
                
            selected.append(best_block)
            uncovered[block_triples[best_block]] = False

        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks."""
        if not selected_indices:
            return selected_indices

        remaining = list(selected_indices)
        selected_set = set(remaining)
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_remaining = []
            for b in remaining:
                if time.time() > deadline - 5.0:
                    break
                can_remove = True
                for tj in block_triples[b]:
                    if cover_count[tj] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for tj in block_triples[b]:
                        cover_count[tj] -= 1
                    changed = True
                else:
                    new_remaining.append(b)
            remaining = new_remaining

        return remaining

    def local_search_swap(selected_indices):
        """Local search: 1-for-0 and 2-for-1 swaps."""
        remaining = list(selected_indices)
        n = len(remaining)
        selected_set = set(remaining)

        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b in remaining:
            cover_count[block_triples[b]] += 1

        def try_remove(pos):
            nonlocal n
            b = remaining[pos]
            for tj in block_triples[b]:
                if cover_count[tj] <= 1:
                    return False
            for tj in block_triples[b]:
                cover_count[tj] -= 1
            remaining.pop(pos)
            selected_set.discard(b)
            n -= 1
            return True

        def try_2for1(pos1, pos2):
            nonlocal n
            b1 = remaining[pos1]
            b2 = remaining[pos2]

            unique_triples = []
            seen = set()
            for tj in block_triples[b1]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)
            for tj in block_triples[b2]:
                if cover_count[tj] == 1 and tj not in seen:
                    unique_triples.append(tj)
                    seen.add(tj)

            if not unique_triples:
                for tj in block_triples[b1]:
                    cover_count[tj] -= 1
                for tj in block_triples[b2]:
                    cover_count[tj] -= 1
                hi = max(pos1, pos2)
                lo = min(pos1, pos2)
                remaining.pop(hi)
                remaining.pop(lo)
                selected_set.discard(b1)
                selected_set.discard(b2)
                n -= 2
                return True

            # Find a block covering all unique_triples
            # unique_triples is small (usually 1-5)
            # Pick the rarest triple in unique_triples to limit candidates
            # Sort unique_triples by freq
            ut_sorted = sorted(unique_triples, key=lambda x: triple_freq[x])
            rarest = ut_sorted[0]
            
            candidates = triple_to_blocks[rarest]
            
            for nb in candidates:
                if nb in selected_set:
                    continue
                # Check if nb covers all unique_triples
                nb_triples_set = set(block_triples[nb])
                if not set(unique_triples).issubset(nb_triples_set):
                    continue
                
                # Do the swap
                for tj in block_triples[b1]:
                    cover_count[tj] -= 1
                for tj in block_triples[b2]:
                    cover_count[tj] -= 1
                for tj in block_triples[nb]:
                    cover_count[tj] += 1
                selected_set.discard(b1)
                selected_set.discard(b2)
                selected_set.add(nb)
                hi = max(pos1, pos2)
                lo = min(pos1, pos2)
                remaining.pop(hi)
                remaining.pop(lo)
                remaining.append(nb)
                n -= 1
                return True

            return False

        # Main loop
        while time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Single removals
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                if try_remove(i):
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: 2-for-1 swaps
            # Sample pairs
            n_pairs_to_try = min(500, n * (n - 1) // 2)
            found_swap = False
            for _ in range(n_pairs_to_try):
                if time.time() > deadline - 5.0:
                    break
                i, j = random.sample(range(n), 2)
                if try_2for1(i, j):
                    found_swap = True
                    break

            if found_swap:
                continue

            # No improvement
            break

        return remaining

    best_solution = None
    best_size = float('inf')

    def try_solution(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]

    # Run multiple greedy passes
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Vary the random seed or tie-breaking
        # Here we just run the "rare triple" greedy
        selected = greedy_cover(np.ones(n_triples, dtype=bool))
        try_solution(selected)
        
        # Apply local search
        if best_solution is not None and time.time() < deadline - 10.0:
            block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
            best_indices = [block_to_idx[tuple(b)] for b in best_solution]
            improved = local_search_swap(best_indices)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        return [list(b) for b in blocks[:100]]

    return best_solution
