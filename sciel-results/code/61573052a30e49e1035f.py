import itertools
import time
import random
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_tsets = []
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for i, block in enumerate(all_blocks):
        tsets_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            tsets_in_block.append(idx)
            tset_to_blocks[idx].append(i)
        block_tsets.append(tsets_in_block)
    
    # Build sparse incidence matrix (blocks x tsets) using numpy
    # Each row i has 1s at positions in block_tsets[i]
    # Use a dense matrix since num_blocks * num_tsets might be manageable
    # For v=19,k=8,t=5: C(19,8)=75582, C(19,5)=11628 -> 75582*11628 is too large for dense
    # So we'll work with the list representation and use a different spectral approach
    
    # Spectral approach: Build a block-block similarity matrix based on shared t-sets
    # But that's O(num_blocks^2) which is too large.
    # Instead: Use the tset-to-blocks structure to build a weighted graph on blocks
    # where edge weight = number of shared t-sets. Then find dense subgraphs.
    
    # Alternative spectral approach: 
    # 1. Compute for each block a "spectral score" based on the eigenvector centrality
    #    in the block-tset bipartite graph, approximated by power iteration on the 
    #    block-block transition matrix.
    # 2. Use these scores to guide a modified greedy that picks high-centrality blocks first.
    
    # For efficiency, use power iteration to approximate top eigenvector of M^T M where
    # M is the incidence matrix. (M^T M)_{ij} = number of shared t-sets between blocks i,j.
    # This is equivalent to: for block i, (M^T M x)_i = sum_j (shared_tsets(i,j)) * x_j
    # We can compute this without forming the full matrix.
    
    def compute_block_similarity_scores(num_iter=20, seed=42):
        """Power iteration to approximate top eigenvector of the block-block similarity matrix."""
        rng = random.Random(seed)
        # Initialize with random positive vector
        x = np.random.RandomState(seed).rand(num_blocks)
        x = x / np.linalg.norm(x)
        
        for _ in range(num_iter):
            # Compute M^T M x without forming M^T M
            # (M^T M x)_i = sum_j (|block_tsets[i] ∩ block_tsets[j]|) * x_j
            # = sum over tsets in block i of (sum over blocks j containing that tset of x_j)
            
            # First, compute for each tset, the sum of x over all blocks containing it
            tset_weight = np.zeros(num_tsets)
            for i in range(num_blocks):
                for ts_idx in block_tsets[i]:
                    tset_weight[ts_idx] += x[i]
            
            # Now compute new_x[i] = sum over tsets in block i of tset_weight[ts]
            new_x = np.zeros(num_blocks)
            for i in range(num_blocks):
                s = 0.0
                for ts_idx in block_tsets[i]:
                    s += tset_weight[ts_idx]
                new_x[i] = s
            
            norm = np.linalg.norm(new_x)
            if norm < 1e-15:
                break
            x = new_x / norm
        
        return x
    
    # Compute spectral scores
    spectral_scores = compute_block_similarity_scores(num_iter=30, seed=42)
    
    # Also compute raw coverage counts for fallback
    raw_counts = np.array([len(bts) for bts in block_tsets], dtype=np.float64)
    
    # 2. Modified Greedy using spectral scores
    def greedy_cover_spectral(seed):
        rng = random.Random(seed)
        uncovered_mask = [True] * num_tsets
        uncovered_count = num_tsets
        selected = []
        # Use a heap or just scan? For speed, we'll scan but use spectral scores as tiebreaker
        # Actually, we need to find the block that covers the most currently-uncovered tsets.
        # We can't easily use spectral scores directly in greedy since we need marginal gain.
        # Instead: use spectral scores to ORDER the initial consideration, and use 
        # a weighted gain: gain = uncovered_covered + alpha * spectral_score[i]
        
        alpha = 0.01  # Small weight for spectral score
        
        while uncovered_count > 0:
            if elapsed() > deadline - 10:
                break
            
            best_idx = -1
            best_score = -1.0
            
            for i in range(num_blocks):
                # Count uncovered tsets in this block
                count = 0
                for ts_idx in block_tsets[i]:
                    if uncovered_mask[ts_idx]:
                        count += 1
                if count == 0:
                    continue
                score = count + alpha * spectral_scores[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            for ts_idx in block_tsets[best_idx]:
                if uncovered_mask[ts_idx]:
                    uncovered_mask[ts_idx] = False
                    uncovered_count -= 1
        
        return selected
    
    # 3. Cluster-based initial cover
    def cluster_based_cover(seed):
        """Use spectral clustering to identify dense block clusters, then pick representatives."""
        rng = random.Random(seed)
        
        # Rank blocks by spectral score
        ranked_blocks = list(range(num_blocks))
        ranked_blocks.sort(key=lambda i: -spectral_scores[i])
        
        # Take top N blocks and try to find a small subset that covers everything
        # This is still a set cover problem, but on a smaller candidate set
        num_candidates = min(5000, num_blocks)
        candidates = ranked_blocks[:num_candidates]
        
        # Greedy on candidates
        uncovered_mask = [True] * num_tsets
        uncovered_count = num_tsets
        selected = []
        
        while uncovered_count > 0:
            if elapsed() > deadline - 15:
                break
            
            best_idx = -1
            best_count = 0
            
            for i in candidates:
                count = 0
                for ts_idx in block_tsets[i]:
                    if uncovered_mask[ts_idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                # Need to go outside candidates
                for i in range(num_blocks):
                    if i in candidates:
                        continue
                    count = 0
                    for ts_idx in block_tsets[i]:
                        if uncovered_mask[ts_idx]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            for ts_idx in block_tsets[best_idx]:
                if uncovered_mask[ts_idx]:
                    uncovered_mask[ts_idx] = False
                    uncovered_count -= 1
        
        return selected
    
    # 4. Local Search
    def get_cover_counts(blocks):
        counts = [0] * num_tsets
        for b in blocks:
            for ts_idx in block_tsets[b]:
                counts[ts_idx] += 1
        return counts
    
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = set(sol)
        
        cover_counts = get_cover_counts(sol)
        
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 30
        
        while elapsed() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                redundant = True
                for ts_idx in block_tsets[b]:
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_tsets[b]:
                        cover_counts[ts_idx] -= 1
                    in_sol.discard(b)
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: Swap
            if sol:
                found_swap = False
                attempts = 30
                
                for _ in range(attempts):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    # Find critical tsets (only covered by b_out)
                    critical_tsets = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_tsets.append(ts_idx)
                    
                    if not critical_tsets:
                        # Redundant
                        for ts_idx in block_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        in_sol.discard(b_out)
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                    
                    # Find candidate that covers all critical tsets
                    min_ts_idx = critical_tsets[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in critical_tsets[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len
                    
                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if b_idx in in_sol:
                            continue
                        # Check if b_idx covers all critical tsets
                        b_tsets_set = set(block_tsets[b_idx])
                        if all(ts in b_tsets_set for ts in critical_tsets):
                            valid_candidates.append(b_idx)
                    
                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        # Update counts
                        for ts_idx in block_tsets[b_out]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_tsets[b_in]:
                            cover_counts[ts_idx] += 1
                        in_sol.discard(b_out)
                        in_sol.add(b_in)
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol
    
    # 5. Main search loop
    best_solution = None
    best_size = float('inf')
    
    # Try multiple strategies
    strategies = [
        ("spectral_greedy", lambda s: greedy_cover_spectral(s)),
        ("cluster", lambda s: cluster_based_cover(s)),
    ]
    
    # Run each strategy with a few seeds
    for name, strategy_fn in strategies:
        for seed in range(3):
            if elapsed() > deadline - 30:
                break
            try:
                sol = strategy_fn(seed)
                if sol:
                    ls_sol = local_search(sol, seed)
                    if len(ls_sol) < best_size:
                        best_size = len(ls_sol)
                        best_solution = ls_sol
            except Exception:
                continue
    
    # ILS on best solution
    if best_solution is not None:
        current_sol = list(best_solution)
        while elapsed() < deadline - 15:
            rng = random.Random(int(elapsed() * 1000))
            perturbed = list(current_sol)
            
            # Remove 2-4 random blocks
            num_remove = min(rng.randint(2, 4), len(perturbed))
            for _ in range(num_remove):
                if not perturbed:
                    break
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
            
            # Fix coverage
            covered = set()
            for b in perturbed:
                for ts_idx in block_tsets[b]:
                    covered.add(ts_idx)
            
            uncovered = set(range(num_tsets)) - covered
            uncovered_list = list(uncovered)
            
            # Greedily add blocks to cover uncovered
            while uncovered_list:
                if elapsed() > deadline - 10:
                    break
                best_idx = -1
                best_count = 0
                for i in range(num_blocks):
                    if i in perturbed:
                        continue
                    count = 0
                    for ts_idx in block_tsets[i]:
                        if ts_idx in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_idx = i
                if best_idx == -1 or best_count == 0:
                    break
                perturbed.append(best_idx)
                for ts_idx in block_tsets[best_idx]:
                    uncovered.discard(ts_idx)
            
            # Local search
            ls_sol = local_search(perturbed, int(elapsed() * 1000))
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                current_sol = best_solution
    
    return [list(all_blocks[i]) for i in best_solution]
