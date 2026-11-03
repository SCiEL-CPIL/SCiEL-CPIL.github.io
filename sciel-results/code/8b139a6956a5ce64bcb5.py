import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # Precompute t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Invert the index: for each t-subset index, list of blocks that cover it.
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    def compute_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    def lagrangian_cover():
        """
        Lagrangian Relaxation heuristic:
        1. Initialize dual variables (penalties) for each t-subset.
        2. Iteratively:
           a. Compute block scores based on dual variables (sum of penalties for covered t-subsets).
           b. Select top blocks by score.
           c. Update dual variables: increase penalty for uncovered t-subsets, decrease for covered.
        3. Use the best solution found during iterations.
        """
        rng = random.Random(42)
        
        # Initial dual variables: uniform
        dual = np.ones(N, dtype=np.float64)
        
        best_blocks = None
        best_score = float('inf')
        
        # Parameters
        max_iter = 500
        alpha = 0.1  # Learning rate for dual updates
        top_k_ratio = 0.05  # Select top 5% of blocks by score
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute block scores: sum of dual variables for covered t-subsets
            # Score[b] = sum_{t in block_cover[b]} dual[t]
            # Efficient computation: for each t, add dual[t] to all blocks covering t
            scores = np.zeros(B, dtype=np.float64)
            for ti in range(N):
                if dual[ti] > 0:  # Only update if penalty is positive
                    for bi in t_to_blocks[ti]:
                        scores[bi] += dual[ti]
                        
            # Select top blocks by score
            num_select = max(1, int(B * top_k_ratio))
            # Use argsort to get top indices
            top_indices = np.argsort(scores)[-num_select:]
            candidate_blocks = [int(i) for i in top_indices]
            
            # Evaluate coverage
            cnt = np.zeros(N, dtype=np.int32)
            for bi in candidate_blocks:
                for ti in block_cover[bi]:
                    cnt[ti] += 1
            
            # Count uncovered
            uncovered_count = np.sum(cnt == 0)
            
            # If fully covered, update best
            if uncovered_count == 0:
                if len(candidate_blocks) < best_score:
                    best_score = len(candidate_blocks)
                    best_blocks = list(candidate_blocks)
            else:
                # If not covered, we can try to add blocks to cover uncovered
                # For now, just track the best partial solution
                pass
            
            # Update dual variables:
            # Increase penalty for uncovered t-subsets
            # Decrease penalty for covered t-subsets
            for ti in range(N):
                if cnt[ti] == 0:
                    dual[ti] += alpha
                else:
                    dual[ti] = max(0.0, dual[ti] - alpha * 0.5)
                    
            # Periodically try to repair uncovered subsets
            if it % 10 == 0 and uncovered_count > 0:
                # Add blocks to cover uncovered subsets
                repair_blocks = list(candidate_blocks)
                cnt = np.zeros(N, dtype=np.int32)
                for bi in repair_blocks:
                    for ti in block_cover[bi]:
                        cnt[ti] += 1
                uncovered = set(np.where(cnt == 0)[0])
                
                while uncovered and time.time() < deadline - 5.0:
                    # Pick a random uncovered t-subset
                    seed_ti = next(iter(uncovered))
                    # Find best block covering it (by current score)
                    best_bi = -1
                    best_score_bi = -1
                    for bi in t_to_blocks[seed_ti]:
                        if bi not in repair_blocks:
                            # Compute how many new uncovered it covers
                            new_cov = 0
                            for ti in block_cover[bi]:
                                if ti in uncovered:
                                    new_cov += 1
                            if new_cov > best_score_bi:
                                best_score_bi = new_cov
                                best_bi = bi
                    if best_bi == -1:
                        break
                    repair_blocks.append(best_bi)
                    for ti in block_cover[best_bi]:
                        cnt[ti] += 1
                        if cnt[ti] == 1:
                            uncovered.discard(ti)
                            
                if not uncovered:
                    if len(repair_blocks) < best_score:
                        best_score = len(repair_blocks)
                        best_blocks = list(repair_blocks)
                        
        return best_blocks if best_blocks is not None else []

    def prune_redundant(blocks):
        """Remove blocks that are not essential."""
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    # Run Lagrangian relaxation
    lag_blocks = lagrangian_cover()
    
    if not lag_blocks:
        # Fallback: simple greedy
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_ti = next(iter(uncovered))
            best_bi = -1
            max_cov = -1
            for bi in t_to_blocks[seed_ti]:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_bi = bi
            if best_bi == -1:
                break
            blocks.append(best_bi)
            for ti in block_cover[best_bi]:
                uncovered.discard(ti)
        lag_blocks = blocks
    
    # Prune
    final_blocks = prune_redundant(lag_blocks)
    
    # Final validation
    cnt = compute_counts(final_blocks)
    if np.min(cnt) < 1:
        # Fallback to greedy if invalid
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_ti = next(iter(uncovered))
            best_bi = -1
            max_cov = -1
            for bi in t_to_blocks[seed_ti]:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_bi = bi
            if best_bi == -1:
                break
            blocks.append(best_bi)
            for ti in block_cover[best_bi]:
                uncovered.discard(ti)
        final_blocks = blocks
    
    return [list(all_blocks[bi]) for bi in final_blocks]
