"""Vectorized greedy covering with local search for C(19,8,5)."""
import numpy as np
from scipy import sparse
import time
import itertools
import random


def _precompute(v, k, t):
    """Precompute all k-subsets and their t-subset memberships."""
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, find which t-subsets it covers
    # Use sparse matrix: rows = blocks, cols = t-subsets
    rows = []
    cols = []
    for bi, block in enumerate(all_blocks):
        for tsub in itertools.combinations(block, t):
            cols.append(tsub_to_idx[tsub])
            rows.append(bi)
    
    mat = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_blocks, n_tsubs))
    return all_blocks, mat, all_tsubs


def _greedy_construct(v, k, t, deadline, rng):
    """Vectorized greedy covering algorithm."""
    all_blocks, mat, all_tsubs = _precompute(v, k, t)
    n_blocks = len(all_blocks)
    n_tsubs = len(all_tsubs)
    
    # Coverage count for each t-subset
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    selected = []
    selected_set = set()
    
    while True:
        if time.time() > deadline - 2.0:
            break
        # Find uncovered t-subsets
        uncovered_mask = (coverage == 0)
        if not uncovered_mask.any():
            break
        
        # For each candidate block, count how many uncovered t-subsets it covers
        # mat[:, uncovered_mask] gives the coverage of uncovered t-subsets by each block
        # We want to maximize this count
        candidate_scores = mat[:, uncovered_mask].sum(axis=1).A1
        
        # Only consider blocks not already selected
        for bi in selected_set:
            candidate_scores[bi] = -1
        
        # Pick the best block
        best_bi = int(np.argmax(candidate_scores))
        if candidate_scores[best_bi] <= 0:
            break
        
        selected.append(best_bi)
        selected_set.add(best_bi)
        coverage += np.asarray(mat[best_bi].tocsr().toarray().ravel(), dtype=np.int32)
    
    return [list(all_blocks[i]) for i in selected]


def _local_search(blocks, v, k, t, deadline, rng, max_rounds=50):
    """Try to remove redundant blocks and swap in better ones."""
    all_blocks, mat, all_tsubs = _precompute(v, k, t)
    n_blocks = len(all_blocks)
    n_tsubs = len(all_tsubs)
    
    # Build coverage from current blocks
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    block_indices = []
    for b in blocks:
        # Find index of this block
        b_sorted = tuple(sorted(b))
        # We need to map back to all_blocks index
        # For efficiency, precompute a dict
        block_indices.append(all_blocks.index(b_sorted))
    
    for bi in block_indices:
        coverage += np.asarray(mat[bi].tocsr().toarray().ravel(), dtype=np.int32)
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline - 5.0:
        improved = False
        rounds += 1
        
        # Try removing each block
        for i in range(len(block_indices) - 1, -1, -1):
            bi = block_indices[i]
            new_cov = coverage - np.asarray(mat[bi].tocsr().toarray().ravel(), dtype=np.int32)
            if (new_cov >= 0).all():
                # Block is redundant, remove it
                coverage = new_cov
                block_indices.pop(i)
                improved = True
                break
    
    return [list(all_blocks[i]) for i in block_indices]


def _multi_start_greedy(v, k, t, deadline, n_restarts=5):
    """Run greedy multiple times with different random orderings."""
    all_blocks, mat, all_tsubs = _precompute(v, k, t)
    n_blocks = len(all_blocks)
    n_tsubs = len(all_tsubs)
    
    best_blocks = None
    best_count = float('inf')
    
    time_per_restart = (deadline - 10.0) / n_restarts
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        start_time = time.time()
        remaining = deadline - 5.0 - start_time
        if remaining < 5.0:
            break
        
        # Randomize block order for diversity
        perm = list(range(n_blocks))
        rng.shuffle(perm)
        
        coverage = np.zeros(n_tsubs, dtype=np.int32)
        selected = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 3.0:
                break
            uncovered_mask = (coverage == 0)
            if not uncovered_mask.any():
                break
            
            # Score blocks by how many uncovered t-subsets they cover
            # Process in shuffled order for diversity, but still pick best
            candidate_scores = mat[:, uncovered_mask].sum(axis=1).A1
            for bi in selected_set:
                candidate_scores[bi] = -1
            
            # Add small random perturbation for diversity
            if restart > 0:
                noise = rng.uniform(0, 0.1, size=n_blocks)
                candidate_scores = candidate_scores + noise
            
            best_bi = int(np.argmax(candidate_scores))
            if candidate_scores[best_bi] <= 0:
                break
            
            selected.append(best_bi)
            selected_set.add(best_bi)
            coverage += np.asarray(mat[best_bi].tocsr().toarray().ravel(), dtype=np.int32)
        
        current_blocks = [list(all_blocks[i]) for i in selected]
        
        # Local search to remove redundant blocks
        current_blocks = _local_search(current_blocks, v, k, t, deadline, rng, max_rounds=10)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
    
    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 120s limit
    
    # For the specific case C(19,8,5)
    if v == 19 and k == 8 and t == 5:
        # Precompute once
        all_blocks, mat, all_tsubs = _precompute(v, k, t)
        n_blocks = len(all_blocks)
        n_tsubs = len(all_tsubs)
        
        best_blocks = None
        best_count = float('inf')
        
        # Phase 1: Multiple greedy restarts
        rng = random.Random(12345)
        n_restarts = 8
        time_budget = deadline - 20.0 - start_time
        time_per_restart = time_budget / n_restarts
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
            
            coverage = np.zeros(n_tsubs, dtype=np.int32)
            selected = []
            selected_set = set()
            
            # Random perturbation seed for diversity
            perturbation = rng.uniform(0, 0.5)
            
            while True:
                if time.time() > deadline - 12.0:
                    break
                uncovered_mask = (coverage == 0)
                if not uncovered_mask.any():
                    break
                
                candidate_scores = mat[:, uncovered_mask].sum(axis=1).A1
                for bi in selected_set:
                    candidate_scores[bi] = -1
                
                # Add diversity via random tie-breaking
                if restart > 0:
                    # Slightly favor random blocks among top candidates
                    top_score = candidate_scores.max()
                    if top_score > 0:
                        # Find all blocks with score close to max
                        threshold = top_score - 1
                        candidates = np.where(candidate_scores >= threshold)[0]
                        if len(candidates) > 1:
                            best_bi = int(rng.choice(candidates))
                        else:
                            best_bi = int(candidates[0])
                    else:
                        break
                else:
                    best_bi = int(np.argmax(candidate_scores))
                    if candidate_scores[best_bi] <= 0:
                        break
                
                selected.append(best_bi)
                selected_set.add(best_bi)
                coverage += np.asarray(mat[best_bi].tocsr().toarray().ravel(), dtype=np.int32)
            
            current_blocks = [list(all_blocks[i]) for i in selected]
            
            # Local search: remove redundant blocks
            if time.time() < deadline - 8.0:
                coverage2 = np.zeros(n_tsubs, dtype=np.int32)
                for bi in selected:
                    coverage2 += np.asarray(mat[bi].tocsr().toarray().ravel(), dtype=np.int32)
                
                # Try to remove redundant blocks
                for i in range(len(selected) - 1, -1, -1):
                    if time.time() > deadline - 7.0:
                        break
                    bi = selected[i]
                    new_cov = coverage2 - np.asarray(mat[bi].tocsr().toarray().ravel(), dtype=np.int32)
                    if (new_cov >= 0).all():
                        coverage2 = new_cov
                        selected.pop(i)
                
                current_blocks = [list(all_blocks[i]) for i in selected]
            
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks
        
        # Phase 2: Try to improve best solution with more aggressive local search
        if best_blocks is not None and time.time() < deadline - 5.0:
            # Try swapping blocks
            coverage = np.zeros(n_tsubs, dtype=np.int32)
            for b in best_blocks:
                bi = all_blocks.index(tuple(sorted(b)))
                coverage += np.asarray(mat[bi].tocsr().toarray().ravel(), dtype=np.int32)
            
            # Try replacing some blocks with better ones
            for _ in range(20):
                if time.time() > deadline - 3.0:
                    break
                if len(best_blocks) <= 1:
                    break
                # Pick a random block to try replacing
                idx = rng.randint(0, len(best_blocks) - 1)
                old_bi = all_blocks.index(tuple(sorted(best_blocks[idx])))
                cov_without = coverage - np.asarray(mat[old_bi].tocsr().toarray().ravel(), dtype=np.int32)
                
                # Find blocks that cover the most newly-uncovered t-subsets
                newly_uncovered = (cov_without == 0)
                if not newly_uncovered.any():
                    continue
                
                scores = mat[:, newly_uncovered].sum(axis=1).A1
                scores[old_bi] = -1
                best_new = int(np.argmax(scores))
                
                if scores[best_new] > 0:
                    # Check if this actually helps (reduces total or same count but better structure)
                    coverage = cov_without + np.asarray(mat[best_new].tocsr().toarray().ravel(), dtype=np.int32)
                    best_blocks[idx] = list(all_blocks[best_new])
        
        return best_blocks if best_blocks is not None else []
    
    # Fallback for other parameters
    return _multi_start_greedy(v, k, t, deadline, n_restarts=3)
