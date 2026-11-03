"""Multi-start greedy with bias and randomized weights for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(v, k, t):
    # All candidate k-subsets of {0..v-1}
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Create incidence matrix M: M[b, s] = 1 if block b covers t-subset s
    M = np.zeros((n_blocks, n_tsubs), dtype=np.int8)
    
    # Create point incidence matrix P: P[b, p] = 1 if point p in block b
    P = np.zeros((n_blocks, v), dtype=np.int8)
    
    # Create T: T[s, p] = 1 if point p in t-subset s
    T = np.zeros((n_tsubs, v), dtype=np.int8)

    for b, blk in enumerate(all_blocks):
        for p in blk:
            P[b, p] = 1
        for ts in itertools.combinations(blk, t):
            s_idx = tsub_to_idx[ts]
            M[b, s_idx] = 1
            
    for s, ts in enumerate(all_tsubs):
        for p in ts:
            T[s, p] = 1

    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    n_restarts = 5
    
    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        blocks = []
        used = np.zeros(n_blocks, dtype=bool)
        
        while uncovered.any():
            # Main score: number of uncovered t-subsets covered
            scores = M.dot(uncovered).astype(np.float64)
            
            # Bias calculation
            point_tsub_count = T.T.dot(uncovered).astype(np.float64)
            max_pt = point_tsub_count.max()
            if max_pt > 0:
                under_cover = 1.0 - (point_tsub_count / max_pt)
            else:
                under_cover = np.ones(v, dtype=np.float64)
                
            bias_scores = P.dot(under_cover)
            
            bias_scale = 0.001
            total_scores = scores + bias_scale * bias_scores
            
            # Add random noise
            noise = rng.uniform(-0.5, 0.5, size=n_blocks)
            total_scores += noise
            
            # Mask used blocks
            total_scores[used] = -1.0
            
            best_idx = int(np.argmax(total_scores))
            best_score = total_scores[best_idx]
            
            if best_score < 0:
                break

            used[best_idx] = True
            blk = all_blocks[best_idx]
            blocks.append(list(blk))
            
            # Update uncovered
            newly_covered = M[best_idx, :] == 1
            uncovered[newly_covered] = 0

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
