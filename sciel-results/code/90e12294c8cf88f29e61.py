import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Allow up to 110 seconds to be safe (limit is 150s, keep margin)
    timeout = 110.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their covered triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)
    
    # Precompute triple degree (number of blocks covering each triple)
    triple_degree = [len(triple_to_blocks[j]) for j in range(num_triples)]
    
    # Bitmask representation: each block's covered triples as a Python int bitmask
    # num_triples can be up to C(24,3)=2024, so we need big ints
    block_bitmasks = []
    for tr_list in block_triples:
        mask = 0
        for tr_idx in tr_list:
            mask |= (1 << tr_idx)
        block_bitmasks.append(mask)
    
    # Spectral analysis: compute normalized Laplacian of bipartite incidence graph
    # and use top eigenvectors to rank blocks by spectral centrality
    # Incidence matrix: rows=blocks, cols=triples
    # For spectral clustering, we use the normalized Laplacian of the bipartite graph
    
    # Build sparse incidence structure for efficient eigenvalue computation
    # A[i,j] = 1 if block i covers triple j
    # We'll use the singular value decomposition of the normalized incidence matrix
    
    # Compute degrees
    block_degree = [len(block_triples[i]) for i in range(num_blocks)]  # C(7,3)=35 for all
    triple_deg_arr = np.array(triple_degree, dtype=np.float64)
    block_deg_arr = np.array(block_degree, dtype=np.float64)
    
    # Normalized incidence: D_block^{-1/2} * A * D_triple^{-1/2}
    # Instead of building full matrix, use power iteration on a smaller problem
    # For efficiency, we'll compute spectral scores using a sampled approach
    
    # Compute spectral centrality score for each block
    # Using the fact that in bipartite graphs, the top singular vectors of 
    # D_B^{-1/2} A D_T^{-1/2} give us the spectral structure
    
    # For C(24,7,3): num_blocks = C(24,7) = 346104, num_triples = C(24,3) = 2024
    # The incidence matrix is 346104 x 2024, which is large but manageable with sparse methods
    
    # Use a simpler spectral score: for each block, compute how well it aligns
    # with the principal eigenvector of the triple-triple co-occurrence matrix
    
    # Triple-triple co-occurrence: M[j1, j2] = number of blocks containing both triple j1 and j2
    # This is expensive to compute fully, so we use a random projection approach
    
    # Alternative: use the block-triple incidence to compute a spectral score
    # via power iteration on the normalized adjacency
    
    # For practical purposes with the time budget, we'll compute spectral scores
    # using a randomized SVD approach on a subsample
    
    # Compute spectral centrality using power iteration on the bipartite graph
    # Score for block i = sum over triples j in block i of (D_T^{-1/2})_j * w_j
    # where w is the principal right singular vector
    
    # Simplified spectral score: weight each triple by its "centrality" in the 
    # triple-triple co-occurrence structure, then score blocks by sum of weights
    
    # Compute triple centrality via one step of power iteration
    # Start with uniform weights on triples
    triple_weights = np.ones(num_triples, dtype=np.float64)
    triple_weights /= triple_weights.sum()
    
    # One iteration: for each triple, update weight based on connected blocks
    # new_weight[j] = sum over blocks i containing j of (block_score[i] / block_degree[i])
    # block_score[i] = sum over triples j in i of (triple_weight[j] / triple_degree[j])
    
    # Do a few iterations of the power method
    for iteration in range(3):
        # Compute block scores
        block_scores = np.zeros(num_blocks, dtype=np.float64)
        for i in range(num_blocks):
            for tr_idx in block_triples[i]:
                block_scores[i] += triple_weights[tr_idx] / triple_degree[tr_idx]
        
        # Normalize block scores
        block_scores /= block_scores.sum() if block_scores.sum() > 0 else 1
        
        # Compute new triple weights
        new_triple_weights = np.zeros(num_triples, dtype=np.float64)
        for j in range(num_triples):
            for bi in triple_to_blocks[j]:
                new_triple_weights[j] += block_scores[bi] / block_degree[bi]
        
        new_triple_weights /= new_triple_weights.sum() if new_triple_weights.sum() > 0 else 1
        triple_weights = new_triple_weights
    
    # Spectral centrality score for each block
    block_spectral_score = np.zeros(num_blocks, dtype=np.float64)
    for i in range(num_blocks):
        for tr_idx in block_triples[i]:
            block_spectral_score[i] += triple_weights[tr_idx]
    
    # Sort blocks by spectral score (descending) for use in tie-breaking
    spectral_order = np.argsort(-block_spectral_score)
    spectral_rank = np.zeros(num_blocks, dtype=np.int64)
    for rank, bi in enumerate(spectral_order):
        spectral_rank[bi] = rank
    
    best_block_indices = None
    best_count = float('inf')
    
    def greedy_cover_weighted(random_seed=None, use_spectral=True):
        """Weighted greedy: prioritize covering rare uncovered triples. Returns list of block indices."""
        if random_seed is not None:
            random.seed(random_seed)
            
        # uncovered as bitmask
        uncovered_mask = (1 << num_triples) - 1
        block_indices = []
        
        while uncovered_mask:
            # Find uncovered triples and pick one randomly
            uncovered_indices = []
            temp = uncovered_mask
            idx = 0
            while temp:
                if temp & 1:
                    uncovered_indices.append(idx)
                temp >>= 1
                idx += 1
            
            if not uncovered_indices:
                break
                
            # Weighted random selection: prefer triples with lower degree (rarer)
            weights = [1.0 / triple_degree[tr] for tr in uncovered_indices]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            target_triple_idx = uncovered_indices[0]
            for tr, w in zip(uncovered_indices, weights):
                cum += w
                if r <= cum:
                    target_triple_idx = tr
                    break
            
            # Get candidate blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]
            
            best_gain = -1
            best_block_idx = -1
            best_spectral_rank = float('inf')
            
            # Evaluate gain using bitmask AND, with spectral tie-breaking
            for bi in candidates:
                gain = (block_bitmasks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_spectral_rank = spectral_rank[bi] if use_spectral else 0
                    if best_gain == 35:  # C(7,3)
                        break
                elif gain == best_gain and use_spectral:
                    # Tie-break by spectral centrality (lower rank = better)
                    if spectral_rank[bi] < best_spectral_rank:
                        best_block_idx = bi
                        best_spectral_rank = spectral_rank[bi]
            
            if best_block_idx != -1:
                block_indices.append(best_block_idx)
                # Update uncovered
                uncovered_mask &= ~block_bitmasks[best_block_idx]
            else:
                break
                
        return block_indices

    def peel(block_indices):
        """Remove redundant blocks. Returns list of block indices."""
        if not block_indices:
            return block_indices
            
        # Count coverage of each triple
        counts = [0] * num_triples
        for bi in block_indices:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
        
        new_indices = []
        for bi in block_indices:
            removable = True
            for tr_idx in block_triples[bi]:
                if counts[tr_idx] == 1:
                    removable = False
                    break
            if not removable:
                new_indices.append(bi)
            else:
                for tr_idx in block_triples[bi]:
                    counts[tr_idx] -= 1
                    
        return new_indices

    def local_search_remove(block_indices):
        """Try to remove redundant blocks iteratively using bitmask coverage. Returns list of block indices."""
        if not block_indices:
            return block_indices
        
        # Build coverage bitmask: bit j is set if triple j is covered at least once
        coverage_mask = 0
        for bi in block_indices:
            coverage_mask |= block_bitmasks[bi]
        
        improved = True
        while improved:
            improved = False
            for i, bi in enumerate(block_indices):
                # Block bi is removable iff every triple it covers is also covered by some other block
                # i.e., coverage_mask without bi's contribution still covers all of bi's triples
                without_bi = coverage_mask & ~block_bitmasks[bi]
                if (without_bi & block_bitmasks[bi]) == block_bitmasks[bi]:
                    # Removable
                    coverage_mask = without_bi
                    block_indices.pop(i)
                    improved = True
                    break
        
        # Second pass: after removals, some blocks may have become redundant
        improved = True
        while improved:
            improved = False
            for i, bi in enumerate(block_indices):
                without_bi = coverage_mask & ~block_bitmasks[bi]
                if (without_bi & block_bitmasks[bi]) == block_bitmasks[bi]:
                    coverage_mask = without_bi
                    block_indices.pop(i)
                    improved = True
                    break
        
        return block_indices

    # Main Loop
    attempts = 0
    while time.time() - start_time < timeout:
        seed = attempts
        # Alternate between using and not using spectral tie-breaking for diversity
        use_spectral = (attempts % 3 != 2)  # 2 out of 3 use spectral
        block_indices = greedy_cover_weighted(seed, use_spectral=use_spectral)
        block_indices = peel(block_indices)
        block_indices = local_search_remove(block_indices)
        
        if len(block_indices) < best_count:
            best_count = len(block_indices)
            best_block_indices = list(block_indices)
            
        attempts += 1
        
        # If we are getting close to timeout, break early to ensure return
        if time.time() - start_time > timeout - 8.0:
            break

    if best_block_indices is None:
        block_indices = greedy_cover_weighted(0, use_spectral=True)
        best_block_indices = peel(block_indices)
        
    return [list(all_blocks[bi]) for bi in best_block_indices]
