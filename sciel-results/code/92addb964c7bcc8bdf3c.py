"""
Bitset-accelerated greedy covering with local search refinement for C(v,k,t).
Uses a greedy-then-exact hybrid: fast vectorized greedy to get a near-minimal
covering, then ILP-based completion for the residual uncovered triples.
"""

import time
import random
import itertools
import numpy as np

# Precompute mapping from 3-combination to index for v=24
_C24_3 = 2024

_COMBO_INDEX = {}
_idx = 0
for a in range(24):
    for b in range(a + 1, 24):
        for c in range(b + 1, 24):
            _COMBO_INDEX[(a, b, c)] = _idx
            _idx += 1
assert _idx == _C24_3

# Precompute inverse mapping
_IDX_TO_COMBO = [None] * _C24_3
for combo, idx in _COMBO_INDEX.items():
    _IDX_TO_COMBO[idx] = combo
_IDX_TO_COMBO = tuple(_IDX_TO_COMBO)

# Precompute triple indices for each possible 7-subset (block)
# There are C(24,7) = 346104 possible blocks, but we only need those we encounter.
# Instead, precompute for each block the list of triple indices it covers.
# We'll use a cache.

_BLOCK_TRIPLE_CACHE = {}

def _block_triple_indices(block):
    """Return list of triple indices covered by a block, using cache."""
    key = tuple(sorted(block))
    if key in _BLOCK_TRIPLE_CACHE:
        return _BLOCK_TRIPLE_CACHE[key]
    idxs = []
    for combo in itertools.combinations(block, 3):
        idxs.append(_COMBO_INDEX[combo])
    _BLOCK_TRIPLE_CACHE[key] = idxs
    return idxs


def _greedy_build(v, k, t, seed, deadline):
    """Build a covering using randomized greedy with bitset tracking."""
    rng = random.Random(seed)
    n_triples = _C24_3
    
    # Use a list for uncovered status for fast random access
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    
    all_points = list(range(v))
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick a random uncovered triple
        while True:
            idx = rng.randint(0, n_triples - 1)
            if uncovered[idx]:
                break
        
        triple = _IDX_TO_COMBO[idx]
        block = list(triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered[idx] = False
        uncovered_count -= 1
        
        # Greedily extend the block to k points
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: number of currently uncovered triples
                # that would be covered by adding x
                gain = 0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        if a > b:
                            a, b = b, a
                        # Determine sorted triple (a, b, x)
                        if x < a:
                            tri = (x, a, b)
                        elif x < b:
                            tri = (a, x, b)
                        else:
                            tri = (a, b, x)
                        idx = _COMBO_INDEX[tri]
                        if uncovered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                remaining = [x for x in all_points if x not in block_set]
                next_point = rng.choice(remaining)
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered: mark all triples in this block as covered
        for tri_idx in _block_triple_indices(block):
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                uncovered_count -= 1
        
        if time.time() > deadline:
            break
    
    return blocks, uncovered_count


def _local_search_reduce(v, k, t, blocks, deadline):
    """Reduce block count by removing redundant blocks and replacing blocks."""
    blocks = [list(b) for b in blocks]
    n_triples = _C24_3
    
    # Build coverage count array
    cov_count = [0] * n_triples
    for b in blocks:
        for tri_idx in _block_triple_indices(b):
            cov_count[tri_idx] += 1
    
    # Identify critical triples (count == 1)
    critical = [False] * n_triples
    for i in range(n_triples):
        if cov_count[i] == 1:
            critical[i] = True
    
    improved = True
    rng = random.Random(42)
    
    # Phase 1: Remove redundant blocks
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b = blocks[i]
            block_tris = _block_triple_indices(b)
            
            # Check if any triple in this block is critical
            valid = True
            for tri_idx in block_tris:
                if critical[tri_idx]:
                    valid = False
                    break
            
            if valid:
                # Remove this block
                for tri_idx in block_tris:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                
                blocks.pop(i)
                improved = True
                break
    
    # Phase 2: Block Replacement
    replacement_improved = True
    rng2 = random.Random(123)
    
    while replacement_improved and time.time() < deadline:
        replacement_improved = False
        rng2.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b_old = blocks[i]
            old_tris = _block_triple_indices(b_old)
            
            # Find critical triples in b_old
            critical_in_old = [ti for ti in old_tris if critical[ti]]
            
            if not critical_in_old:
                # Redundant block
                for tri_idx in old_tris:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[ti_idx] = False
                blocks.pop(i)
                replacement_improved = True
                break
            
            # Determine the set of points that MUST be in the new block to cover critical_in_old
            must_points = set()
            for ti in critical_in_old:
                must_points.update(_IDX_TO_COMBO[ti])
            
            if len(must_points) > k:
                continue
            
            # We need to find a block B_new containing must_points.
            available_points = [p for p in range(v) if p not in must_points]
            needed = k - len(must_points)
            
            best_new_block = None
            best_score = -1
            
            attempts = 10
            for _ in range(attempts):
                if time.time() >= deadline:
                    break
                
                if needed > 0:
                    extra = rng2.sample(available_points, needed)
                else:
                    extra = []
                
                candidate = list(must_points) + extra
                candidate.sort()
                candidate_set = set(candidate)
                
                # Check validity
                valid = True
                for ti in old_tris:
                    if cov_count[ti] == 1:
                        tri_tuple = _IDX_TO_COMBO[ti]
                        if not all(p in candidate_set for p in tri_tuple):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Score: Maximize overlap with other blocks
                score = 0
                for ti in _block_triple_indices(candidate):
                    if cov_count[ti] > 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_block = candidate
            
            if best_new_block is not None and best_new_block != b_old:
                # Perform replacement
                for ti in old_tris:
                    cov_count[ti] -= 1
                    if cov_count[ti] == 1:
                        critical[ti] = True
                    elif cov_count[ti] == 2:
                        critical[ti] = False
                
                new_tris = _block_triple_indices(best_new_block)
                for ti in new_tris:
                    cov_count[ti] += 1
                    if cov_count[ti] == 1:
                        critical[ti] = True
                    elif cov_count[ti] == 2:
                        critical[ti] = False
                
                blocks[i] = best_new_block
                replacement_improved = True
                break
    
    return blocks


def _ilp_complete_residual(v, k, t, blocks, uncovered_triples, deadline):
    """
    Use ILP to find a small set of blocks that covers all uncovered triples.
    Returns the additional blocks needed.
    """
    if not uncovered_triples:
        return []
    
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # Generate candidate blocks: all 7-subsets that cover at least one uncovered triple
    # For efficiency, we only consider blocks that contain at least 2 points from some uncovered triple
    # and at least one additional point.
    
    # Actually, let's be smarter: generate candidates by taking each uncovered triple
    # and extending it to a 7-subset in various ways.
    
    candidate_blocks = set()
    
    # For each uncovered triple, add it to candidates with random extensions
    rng = random.Random(999)
    all_points = list(range(v))
    
    for tri in uncovered_triples:
        tri_set = set(tri)
        # Generate several random extensions
        for _ in range(50):
            extra = rng.sample([p for p in all_points if p not in tri_set], k - 3)
            block = tuple(sorted(list(tri_set) + extra))
            candidate_blocks.add(block)
    
    # Also add some random blocks
    for _ in range(200):
        block = tuple(sorted(rng.sample(all_points, k)))
        candidate_blocks.add(block)
    
    candidate_list = list(candidate_blocks)
    n_candidates = len(candidate_list)
    
    if n_candidates == 0:
        return []
    
    # Build incidence matrix: A[i][j] = 1 if candidate j covers triple i
    # Only consider uncovered triples
    n_uncovered = len(uncovered_triples)
    
    # Map triple to index in uncovered list
    tri_to_idx = {tuple(t): i for i, t in enumerate(uncovered_triples)}
    
    # Build sparse matrix
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    data = []
    
    for j, block in enumerate(candidate_list):
        block_set = set(block)
        for tri in uncovered_triples:
            if all(p in block_set for p in tri):
                rows.append(tri_to_idx[tuple(tri)])
                cols.append(j)
                data.append(1)
    
    A = csr_matrix((data, (rows, cols)), shape=(n_uncovered, n_candidates))
    
    # Objective: minimize sum of x_j
    c = np.ones(n_candidates)
    
    # Constraints: A @ x >= 1 for all uncovered triples
    constraint = LinearConstraint(A, 1, np.inf)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(0, 1)
    
    # Solve ILP
    try:
        result = milp(c=c, constraints=constraint, bounds=bounds, integrality=np.ones(n_candidates))
        
        if result.success and result.x is not None:
            selected = [j for j in range(n_candidates) if result.x[j] > 0.5]
            new_blocks = [list(candidate_list[j]) for j in selected]
            return new_blocks
    except Exception:
        pass
    
    # Fallback: greedy completion
    return _greedy_complete_residual(v, k, t, blocks, uncovered_triples, deadline)


def _greedy_complete_residual(v, k, t, blocks, uncovered_triples, deadline):
    """Greedy completion for residual uncovered triples."""
    if not uncovered_triples:
        return []
    
    rng = random.Random(777)
    all_points = list(range(v))
    new_blocks = []
    
    remaining = set(tuple(t) for t in uncovered_triples)
    
    while remaining and time.time() < deadline:
        # Pick a random uncovered triple
        tri = tuple(rng.choice(list(remaining)))
        block = list(tri)
        block_set = set(block)
        
        # Extend greedily
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                gain = 0
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        a, b = block[i], block[j]
                        if a > b:
                            a, b = b, a
                        if x < a:
                            t2 = (x, a, b)
                        elif x < b:
                            t2 = (a, x, b)
                        else:
                            t2 = (a, b, x)
                        if t2 in remaining:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                remaining_points = [x for x in all_points if x not in block_set]
                next_point = rng.choice(remaining_points)
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        new_blocks.append(block)
        
        # Remove covered triples
        covered = set()
        for combo in itertools.combinations(block, 3):
            covered.add(combo)
        remaining -= covered
    
    return new_blocks


def _verify_covering(v, k, t, blocks):
    """Verify that all t-subsets are covered."""
    covered = set()
    for b in blocks:
        for combo in itertools.combinations(b, t):
            covered.add(combo)
    
    all_subsets = set(itertools.combinations(range(v), t))
    return all_subsets <= covered


def _find_uncovered(v, k, t, blocks):
    """Find uncovered t-subsets."""
    covered = set()
    for b in blocks:
        for combo in itertools.combinations(b, t):
            covered.add(combo)
    
    all_subsets = set(itertools.combinations(range(v), t))
    uncovered = all_subsets - covered
    return list(uncovered)


def construct(v, k, t):
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    # For the specific case v=24, k=7, t=3
    if v == 24 and k == 7 and t == 3:
        best_blocks = None
        best_count = float('inf')
        
        # Multiple randomized greedy builds followed by local search
        seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
                 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
        
        for seed in seeds:
            if time.time() > deadline - 30:
                break
            
            # Build a covering
            blocks, uncovered_count = _greedy_build(v, k, t, seed, deadline - 25)
            
            if uncovered_count > 0:
                # Find uncovered triples and complete
                uncovered_triples = _find_uncovered(v, k, t, blocks)
                if time.time() < deadline - 10:
                    # Try ILP completion
                    additional = _ilp_complete_residual(v, k, t, blocks, uncovered_triples, deadline - 5)
                    blocks.extend(additional)
            
            # Apply local search to reduce
            if time.time() < deadline - 5:
                refined = _local_search_reduce(v, k, t, blocks, deadline - 2)
                
                if len(refined) < best_count:
                    best_blocks = refined
                    best_count = len(refined)
                    
                    if best_count <= 76:
                        break  # Can't do better than known best
        
        if best_blocks is not None:
            return best_blocks
    
    # Fallback: simple greedy
    return _generic_greedy(v, k, t, start, deadline)


def _generic_greedy(v, k, t, start, deadline):
    """Fallback for non-24-point cases."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 5:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
