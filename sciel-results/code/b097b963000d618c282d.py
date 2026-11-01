import itertools
import time
import random
import numpy as np

def _affine_covering(v, k, t, seed=42):
    """
    Construct a covering using affine planes over GF(2) for 16 points,
    then extend to v points with heuristic blocks.
    """
    rng = random.Random(seed)
    blocks = []
    
    # Affine planes over GF(2)^4: 16 points, 35 planes of size 8
    # Each plane is a 2D affine subspace: {a, a+u, a+v, a+u+v} for basis u,v
    # But we want 7-element blocks. Let's use 7-element subsets of 8-point planes.
    # Actually, let's just generate all 7-subsets of the 35 affine planes.
    
    # Better: generate a good initial covering using structured blocks
    # For v=24, k=7, t=3:
    # Use 16 points as GF(2)^4, 8 extra points
    # Affine planes of GF(2)^4 give 35 planes of 8 points
    # From each plane, take a 7-subset (drop one point)
    
    points_16 = list(range(16))
    extra_points = list(range(16, v))
    
    # Generate all 2D affine subspaces of GF(2)^4
    # A 2D affine subspace is defined by a point a and a 2D linear subspace L
    # 2D linear subspaces of GF(2)^4: choose 2 independent vectors
    # Number of 2D subspaces: C(15,1)*C(14,1)/... = 35
    
    # Let's enumerate all 2D linear subspaces
    linear_subspaces = []
    seen = set()
    for u in range(1, 16):
        for v in range(u+1, 16):
            # Check independence: u & v != 0 means they share a bit, but in GF(2)^4
            # independence means u XOR v != 0, which is always true if u != v
            # Actually in GF(2)^4, two vectors are independent iff u XOR v != 0
            # which is always true if u != v. But we need the span to be 2D.
            # The span of {u, v} is {0, u, v, u^v}. This is 2D iff u != v and u != 0 and v != 0.
            # So any two distinct non-zero vectors give a 2D subspace? No!
            # {0, u, v, u^v} has 4 elements iff u, v are linearly independent,
            # i.e., u != v, u != 0, v != 0, and u^v != 0 (always true if u!=v).
            # But we also need u^v != u and u^v != v, which is true if v != 0 and u != 0.
            # So yes, any two distinct non-zero vectors give a 2D subspace.
            # But different pairs can give the same subspace.
            subspace = frozenset([0, u, v, u ^ v])
            if subspace not in seen:
                seen.add(subspace)
                linear_subspaces.append(list(subspace))
    
    # Generate all affine planes: translate each 2D subspace by each point
    affine_planes = []
    for L in linear_subspaces:
        for a in range(16):
            plane = frozenset([(a + x) % 16 for x in L])
            # In GF(2)^4, addition is XOR
            plane = frozenset([a ^ x for x in L])
            affine_planes.append(list(plane))
    
    # Remove duplicates
    unique_planes = list(set(tuple(sorted(p)) for p in affine_planes))
    
    # From each 8-point plane, create 7-point blocks by dropping each point
    # But we don't need all 8*35 = 280 blocks. Let's be selective.
    
    # Strategy: for each plane, pick one 7-subset that covers the most uncovered t-sets
    # Start with empty covering, greedily add blocks
    
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    
    # Precompute t-sets for each candidate block
    candidate_blocks = []
    for plane in unique_planes:
        for drop in plane:
            block = tuple(x for x in plane if x != drop)
            candidate_blocks.append(block)
    
    # Also add some random blocks involving extra points
    for _ in range(200):
        block = tuple(sorted(rng.sample(range(v), k)))
        candidate_blocks.append(block)
    
    # Greedily select blocks
    # Use vectorized approach
    tset_array = np.array(all_tsets, dtype=np.int32)
    
    # For each candidate block, compute which t-sets it covers
    # t-sets covered by block B are all 3-subsets of B
    # Map t-sets to indices
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Greedy selection
    covered_mask = np.zeros(len(all_tsets), dtype=bool)
    selected_blocks = []
    
    # For speed, precompute t-sets for each candidate
    block_tsets = []
    for block in candidate_blocks:
        tsets_in_block = [tset_to_idx[ts] for ts in itertools.combinations(block, t) if ts in tset_to_idx]
        block_tsets.append(tsets_in_block)
    
    # Greedy: pick block that covers most uncovered t-sets
    # This is O(n_blocks * n_tsets) which is too slow.
    # Instead, use a faster heuristic.
    
    # Simpler: just take a subset of candidate blocks
    # Let's take all 35*8 = 280 affine-derived blocks, then reduce
    affine_blocks = [list(b) for b in candidate_blocks[:len(unique_planes)*8]]
    
    # Check coverage
    for block in affine_blocks:
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    # If not all covered, add random blocks
    while uncovered:
        if not uncovered:
            break
        # Pick random uncovered t-set, make a block containing it
        ts = random.choice(list(uncovered))
        remaining = list(range(v))
        for x in ts:
            remaining.remove(x)
        extra = rng.sample(remaining, k - t)
        block = sorted(ts + extra)
        blocks.append(block)
        for t2 in itertools.combinations(block, t):
            uncovered.discard(t2)
    
    return blocks

def _build_coverage_matrix(blocks, v, k, t):
    """
    Build a coverage matrix: for each t-set, which blocks cover it.
    Returns: tset_to_blocks (dict), blocks list
    """
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    tset_to_blocks = {i: [] for i in range(len(all_tsets))}
    
    for bi, block in enumerate(blocks):
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            tset_to_blocks[idx].append(bi)
    
    return all_tsets, tset_to_idx, tset_to_blocks

def _local_search_vectorized(blocks, v, k, t, rng, time_limit):
    """
    Vectorized local search: try to remove or replace blocks.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    all_tsets, tset_to_idx, tset_to_blocks = _build_coverage_matrix(blocks, v, k, t)
    n_tsets = len(all_tsets)
    
    # Count coverage for each t-set
    coverage_count = np.array([len(tset_to_blocks[i]) for i in range(n_tsets)], dtype=np.int32)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove redundant blocks
        for bi in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            block = blocks[bi]
            block_set = set(block)
            
            # Check if any t-set is covered only by this block
            only_covered = False
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx[ts]
                if coverage_count[idx] == 1:
                    only_covered = True
                    break
            
            if not only_covered:
                # Remove block bi
                blocks.pop(bi)
                # Rebuild
                all_tsets, tset_to_idx, tset_to_blocks = _build_coverage_matrix(blocks, v, k, t)
                coverage_count = np.array([len(tset_to_blocks[i]) for i in range(n_tsets)], dtype=np.int32)
                improved = True
                break
        
        if improved:
            continue
        
        # Try to replace a block with a better one
        for bi in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            
            block = blocks[bi]
            block_set = set(block)
            
            # Find critical t-sets (covered only by this block)
            critical = []
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx[ts]
                if coverage_count[idx] == 1:
                    critical.append(ts)
            
            if not critical:
                # Should have been removed above, but just in case
                blocks.pop(bi)
                all_tsets, tset_to_idx, tset_to_blocks = _build_coverage_matrix(blocks, v, k, t)
                coverage_count = np.array([len(tset_to_blocks[i]) for i in range(n_tsets)], dtype=np.int32)
                improved = True
                break
            
            # Try random replacements
            for _ in range(30):
                if time.time() > deadline:
                    return blocks
                
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)
                
                # Check if cand covers all critical t-sets
                covers_all = all(set(ts).issubset(cand_set) for ts in critical)
                if not covers_all:
                    continue
                
                # Check that removing old block doesn't uncover anything
                # All t-sets in old block must be covered by cand or other blocks
                valid = True
                for ts in itertools.combinations(block, t):
                    if set(ts).issubset(cand_set):
                        continue
                    idx = tset_to_idx[ts]
                    # Check if covered by another block
                    covered = False
                    for bj in tset_to_blocks[idx]:
                        if bj != bi:
                            covered = True
                            break
                    if not covered:
                        valid = False
                        break
                
                if valid:
                    blocks[bi] = cand
                    all_tsets, tset_to_idx, tset_to_blocks = _build_coverage_matrix(blocks, v, k, t)
                    coverage_count = np.array([len(tset_to_blocks[i]) for i in range(n_tsets)], dtype=np.int32)
                    improved = True
                    break
            
            if improved:
                break
    
    return blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using structured algebraic construction + local search.
    """
    start_time = time.time()
    total_time_limit = 115  # Leave 5s margin before 120s limit
    
    best_blocks = None
    best_len = float('inf')
    
    # Step 1: Structured initial covering
    initial = _affine_covering(v, k, t, seed=42)
    best_blocks = initial
    best_len = len(initial)
    
    # Step 2: Local search reduction
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 5:
        reduced = _local_search_vectorized(best_blocks, v, k, t, random.Random(123), remaining * 0.7)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced
    
    # Step 3: Another local search pass with different seed
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 5:
        reduced2 = _local_search_vectorized(best_blocks, v, k, t, random.Random(789), remaining * 0.5)
        if len(reduced2) < best_len:
            best_len = len(reduced2)
            best_blocks = reduced2
    
    # Final safety check
    if time.time() - start_time > total_time_limit:
        return best_blocks
    
    return best_blocks
