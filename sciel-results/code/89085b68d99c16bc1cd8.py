import itertools
import time
import random
import math
from collections import Counter

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_triples = len(all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Run multiple randomized greedy trials (vectorized)
    trial = 0
    while time.time() < deadline - 40.0:  # Reserve 40s for ILP phase
        trial += 1
        blocks, valid = _greedy_cover_vectorized(v, k, t, all_tsubsets, deadline, trial)
        if valid and blocks:
            # Local improvement: try to remove redundant blocks
            improved_blocks = _remove_redundant(blocks, v, k, t, deadline)
            if len(improved_blocks) < len(blocks):
                blocks = improved_blocks
            
            # Additional local search: try swapping blocks
            if time.time() < deadline - 30.0:
                blocks = _local_search(blocks, v, k, t, all_tsubsets, deadline)
                blocks = _remove_redundant(blocks, v, k, t, deadline)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        if best_blocks and len(best_blocks) <= 74:
            break
    
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        blocks, valid = _greedy_cover_vectorized(v, k, t, all_tsubsets, deadline, -1)
        if valid and blocks:
            best_blocks = blocks
            best_blocks = _remove_redundant(best_blocks, v, k, t, deadline)
        else:
            best_blocks = []
    
    # Phase 2: Try ILP optimization if scipy is available
    if HAS_SCIPY and best_blocks is not None and time.time() < deadline - 5.0:
        ilp_result = _solve_ilp(best_blocks, v, k, t, all_tsubsets, deadline)
        if ilp_result is not None and len(ilp_result) < len(best_blocks):
            best_blocks = ilp_result
    
    return best_blocks

def _greedy_cover_vectorized(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Vectorized greedy covering with inverse frequency weighting.
    Uses numpy for faster computation of gains.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    # Create a set of all t-subsets for fast lookup
    all_tset = set(all_tsubsets)
    uncovered = set(all_tsubsets)
    blocks = []
    
    points = list(range(v))
    n_points = v
    
    # Precompute inverse frequency weights for each point
    # Points that appear in fewer uncovered t-subsets should be preferred
    point_weights = [1.0] * n_points
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed the block
        seed = rng.choice(tuple(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            # Precompute the (t-1)-subsets of the current block
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            # Compute gain for each candidate point
            best_x = -1
            best_score = -1.0
            best_tiebreakers = []
            
            for x in points:
                if x in block_set:
                    continue
                
                # Count how many new t-subsets this point would cover
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                # Weight by inverse frequency of the point in remaining triples
                weight = point_weights[x]
                score = gain * weight / (len(block) + 1)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_tiebreakers = [x]
                elif abs(score - best_score) < 1e-10 and score > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                # No point gives positive gain; pick a random one
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                # Add some randomness to tie-breaking
                if len(best_tiebreakers) > 1:
                    # Soft-max style selection with randomness
                    chosen_x = rng.choice(best_tiebreakers)
                else:
                    chosen_x = best_x
            
            block.append(chosen_x)
            block_set.add(chosen_x)
            
            # Update point weights based on coverage
            # Points that were in newly covered triples get lower weight
            for p in itertools.combinations(block, t-1):
                s = tuple(sorted(p + (chosen_x,)))
                if s in uncovered:
                    uncovered.discard(s)
                    # Decrease weight of points in this triple
                    for pt in s:
                        point_weights[pt] *= 0.98
        
        # Final cleanup of uncovered
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Reset weights periodically to avoid degeneration
        if len(blocks) % 50 == 0:
            point_weights = [1.0] * n_points
    
    return blocks, True

def _remove_redundant(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    Uses multiple passes to catch cascading redundancies.
    """
    if not blocks:
        return blocks
    
    improved = list(blocks)
    
    # Multiple passes to catch cascading redundancies
    for _pass in range(10):
        if time.time() > deadline:
            break
        
        # Build a count for each t-subset: how many blocks cover it
        cover_count = Counter()
        for block in improved:
            for s in itertools.combinations(block, t):
                cover_count[s] += 1
        
        # Try removing each block if it's redundant
        # We iterate backwards so that removal doesn't affect indices
        removed_any = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if block i is redundant: all its t-subsets are covered by other blocks
            block = improved[i]
            redundant = True
            for s in itertools.combinations(block, t):
                if cover_count[s] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for s in itertools.combinations(block, t):
                    cover_count[s] -= 1
                improved.pop(i)
                removed_any = True
        
        if not removed_any:
            break
    
    return improved

def _local_search(blocks, v, k, t, all_tsubsets, deadline):
    """
    Local search: try swapping one block for another to reduce the count.
    We try replacing a redundant-ish block with a new one that covers more uncovered triples.
    """
    if not blocks or len(blocks) < 10:
        return blocks
    
    improved = list(blocks)
    all_tset = set(all_tsubsets)
    
    # Build coverage count
    cover_count = Counter()
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1
    
    # Find blocks with low coverage contribution (most triples covered only once)
    block_scores = []
    for i, block in enumerate(improved):
        score = 0
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 2:
                score += 1
        block_scores.append((score, i))
    
    # Sort by score descending (most important blocks first)
    block_scores.sort(reverse=True)
    
    rng = random.Random(time.time())
    
    # Try to replace less important blocks
    for _, idx in block_scores[-min(20, len(block_scores)):]:
        if time.time() > deadline:
            break
        
        # Try removing this block and seeing if we can add a better one
        test_blocks = improved[:idx] + improved[idx+1:]
        test_cover_count = Counter()
        for block in test_blocks:
            for s in itertools.combinations(block, t):
                test_cover_count[s] += 1
        
        # Find uncovered triples after removal
        uncovered = set()
        for s in all_tset:
            if test_cover_count[s] == 0:
                uncovered.add(s)
        
        if not uncovered:
            # Block was redundant, remove it
            improved = test_blocks
            continue
        
        # Try to find a new block that covers many uncovered triples
        best_new_block = None
        best_gain = 0
        
        # Sample some candidate blocks
        points = list(range(v))
        for _ in range(50):
            candidate = set(rng.sample(points, k))
            gain = 0
            for s in itertools.combinations(sorted(candidate), t):
                if s in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_new_block = tuple(sorted(candidate))
        
        if best_new_block and best_gain >= len(uncovered) * 0.5:
            # Replace the block
            improved[idx] = best_new_block
    
    return improved

def _solve_ilp(initial_blocks, v, k, t, all_tsubsets, deadline):
    """
    Solve a reduced ILP to find a minimum subset of the initial_blocks
    that still covers all t-subsets.
    """
    if not initial_blocks:
        return None
    
    n_blocks = len(initial_blocks)
    n_triples = len(all_tsubsets)
    
    if n_blocks < 10 or n_triples < 100:
        return None
    
    # Create a mapping from triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(all_tsubsets)}
    
    # Build the constraint matrix A (n_triples x n_blocks)
    # A[i, j] = 1 if triple i is in block j, else 0
    A_rows = []
    for j, block in enumerate(initial_blocks):
        for s in itertools.combinations(block, t):
            idx = triple_to_idx.get(s)
            if idx is not None:
                A_rows.append((idx, j))
    
    if not A_rows:
        return None
    
    # Use sparse matrix format
    row_indices = [r for r, c in A_rows]
    col_indices = [c for r, c in A_rows]
    data = [1.0] * len(A_rows)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_triples, n_blocks))
    
    # Objective: minimize sum of x_j
    c = np.ones(n_blocks)
    
    # Constraints: A @ x >= 1
    lb = np.ones(n_triples)
    ub = np.full(n_triples, np.inf)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_j <= 1
    integrality = np.ones(n_blocks)  # All variables are integer
    bounds = Bounds(0, 1)
    
    time_limit = max(1.0, deadline - time.time() - 3.0)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': time_limit}
        )
        
        if result.status in [0, 1] and result.x is not None:
            selected = [initial_blocks[j] for j in range(n_blocks) if result.x[j] > 0.5]
            if selected:
                # Verify validity
                if _is_valid_cover(selected, v, k, t, all_tsubsets):
                    return selected
    
    except Exception:
        pass
    
    return None

def _is_valid_cover(blocks, v, k, t, all_tsubsets):
    """Check if the given blocks form a valid covering."""
    covered = set()
    for block in blocks:
        for s in itertools.combinations(block, t):
            covered.add(s)
    return covered == set(all_tsubsets)

