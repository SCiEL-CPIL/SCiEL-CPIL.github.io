"""
Randomized-rounding + local-search heuristic for C(19,7,4).

Strategy:
  1. Generate a pool of random 7-subsets of [0..18].
  2. Solve the LP relaxation (minimize sum x_i s.t. A x >= 1, 0<=x<=1)
     using scipy.optimize.linprog.
  3. Round the LP solution to get an initial covering (greedy on residual).
  4. Refine with local search: try replacing blocks, removing redundant ones,
     and 2-exchanges to reduce block count.
"""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

def construct(v, k, t):
    deadline = time.time() + 145  # safety margin before 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map t-subset to index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Generate pool of candidate blocks
    # We need enough candidates to have a good chance of covering everything
    # C(19,7) = 50388, so we can sample a subset
    rng = np.random.default_rng(42)
    
    # First, try a structured set: all blocks from a few random permutations
    # plus random sampling
    pool = set()
    
    # Add some systematic blocks
    # Random permutations give different 7-subsets
    for trial in range(50):
        perm = rng.permutation(v)
        for i in range(v - k + 1):
            block = tuple(sorted(perm[i:i+k]))
            pool.add(block)
    
    # Add random blocks
    while len(pool) < 5000:
        block = tuple(sorted(rng.choice(v, size=k, replace=False)))
        pool.add(block)
    
    pool = list(pool)
    n_blocks = len(pool)
    
    # Build incidence matrix A: n_tsubs x n_blocks, A[i,j] = 1 if tsub i is in block j
    # This is sparse, but let's build it efficiently
    A = np.zeros((n_tsubs, n_blocks), dtype=np.int8)
    for j, block in enumerate(pool):
        block_set = set(block)
        for tsub in all_tsubs:
            if all(p in block_set for p in tsub):
                A[tsub_to_idx[tsub], j] = 1
    
    # Solve LP relaxation: min sum(x_i) s.t. A x >= 1, 0 <= x <= 1
    c = np.ones(n_blocks)
    A_ub = -A  # -A x <= -1  i.e. A x >= 1
    b_ub = -np.ones(n_tsubs)
    bounds = [(0, 1)] * n_blocks
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            lp_solution = res.x
        else:
            lp_solution = None
    except Exception:
        lp_solution = None
    
    # Get initial covering
    blocks = []
    uncovered = bytearray(b'\x01') * n_tsubs
    uncovered_count = n_tsubs
    
    # If LP solution available, use it to guide selection
    if lp_solution is not None:
        # Sort blocks by LP weight (descending)
        order = np.argsort(-lp_solution)
        for idx in order:
            if time.time() > deadline - 10:
                break
            block = pool[idx]
            # Check if this block covers any uncovered t-subsets
            gain = 0
            for tsub in itertools.combinations(block, t):
                i = tsub_to_idx[tsub]
                if uncovered[i]:
                    gain += 1
            if gain > 0:
                blocks.append(list(block))
                for tsub in itertools.combinations(block, t):
                    i = tsub_to_idx[tsub]
                    if uncovered[i]:
                        uncovered[i] = 0
                        uncovered_count -= 1
                if uncovered_count == 0:
                    break
    
    # Greedy completion for any remaining uncovered t-subsets
    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break
        
        # Find first uncovered t-subset
        first_uncovered = -1
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered = i
                break
        if first_uncovered == -1:
            break
        
        # Find best block to add (maximizes coverage of uncovered t-subsets)
        best_block = None
        best_gain = -1
        
        # Try blocks from pool first
        for j, block in enumerate(pool):
            gain = 0
            for tsub in itertools.combinations(block, t):
                i = tsub_to_idx[tsub]
                if uncovered[i]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                if gain == 0:
                    break
        
        if best_block is None or best_gain <= 0:
            # Generate a new random block
            for _ in range(100):
                block = tuple(sorted(rng.choice(v, size=k, replace=False)))
                gain = 0
                for tsub in itertools.combinations(block, t):
                    i = tsub_to_idx[tsub]
                    if uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                if best_gain > 0:
                    break
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        for tsub in itertools.combinations(best_block, t):
            i = tsub_to_idx[tsub]
            if uncovered[i]:
                uncovered[i] = 0
                uncovered_count -= 1
    
    # Verify coverage
    if uncovered_count > 0:
        # Emergency: add blocks to cover remaining
        while uncovered_count > 0:
            if time.time() > deadline - 1:
                break
            first_uncovered = -1
            for i in range(n_tsubs):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            # Just add a block containing this t-subset
            tsub = all_tsubs[first_uncovered]
            block = list(tsub)
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            for tsub2 in itertools.combinations(block, t):
                i = tsub_to_idx[tsub2]
                if uncovered[i]:
                    uncovered[i] = 0
                    uncovered_count -= 1
    
    # Local search refinement: try to remove redundant blocks
    if time.time() < deadline - 15:
        # Try removing blocks that are not essential
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                # Check if block i is redundant
                temp_uncovered = bytearray(b'\x00') * n_tsubs
                # Mark what would be uncovered if we remove block i
                # First, compute current coverage without block i
                for j, block in enumerate(blocks):
                    if j == i:
                        continue
                    for tsub in itertools.combinations(block, t):
                        temp_uncovered[tsub_to_idx[tsub]] = 1  # covered
                
                # Check if any t-subset is not covered
                redundant = True
                for idx2, val in enumerate(temp_uncovered):
                    if val == 0:
                        redundant = False
                        break
                
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
    
    # Try 1-exchange: replace a block with a better one
    if time.time() < deadline - 5:
        for _ in range(50):  # limited iterations
            if time.time() > deadline - 5:
                break
            # Pick a random block to try replacing
            if len(blocks) == 0:
                break
            i = rng.integers(0, len(blocks))
            old_block = blocks[i]
            
            # Try replacing with a random block
            new_block = tuple(sorted(rng.choice(v, size=k, replace=False)))
            
            # Check if new_block is already in blocks
            if any(tuple(sorted(b)) == new_block for b in blocks if b != old_block):
                continue
            
            # Check coverage: would removing old_block and adding new_block still cover everything?
            # Compute coverage without old_block
            coverage_without_old = bytearray(b'\x00') * n_tsubs
            for j, block in enumerate(blocks):
                if j == i:
                    continue
                for tsub in itertools.combinations(block, t):
                    coverage_without_old[tsub_to_idx[tsub]] = 1
            
            # Check what new_block covers
            new_covers = set()
            for tsub in itertools.combinations(new_block, t):
                new_covers.add(tsub_to_idx[tsub])
            
            # Check if all t-subsets are covered
            valid = True
            for idx2 in range(n_tsubs):
                if coverage_without_old[idx2] == 0 and idx2 not in new_covers:
                    valid = False
                    break
            
            if valid:
                blocks[i] = list(new_block)
    
    return blocks
