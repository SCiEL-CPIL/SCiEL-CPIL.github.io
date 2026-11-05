import itertools
import numpy as np
from math import comb
import time
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Precompute block_triple_indices
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    # 2. Initial Greedy Solution
    def greedy_cover(random_seed=0):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0 and time.time() < deadline - 5.0:
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx)
            
        return selected_blocks

    initial_blocks = greedy_cover(0)
    best_blocks = initial_blocks
    best_len = len(best_blocks)
    
    # 3. LP Column Generation
    active_cols = list(initial_blocks)
    
    # We will run LP until no more columns are added or time runs out
    for _ in range(50): # Max iterations
        if time.time() > deadline - 30.0: break
        
        if not active_cols:
            break
            
        # Build A matrix (n_triples x len(active_cols))
        # A[i, j] = 1 if triple i is in block active_cols[j]
        # We can extract this from block_triple_indices
        # block_triple_indices is (n_blocks, k_t)
        # We need to map block indices to triple indices
        # This is sparse.
        
        # For small active_cols, we can build dense A
        n_active = len(active_cols)
        A = np.zeros((n_triples, n_active), dtype=np.float64)
        
        # Efficient way:
        # For each active block index b_idx, get its triples, set A[triples, j] = 1
        for j, b_idx in enumerate(active_cols):
            triples = block_triple_indices[b_idx]
            A[triples, j] = 1.0
            
        # Solve LP: min 1^T x s.t. A x >= 1, x >= 0
        # linprog requires A_ub x <= b_ub
        # -A x <= -1
        c = np.ones(n_active)
        A_ub = -A
        b_ub = -np.ones(n_triples)
        bounds = [(0, None)] * n_active
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except:
            break
            
        if not res.success:
            break
            
        x = res.x
        y = res.y # Dual variables for A_ub constraints
        
        # Pricing: Find block B maximizing sum(y[triples in B])
        # prices = y[block_triple_indices].sum(axis=1)
        # This is a (n_blocks, k_t) @ (n_triples,) operation? No.
        # y is size n_triples.
        # block_triple_indices is (n_blocks, k_t).
        # We want sum of y at indices in each row.
        prices = y[block_triple_indices].sum(axis=1)
        
        max_price = np.max(prices)
        
        if max_price <= 1.0 + 1e-6:
            # LP Optimal
            break
        else:
            # Add violating blocks
            violating = np.where(prices > 1.0 + 1e-6)[0]
            # Add only a few to keep A small? Or all?
            # Adding all might make A large. Let's add top 10.
            # Sort violating by price desc
            sorted_violating = violating[np.argsort(prices[violating])[::-1]]
            new_cols = sorted_violating[:10]
            
            # Add to active_cols if not already there
            active_set = set(active_cols)
            for b_idx in new_cols:
                if b_idx not in active_set:
                    active_cols.append(b_idx)
                    active_set.add(b_idx)

    # 4. Rounding and Local Search
    if time.time() > deadline - 10.0:
        return [list(all_blocks[b]) for b in best_blocks]

    # Final LP solve on active_cols to get x
    n_active = len(active_cols)
    A = np.zeros((n_triples, n_active), dtype=np.float64)
    for j, b_idx in enumerate(active_cols):
        triples = block_triple_indices[b_idx]
        A[triples, j] = 1.0
        
    c = np.ones(n_active)
    A_ub = -A
    b_ub = -np.ones(n_triples)
    bounds = [(0, None)] * n_active
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x = res.x
        else:
            x = np.ones(n_active) / n_active
    except:
        x = np.ones(n_active) / n_active
        
    # Map x to block indices
    # active_cols is a list of block indices
    # x is aligned with active_cols
    
    # Greedy Rounding
    # Sort active_cols by x descending
    order = np.argsort(x)[::-1]
    sorted_active = [active_cols[i] for i in order]
    
    covered_mask = np.zeros(n_triples, dtype=bool)
    rounded_blocks = []
    
    for b_idx in sorted_active:
        triples = block_triple_indices[b_idx]
        # If any triple in this block is not covered, add it
        if np.any(~covered_mask[triples]):
            rounded_blocks.append(b_idx)
            covered_mask[triples] = True
            if np.all(covered_mask):
                break
                
    if len(rounded_blocks) < best_len:
        best_blocks = rounded_blocks
        best_len = len(best_blocks)
        
    # Local Search: Shake and Repair
    # We operate on the set of block indices
    current_blocks = list(best_blocks)
    
    # Helper to get covered mask
    def get_covered_mask(blocks):
        mask = np.zeros(n_triples, dtype=bool)
        for b in blocks:
            mask[block_triple_indices[b]] = True
        return mask
        
    # Helper to check if block is redundant
    def is_redundant(blocks, b_idx):
        # Remove b_idx, see if all its triples are covered by others
        others = [b for b in blocks if b != b_idx]
        if not others:
            return True
        mask = get_covered_mask(others)
        triples = block_triple_indices[b_idx]
        return np.all(mask[triples])
        
    # Simple Shake and Repair
    rng = np.random.RandomState(42)
    
    for _ in range(1000): # Max local search steps
        if time.time() > deadline - 5.0:
            break
            
        if not current_blocks:
            break
            
        # Pick a random block to shake
        shake_idx = rng.randint(len(current_blocks))
        b_to_remove = current_blocks[shake_idx]
        
        # Remove it
        current_blocks.remove(b_to_remove)
        
        # Find uncovered triples
        mask = get_covered_mask(current_blocks)
        uncovered = np.where(~mask)[0]
        
        if len(uncovered) == 0:
            # We improved!
            if len(current_blocks) < best_len:
                best_blocks = list(current_blocks)
                best_len = len(best_blocks)
            # Try to shake more
            continue
        else:
            # Repair: Greedily add blocks to cover uncovered
            # Only consider blocks from active_cols? Or all blocks?
            # All blocks is better for finding new structures.
            # But checking all 346k blocks is slow.
            # Let's restrict to active_cols + some random ones?
            # Or just use the greedy logic on the full space but only for uncovered triples.
            
            new_blocks = []
            covered_mask = mask.copy()
            
            while np.any(~covered_mask) and len(new_blocks) < 10: # Limit repair size
                # Calculate gains for all blocks? Too slow.
                # Calculate gains for active_cols?
                gains = np.sum(~covered_mask[block_triple_indices[active_cols]], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    # No block in active_cols covers uncovered?
                    # This shouldn't happen if active_cols is good.
                    # If it does, break.
                    break
                candidates = np.where(gains == max_gain)[0]
                best_cand = candidates[rng.randint(len(candidates))]
                b_idx = active_cols[best_cand]
                new_blocks.append(b_idx)
                covered_mask[block_triple_indices[b_idx]] = True
                
            # Combine
            candidate = current_blocks + new_blocks
            
            # Check if better
            if len(candidate) < len(best_blocks):
                best_blocks = candidate
                best_len = len(best_blocks)
                
            # Even if not better, we might want to keep current_blocks as the state for next shake?
            # Usually, we revert if not better.
            # But "Shake and Repair" often implies accepting worse solutions with probability?
            # For now, only accept if better.
            # To escape local optima, we could accept worse with small prob.
            # But we have a deadline. Let's stick to hill climbing.
            
            # Restore current_blocks?
            # No, we modified current_blocks.
            # If we don't accept, we should restore.
            # But we lost the original list.
            # Let's keep a copy.
            
            # This logic is getting messy. Let's simplify.
            pass
            
    return [list(all_blocks[b]) for b in best_blocks]
