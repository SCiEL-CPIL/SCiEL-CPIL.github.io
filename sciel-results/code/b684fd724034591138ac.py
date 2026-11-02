"""Covering design C(19,8,5): Multi-phase optimization.
Strategy:
1. Greedy initial solution with multiple restarts.
2. Iterative redundancy removal.
3. Local search with block swaps (1-point swaps).
4. Time management with explicit deadline.
"""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin (150s limit, keep 40s buffer)
    
    # Precompute all t-subsets and their indices
    all_t = list(itertools.combinations(range(v), t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-subsets (blocks)
    all_k = list(itertools.combinations(range(v), k))
    N_k = len(all_k)
    
    # For each block, precompute which t-subsets it covers (as a bitset or list)
    # Since N_t = C(19,5) = 11628, we can use a numpy array of uint16 or just a list
    block_covers = []
    for b in all_k:
        covered = []
        for s in itertools.combinations(b, t):
            covered.append(t_idx[s])
        block_covers.append(covered)
    
    # Helper: get coverage count for a set of blocks
    def get_coverage_counts(blocks):
        counts = np.zeros(N_t, dtype=np.int32)
        for b in blocks:
            for s in itertools.combinations(b, t):
                counts[t_idx[s]] += 1
        return counts
    
    # Helper: check if a block is redundant given coverage counts
    def is_redundant(block, counts):
        for s in itertools.combinations(block, t):
            if counts[t_idx[s]] == 1:
                return False
        return True
    
    # Helper: remove redundant blocks iteratively
    def remove_redundant(blocks):
        changed = True
        while changed:
            changed = False
            counts = get_coverage_counts(blocks)
            new_blocks = []
            for b in blocks:
                if is_redundant(b, counts):
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks
    
    # Greedy initial solution
    def greedy_initial(seed, deadline):
        random.seed(seed)
        uncovered = set(all_t)
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset to start
            if random.random() < 0.3:
                # Greedy: pick the t-subset that appears in the most blocks
                # Approximate: just pick min for speed
                seed_t = min(uncovered)
            else:
                seed_t = random.choice(list(uncovered))
            
            # Build a block around seed_t
            block_set = set(seed_t)
            
            # Greedily add points to maximize new coverage
            while len(block_set) < k:
                best_gain = -1
                best_pts = []
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many new t-subsets would be covered
                    gain = 0
                    # Need to count t-subsets containing x and (t-1) elements from block_set
                    for rest in itertools.combinations(sorted(block_set), t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pts = [x]
                    elif gain == best_gain:
                        best_pts.append(x)
                
                if not best_pts:
                    break
                block_set.add(random.choice(best_pts))
            
            if len(block_set) < k:
                remaining = [x for x in range(v) if x not in block_set]
                random.shuffle(remaining)
                block_set.update(remaining[:k - len(block_set)])
            
            block = tuple(sorted(block_set))
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
        
        return blocks
    
    # Local search: try to swap one point in a block
    def local_search(blocks, deadline):
        current_blocks = list(blocks)
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline and no_improve < max_no_improve:
            counts = get_coverage_counts(current_blocks)
            improved = False
            
            # Try to remove redundant blocks first
            new_blocks = []
            for b in current_blocks:
                if is_redundant(b, counts):
                    pass  # Skip redundant
                else:
                    new_blocks.append(b)
            if len(new_blocks) < len(current_blocks):
                current_blocks = new_blocks
                improved = True
                no_improve = 0
                continue
            
            # Try 1-point swaps
            random.shuffle(current_blocks)
            for i, b in enumerate(current_blocks):
                if time.time() > deadline:
                    break
                
                # Try swapping each point in b
                for j in range(k):
                    p_out = b[j]
                    b_partial = [x for x in b if x != p_out]
                    
                    for p_in in range(v):
                        if p_in in b_partial:
                            continue
                        b_new = tuple(sorted(b_partial + [p_in]))
                        
                        # Check if b_new is already in current_blocks
                        if b_new in current_blocks:
                            continue
                        
                        # Simulate the swap
                        # Remove b, add b_new
                        # Check if coverage is maintained and if any block becomes redundant
                        
                        # Quick check: does b_new cover all that b covered that was uniquely covered?
                        # More precisely: after swap, is coverage still complete?
                        # This is expensive. Use heuristic: accept if b_new covers 
                        # at least as many uniquely covered t-subsets as b did.
                        
                        # Count unique coverage of b
                        unique_b = 0
                        for s in itertools.combinations(b, t):
                            if counts[t_idx[s]] == 1:
                                unique_b += 1
                        
                        # Count how many of those unique_b are covered by b_new
                        covered_by_new = 0
                        for s in itertools.combinations(b, t):
                            if counts[t_idx[s]] == 1:
                                if all(x in b_new for x in s):
                                    covered_by_new += 1
                        
                        # Accept if b_new covers all unique parts, or if it might enable removal
                        if covered_by_new == unique_b:
                            # Do the swap
                            current_blocks[i] = b_new
                            improved = True
                            no_improve = 0
                            break
                        
                        # Also try: does b_new allow removing another block?
                        # This is more expensive; skip for now
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return current_blocks
    
    # Main: run greedy with multiple restarts, then local search
    best_blocks = None
    best_size = float('inf')
    
    # Phase 1: Greedy restarts (30s budget)
    greedy_deadline = t0 + 30.0
    for seed in range(50):
        if time.time() > greedy_deadline:
            break
        try:
            blocks = greedy_initial(seed, greedy_deadline)
            blocks = remove_redundant(blocks)
            if blocks and len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
        except Exception:
            pass
    
    if best_blocks is None:
        # Fallback: return empty (invalid, but shouldn't happen)
        return []
    
    # Phase 2: Local search (remaining time)
    if time.time() < deadline - 5:
        try:
            ls_blocks = local_search(best_blocks, deadline - 5)
            ls_blocks = remove_redundant(ls_blocks)
            # Verify validity
            counts = get_coverage_counts(ls_blocks)
            if np.all(counts > 0):
                if len(ls_blocks) < best_size:
                    best_blocks = ls_blocks
                    best_size = len(ls_blocks)
        except Exception:
            pass
    
    return [list(b) for b in best_blocks]
