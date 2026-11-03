import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 118.0

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
    # Block coverage: block_cover[bi] is a list of t-subset indices
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)

    # Reverse map: t_to_blocks[ti] is list of block indices covering ti
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    # Precompute block sizes (constant = C(k,t))
    block_size = len(block_cover[0])

    # --- Initialization: Randomized Greedy ---
    def initialize():
        # Start with a set of random blocks to create variety, then fill gaps with greedy
        # Or just pure greedy with randomized tie-breaking?
        # Let's do: Pick a seed, run greedy, but allow some randomness in start?
        # Actually, standard greedy is strong. Let's do "Randomized Greedy":
        # 1. Start with empty set
        # 2. At each step, among the best candidates (top 10% coverage), pick one randomly.
        
        uncovered = set(range(N))
        blocks = []
        rng = random.Random(42)
        
        while uncovered:
            # Find max coverage
            # Optimization: Pick a random seed from uncovered to narrow candidates
            seed = next(iter(uncovered))
            candidates = t_to_blocks[seed]
            
            # Calculate coverage for candidates
            covs = []
            for bi in candidates:
                c = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        c += 1
                if c > 0:
                    covs.append((c, bi))
            
            if not covs:
                break
                
            # Sort by coverage desc
            covs.sort(reverse=True)
            max_cov = covs[0][0]
            
            # Select from top fraction (e.g. top 20% or at least 5)
            top_count = max(5, len(covs) // 5)
            top_candidates = covs[:top_count]
            
            # Weighted random pick? Or uniform?
            # Uniform from top candidates introduces diversity
            chosen_cov, chosen_b = top_candidates[rng.randrange(len(top_candidates))]
            
            blocks.append(chosen_b)
            for ti in block_cover[chosen_b]:
                uncovered.discard(ti)
                
        return blocks

    # --- Local Search: Simulated Annealing ---
    def local_search(blocks_in, max_iters, T_init, T_min):
        current_blocks = list(blocks_in)
        if not current_blocks:
            return current_blocks
        
        # State
        cnt = np.zeros(N, dtype=np.int32)
        in_set = np.zeros(B, dtype=bool)
        for bi in current_blocks:
            in_set[bi] = True
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        # Validity check
        if np.min(cnt) < 1:
            # Repair is needed. This shouldn't happen if init is valid, but just in case.
            # Simple greedy repair
            uncovered = set(np.where(cnt == 0)[0])
            while uncovered and time.time() < deadline - 5.0:
                seed = next(iter(uncovered))
                best_b = -1
                max_cov = -1
                for bi in t_to_blocks[seed]:
                    if in_set[bi]: continue
                    c = 0
                    for ti in block_cover[bi]:
                        if ti in uncovered: c += 1
                    if c > max_cov:
                        max_cov = c
                        best_b = bi
                if best_b == -1: break
                current_blocks.append(best_b)
                in_set[best_b] = True
                for ti in block_cover[best_b]:
                    cnt[ti] += 1
                    uncovered.discard(ti)
            # Re-verify
            if np.min(cnt) < 1:
                return initialize() # Fallback to fresh init

        best_blocks = list(current_blocks)
        best_cost = len(best_blocks)
        
        rng = random.Random(100)
        T = T_init
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Generate Neighbor
            new_blocks = list(current_blocks)
            new_cost = len(new_blocks)
            
            # Random Move Type
            move_type = rng.randint(0, 2)
            
            if move_type == 0:
                # Try to Remove a block (cost -1)
                if not new_blocks:
                    continue
                idx_rm = rng.randrange(len(new_blocks))
                b_rm = new_blocks[idx_rm]
                
                # Check if b_rm is removable (no unique coverage)
                removable = True
                for ti in block_cover[b_rm]:
                    if cnt[ti] == 1:
                        removable = False
                        break
                        
                if removable:
                    new_blocks.pop(idx_rm)
                    new_cost -= 1
                    # Update counts (tentatively)
                    # We need to check validity against the *new* set of blocks?
                    # If it's removable in the current state, removing it keeps validity.
                    # So we can accept this move immediately if we want, or evaluate cost.
                    # Cost change is -1. This is always an improvement!
                    # So we apply it directly to current state.
                    
                    # Apply Move to Current State
                    current_blocks.pop(idx_rm)
                    in_set[b_rm] = False
                    for ti in block_cover[b_rm]:
                        cnt[ti] -= 1
                    
                    current_blocks = new_blocks
                    best_blocks = list(current_blocks)
                    best_cost = len(current_blocks)
                    T = max(T_min, T * 0.99)
                    continue
            else:
                # 1-Exchange or 2-Exchange?
                # Let's focus on 1-Exchange (Swap) and 2-Exchange (2-for-1)
                if move_type == 1:
                    # 1-Exchange: Remove b1, Add b2. Cost same.
                    # Only useful if it allows future removals.
                    # Heuristic: Pick b1 that is "redundant" (covers many already-covered sets)
                    # and b2 that covers unique sets well?
                    # Actually, 1-exchange with same cost is neutral.
                    # We use SA to accept neutral moves to escape local minima.
                    
                    if len(new_blocks) < 1:
                        continue
                        
                    idx1 = rng.randrange(len(new_blocks))
                    b1 = new_blocks[idx1]
                    
                    # Pick b2 randomly? Or from candidates covering some of b1's "weak" points?
                    # Let's pick b2 randomly from all blocks not in set.
                    # This is expensive to check membership.
                    # Let's just pick a random block index and check.
                    b2 = rng.randrange(B)
                    while in_set[b2]:
                        b2 = rng.randrange(B)
                    
                    # Check if swap is valid:
                    # After removing b1, are all sets still covered?
                    # For all ti in Cover(b1):
                    #   new_cnt[ti] = cnt[ti] - 1 + (1 if ti in Cover(b2) else 0)
                    #   We need new_cnt[ti] >= 1
                    #   => cnt[ti] + (1 if ti in Cover(b2) else 0) >= 2
                    #   => If cnt[ti] == 1, then ti MUST be in Cover(b2).
                    
                    valid = True
                    for ti in block_cover[b1]:
                        if cnt[ti] == 1:
                            # b1 is the only cover for ti
                            # So b2 MUST cover ti
                            if ti not in block_cover[b2]: # Set membership check is slow?
                                valid = False
                                break
                            # If b2 covers ti, then new_cnt[ti] = 1 - 1 + 1 = 1 (OK)
                        else:
                            # cnt[ti] >= 2
                            # Even if b2 doesn't cover ti, new_cnt[ti] = cnt[ti] - 1 >= 1 (OK)
                            # If b2 covers ti, new_cnt[ti] = cnt[ti] - 1 + 1 = cnt[ti] >= 2 (OK)
                            pass
                    
                    if not valid:
                        continue
                        
                    # Calculate Delta Cost
                    # Cost is same (len - 1 + 1 = len)
                    # But we can define a secondary objective: Minimize total coverage count?
                    # Sum(cnt)
                    # Old Sum = S
                    # New Sum = S - sum(1 for ti in Cover(b1)) + sum(1 for ti in Cover(b2))
                    # Actually, we only care about the change.
                    # But since cost is primary, and it's 0, we accept with probability p.
                    # To make it useful, let's accept if it reduces "over-coverage".
                    # Over-coverage metric: sum(max(0, cnt[ti]-1))
                    # Let's compute delta over-coverage.
                    
                    # This calculation is expensive.
                    # Alternative: Just accept with probability exp(1/T) for neutral moves?
                    # Or just always accept neutral moves? (Random walk)
                    # Let's try: Accept if it reduces the sum of counts?
                    # sum_counts_old = sum(cnt)
                    # sum_counts_new = sum_counts_old - block_size + block_size = sum_counts_old
                    # Wait, block_size is constant.
                    # So sum of counts is invariant under 1-exchange?
                    # No.
                    # Cover(b1) has size K. Cover(b2) has size K.
                    # We remove K counts, add K counts. Sum is same.
                    # So 1-exchange preserves sum of counts.
                    # It changes the *distribution* of counts.
                    # We want to concentrate coverage to make removals easier.
                    # So we want to minimize the number of sets with cnt=1?
                    # Or maximize cnt for some sets so they can be "shared"?
                    # Actually, to remove a block, we need NO sets with cnt=1 in its cover.
                    # So we want to make blocks "redundant".
                    # If we swap b1 (which has some unique points) for b2 (which shares more with others),
                    # we might increase the minimum count in b1's domain?
                    
                    # Let's just use a simple heuristic: Accept if b2 covers more of b1's unique points?
                    # We already ensured validity.
                    # Let's just accept all valid 1-exchanges with probability 0.5?
                    # Or use SA with delta = 0?
                    if rng.random() < 0.5:
                        # Apply Swap
                        new_blocks[idx1] = b2
                        current_blocks = new_blocks
                        
                        # Update in_set and cnt
                        in_set[b1] = False
                        for ti in block_cover[b1]:
                            cnt[ti] -= 1
                        in_set[b2] = True
                        for ti in block_cover[b2]:
                            cnt[ti] += 1
                else:
                    # 2-Exchange: Remove b1, b2; Add b3. Cost -1.
                    if len(new_blocks) < 2:
                        continue
                        
                    idx1, idx2 = rng.sample(range(len(new_blocks)), 2)
                    b1 = new_blocks[idx1]
                    b2 = new_blocks[idx2]
                    
                    # Identify Uncovered U if b1, b2 removed
                    U = set()
                    # Union of covers
                    c1 = set(block_cover[b1])
                    c2 = set(block_cover[b2])
                    union = c1 | c2
                    
                    for ti in union:
                        c1_flag = 1 if ti in c1 else 0
                        c2_flag = 1 if ti in c2 else 0
                        if cnt[ti] - c1_flag - c2_flag == 0:
                            U.add(ti)
                    
                    if U:
                        # Find b3 that covers U
                        seed = next(iter(U))
                        best_b3 = -1
                        max_cov = -1
                        for bi in t_to_blocks[seed]:
                            if in_set[bi]: continue
                            c = 0
                            for ti in block_cover[bi]:
                                if ti in U: c += 1
                            if c > max_cov:
                                max_cov = c
                                best_b3 = bi
                                
                        if best_b3 == -1 or max_cov < len(U):
                            continue
                            
                        # Valid 2-exchange found! Cost -1.
                        # Apply immediately.
                        # Remove b1, b2
                        # Be careful with indices
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        # Remove higher index first
                        b_rm1 = current_blocks[idx1]
                        b_rm2 = current_blocks[idx2]
                        
                        current_blocks.pop(idx1)
                        in_set[b_rm1] = False
                        for ti in block_cover[b_rm1]:
                            cnt[ti] -= 1
                        
                        current_blocks.pop(idx2)
                        in_set[b_rm2] = False
                        for ti in block_cover[b_rm2]:
                            cnt[ti] -= 1
                        
                        # Add b3
                        current_blocks.append(best_b3)
                        in_set[best_b3] = True
                        for ti in block_cover[best_b3]:
                            cnt[ti] += 1
                            
                        best_blocks = list(current_blocks)
                        best_cost = len(current_blocks)
                        T = max(T_min, T * 0.99)
                        continue
                    else:
                        # U is empty, so we can just remove b1 and b2 (Cost -2)
                        # This is effectively two 1-removals.
                        # We already handled 1-removal in move_type 0.
                        # Why did move_type 0 not catch this?
                        # Because move_type 0 only tries to remove 1 block.
                        # If both b1 and b2 are individually non-removable, but jointly removable,
                        # this path catches it.
                        # Apply removal of both.
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        b_rm1 = current_blocks[idx1]
                        b_rm2 = current_blocks[idx2]
                        
                        current_blocks.pop(idx1)
                        in_set[b_rm1] = False
                        for ti in block_cover[b_rm1]:
                            cnt[ti] -= 1
                        
                        current_blocks.pop(idx2)
                        in_set[b_rm2] = False
                        for ti in block_cover[b_rm2]:
                            cnt[ti] -= 1
                            
                        best_blocks = list(current_blocks)
                        best_cost = len(current_blocks)
                        T = max(T_min, T * 0.99)
                        continue

            # Cooling
            T = max(T_min, T * 0.995)
            
        return best_blocks

    # --- Main Loop ---
    best_overall = initialize()
    best_cost = len(best_overall)
    
    # Run local search with increasing budget or fixed budget?
    # Let's run multiple restarts.
    rng = random.Random(42)
    
    # Initial Local Search
    best_overall = local_search(best_overall, max_iters=2000, T_init=10.0, T_min=0.1)
    best_cost = len(best_overall)
    
    # Iterated Local Search
    while time.time() < deadline - 10.0:
        # Perturb
        # Remove 10% of blocks and repair?
        # Or just swap some blocks?
        # Let's try: Take best, remove a few random blocks, repair greedily.
        
        perturbed = list(best_overall)
        if not perturbed:
            break
            
        # Remove 5-10%
        num_rm = max(1, int(len(perturbed) * rng.uniform(0.05, 0.15)))
        indices_to_rm = rng.sample(range(len(perturbed)), num_rm)
        for i in sorted(indices_to_rm, reverse=True):
            perturbed.pop(i)
            
        # Repair
        # Rebuild counts
        cnt = np.zeros(N, dtype=np.int32)
        in_set = np.zeros(B, dtype=bool)
        for bi in perturbed:
            in_set[bi] = True
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        uncovered = set(np.where(cnt == 0)[0])
        
        while uncovered and time.time() < deadline - 5.0:
            seed = next(iter(uncovered))
            best_b = -1
            max_cov = -1
            for bi in t_to_blocks[seed]:
                if in_set[bi]: continue
                c = 0
                for ti in block_cover[bi]:
                    if ti in uncovered: c += 1
                if c > max_cov:
                    max_cov = c
                    best_b = bi
            if best_b == -1: break
            perturbed.append(best_b)
            in_set[best_b] = True
            for ti in block_cover[best_b]:
                cnt[ti] += 1
                uncovered.discard(ti)
                
        # Local Search on perturbed
        improved = local_search(perturbed, max_iters=1000, T_init=5.0, T_min=0.1)
        
        if len(improved) < best_cost:
            best_cost = len(improved)
            best_overall = improved

    # Final Prune
    def prune(blocks):
        if not blocks:
            return blocks
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

    best_overall = prune(best_overall)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_overall:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in initialize()]

    return [list(all_blocks[bi]) for bi in best_overall]
