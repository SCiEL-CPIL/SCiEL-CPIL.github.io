import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0

    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    def greedy_construct(v, k, t, seed=0, bias=0.0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                
                sorted_pts = sorted(gains.items(), key=lambda x: -x[1])
                best_point = sorted_pts[0][0]
                
                if bias > 0:
                    r = rng.random()
                    if r < bias:
                        if len(sorted_pts) > 1 and r < bias * 0.5:
                            best_point = sorted_pts[1][0]
                        else:
                            candidates = [p for p in range(v) if p not in block_set]
                            best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # Phase 1: Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        bias = 0.2 + 0.05 * seed
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # Phase 2: Local Search
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def get_block_triplets(block):
        return set(get_id(triplet) for triplet in itertools.combinations(block, t))

    def try_swap_2_for_1(blocks, block_weights=None):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        block_triplet_ids = [get_block_triplets(b) for b in blocks]
        
        if block_weights is not None:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
            pair_indices.sort(key=lambda ij: block_weights[ij[0]] + block_weights[ij[1]])
        else:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for i, j in pair_indices:
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids[i]
            b2 = blocks[j]
            ids_b2 = block_triplet_ids[j]
            
            critical = set()
            for tid in ids_b1:
                c = count[tid]
                covered = (1 if tid in ids_b2 else 0) + 1
                if c == covered:
                    critical.update(all_triplets[tid])
            for tid in ids_b2:
                if tid in ids_b1:
                    continue
                c = count[tid]
                if c == 1:
                    critical.update(all_triplets[tid])
            
            if not critical:
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
            
            if len(critical) <= k:
                # Greedily build new block from critical points
                new_block = list(critical)
                new_block_set = set(new_block)
                while len(new_block) < k:
                    best_p = -1
                    best_gain = -1
                    for p in range(v):
                        if p in new_block_set:
                            continue
                        gain = 0
                        for triplet in itertools.combinations(new_block, t - 1):
                            tt = tuple(sorted(triplet + (p,)))
                            tid = get_id(tt)
                            if count[tid] <= 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    if best_p == -1:
                        break
                    new_block.append(best_p)
                    new_block_set.add(best_p)
                new_block.sort()
                
                # Verify the new block covers all critical triplets
                new_ids = get_block_triplets(new_block)
                critical_ok = True
                for tid in ids_b1 | ids_b2:
                    c = count[tid]
                    in_b1 = 1 if tid in ids_b1 else 0
                    in_b2 = 1 if tid in ids_b2 else 0
                    in_new = 1 if tid in new_ids else 0
                    if c - in_b1 - in_b2 + in_new < 1:
                        critical_ok = False
                        break
                if not critical_ok:
                    continue
                
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                new_blocks.append(new_block)
                return new_blocks, True

        return blocks, False

    def try_swap_1_for_2(blocks):
        """Replace one block with two blocks that together cover all its triplets."""
        n = len(blocks)
        if n == 0:
            return blocks, False
        
        for idx in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b = blocks[idx]
            ids_b = get_block_triplets(b)
            
            # Find triplets only covered by this block
            unique_tids = set()
            for tid in ids_b:
                if count[tid] == 1:
                    unique_tids.add(tid)
            
            if not unique_tids:
                continue
            
            # Try to split b into two blocks that cover all of b's triplets
            # Greedily: pick first t points for block1, rest for block2
            best_pair = None
            best_unique_covered = -1
            
            # Try different splits
            for split_size in range(t, k - t + 1):
                if split_size >= k:
                    break
                # Pick split_size points for block1
                subset = list(b[:split_size])
                rest = list(b[split_size:])
                
                # Extend subset to k points
                subset_set = set(subset)
                while len(subset) < k:
                    best_p = -1
                    best_gain = -1
                    for p in range(v):
                        if p in subset_set:
                            continue
                        gain = 0
                        for triplet in itertools.combinations(subset, t - 1):
                            tt = tuple(sorted(triplet + (p,)))
                            tid = get_id(tt)
                            if tid in ids_b and count[tid] == 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    if best_p == -1:
                        break
                    subset.append(best_p)
                    subset_set.add(best_p)
                
                # Extend rest to k points
                rest_set = set(rest)
                while len(rest) < k:
                    best_p = -1
                    best_gain = -1
                    for p in range(v):
                        if p in rest_set:
                            continue
                        gain = 0
                        for triplet in itertools.combinations(rest, t - 1):
                            tt = tuple(sorted(triplet + (p,)))
                            tid = get_id(tt)
                            if tid in ids_b and count[tid] == 1:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    if best_p == -1:
                        break
                    rest.append(best_p)
                    rest_set.add(best_p)
                
                subset.sort()
                rest.sort()
                
                # Check coverage
                ids_s = get_block_triplets(subset)
                ids_r = get_block_triplets(rest)
                covered_unique = sum(1 for tid in unique_tids if tid in ids_s or tid in ids_r)
                
                if covered_unique > best_unique_covered:
                    best_unique_covered = covered_unique
                    best_pair = (subset, rest)
                
                if best_unique_covered == len(unique_tids):
                    break
            
            if best_pair and best_unique_covered == len(unique_tids):
                b1, b2 = best_pair
                # Verify no other block is harmed
                ids_b1 = get_block_triplets(b1)
                ids_b2 = get_block_triplets(b2)
                ok = True
                for tid in ids_b:
                    c = count[tid]
                    in_new = (1 if tid in ids_b1 else 0) + (1 if tid in ids_b2 else 0)
                    if c - 1 + in_new < 1:
                        ok = False
                        break
                if ok:
                    update_counts([b], -1)
                    update_counts([b1, b2], 1)
                    new_blocks = [b for i2, b in enumerate(blocks) if i2 != idx]
                    new_blocks.extend([b1, b2])
                    return new_blocks, True
        
        return blocks, False

    def try_replace_block(blocks):
        """Replace a block with a different block that covers all its unique triplets."""
        n = len(blocks)
        if n == 0:
            return blocks, False
        
        for idx in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b = blocks[idx]
            ids_b = get_block_triplets(b)
            
            unique_tids = set()
            for tid in ids_b:
                if count[tid] == 1:
                    unique_tids.add(tid)
            
            if not unique_tids:
                continue
            
            # Build a new block greedily to cover unique_tids
            new_block = []
            new_block_set = set()
            
            # Start with points from unique triplets
            for tid in list(unique_tids)[:t]:
                for p in all_triplets[tid]:
                    if p not in new_block_set:
                        new_block.append(p)
                        new_block_set.add(p)
                        if len(new_block) >= k:
                            break
                if len(new_block) >= k:
                    break
            
            if len(new_block) < t:
                continue
            
            while len(new_block) < k:
                best_p = -1
                best_gain = -1
                for p in range(v):
                    if p in new_block_set:
                        continue
                    gain = 0
                    for triplet in itertools.combinations(new_block, t - 1):
                        tt = tuple(sorted(triplet + (p,)))
                        tid = get_id(tt)
                        if tid in unique_tids:
                            gain += 10
                        elif tid in ids_b:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1:
                    break
                new_block.append(best_p)
                new_block_set.add(best_p)
            
            new_block.sort()
            
            if len(new_block) < k:
                continue
            
            new_ids = get_block_triplets(new_block)
            
            # Check: new block must cover all unique triplets
            ok = True
            for tid in unique_tids:
                if tid not in new_ids:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check: no triplet loses coverage
            for tid in ids_b:
                c = count[tid]
                in_new = 1 if tid in new_ids else 0
                if c - 1 + in_new < 1:
                    ok = False
                    break
            if not ok:
                continue
            
            # Only accept if new block is different
            if new_block == b:
                continue
            
            update_counts([b], -1)
            update_counts([new_block], 1)
            new_blocks = list(blocks)
            new_blocks[idx] = new_block
            return new_blocks, True
        
        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True
        
        if not progress_made and time.time() < deadline - 10:
            current_blocks, swapped = try_replace_block(current_blocks)
            if swapped:
                progress_made = True

    # Final removal pass
    current_blocks, _ = try_remove_redundant(current_blocks)
    
    # Phase 3: Continuous Relaxation with Smooth Coverage Penalty
    if time.time() < deadline - 5:
        n_blocks = len(current_blocks)
        
        A = np.zeros((n_triplets, n_blocks), dtype=np.float64)
        for i, block in enumerate(current_blocks):
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                A[tid, i] = 1.0
        
        x = np.ones(n_blocks, dtype=np.float64)
        
        lambda_cov = 50.0
        lambda_l1 = 0.05
        lr = 0.1
        n_iters = 150
        
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            cov = A @ x
            
            # Smooth penalty: use soft-min approximation
            # max(0, 1-cov) approximated smoothly
            deficit = np.maximum(0.0, 1.0 - cov)
            # Smooth version: use exp-based soft max
            temp = 0.1
            log_sum_exp = np.log(np.sum(np.exp(deficit / temp), axis=0)) * temp
            # Gradient of soft penalty
            grad_cov = -2.0 * lambda_cov * (A.T @ (deficit / (deficit + 1e-6)))
            
            grad_sum = np.ones(n_blocks)
            grad_l1 = lambda_l1 * np.sign(x)
            
            grad = grad_sum + grad_cov + grad_l1
            x = x - lr * grad
            x = np.clip(x, 0.0, 1.0)
        
        block_weights = x.copy()
        
        if time.time() < deadline - 3:
            progress = True
            while progress and time.time() < deadline - 3:
                progress = False
                current_blocks, swapped = try_swap_2_for_1(current_blocks, block_weights)
                if swapped:
                    progress = True
                    n_blocks_new = len(current_blocks)
                    if n_blocks_new != n_blocks:
                        A = np.zeros((n_triplets, n_blocks_new), dtype=np.float64)
                        for i, block in enumerate(current_blocks):
                            for triplet in itertools.combinations(block, t):
                                tid = get_id(triplet)
                                A[tid, i] = 1.0
                        x = np.ones(n_blocks_new, dtype=np.float64)
                        for it in range(30):
                            if time.time() > deadline - 3:
                                break
                            cov = A @ x
                            deficit = np.maximum(0.0, 1.0 - cov)
                            grad_cov = -2.0 * lambda_cov * (A.T @ (deficit / (deficit + 1e-6)))
                            grad_sum = np.ones(n_blocks_new)
                            grad_l1 = lambda_l1 * np.sign(x)
                            grad = grad_sum + grad_cov + grad_l1
                            x = x - lr * grad
                            x = np.clip(x, 0.0, 1.0)
                        block_weights = x.copy()
                    else:
                        block_weights = block_weights[:n_blocks_new]
        
        current_blocks, _ = try_remove_redundant(current_blocks)
    
    return current_blocks

if __name__ == "__main__":
    pass
