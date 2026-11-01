import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    full_mask = (1 << n_tsubs) - 1

    def greedy_cover(uncovered_mask, remaining, used_set, deadline, dual_weights=None):
        """Run greedy selection until coverage is complete or deadline is reached.
        
        If dual_weights is provided, use Lagrangian-guided scoring:
        score(j) = sum over uncovered tsubs in block j of (1 + dual_weight[tsub])
        This encourages covering rare/difficult t-subsets first.
        """
        blocks_idx = []
        while remaining > 0:
            if time.time() > deadline:
                return None, None  # Timeout, return None to indicate failure
            
            best_j = -1
            best_score = 0

            # Scan all candidates to find the one with max score
            for j in range(n_ksubs):
                if j in used_set:
                    continue
                
                # Calculate score: weighted gain
                common = uncovered_mask & ksub_masks[j]
                if dual_weights is not None:
                    # Score = sum of (1 + dual_weight[i]) for each uncovered tsub i in this block
                    # We can compute this as: gain + sum of dual_weights for uncovered tsubs in block
                    gain = common.bit_count()
                    # Sum dual weights for uncovered tsubs in this block
                    # dual_weights is a list/array of length n_tsubs
                    # We need sum of dual_weights[i] for i where common has bit i set
                    # Efficient way: iterate over set bits of common
                    dw_sum = 0.0
                    c = common
                    while c:
                        # Get lowest set bit index
                        lsb = c & (-c)
                        idx = lsb.bit_length() - 1
                        dw_sum += dual_weights[idx]
                        c ^= lsb
                    score = gain + dw_sum
                else:
                    score = common.bit_count()
                
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j == -1 or best_score == 0:
                # No more progress possible
                return blocks_idx, uncovered_mask

            # Add best block
            blocks_idx.append(best_j)
            used_set.add(best_j)
            
            # Update uncovered mask
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= (uncovered_mask ^ (uncovered_mask | ksub_masks[best_j])).bit_count()
            # Actually, let's just recalculate remaining properly
            # remaining = number of uncovered tsubs
            # Simpler: count bits in uncovered_mask
            remaining = uncovered_mask.bit_count()

        return blocks_idx, uncovered_mask

    def update_dual_weights(uncovered_mask, dual_weights, step_size, deadline):
        """Update dual weights via subgradient descent.
        
        For each uncovered t-subset, increase its weight (it's hard to cover).
        For each covered t-subset, decrease its weight slightly.
        """
        if time.time() > deadline:
            return dual_weights
        
        for i in range(n_tsubs):
            if (uncovered_mask >> i) & 1:
                # Uncovered: increase weight
                dual_weights[i] += step_size
            else:
                # Covered: decrease weight slightly
                dual_weights[i] = max(0.0, dual_weights[i] - step_size * 0.1)
        
        return dual_weights

    def remove_redundant(blocks_idx, deadline):
        """Remove redundant blocks from the selected set."""
        if not blocks_idx:
            return blocks_idx
            
        active_indices = list(blocks_idx)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active_indices):
                # Calculate union of all OTHER active blocks
                other_mask = 0
                for j_idx, bj in enumerate(active_indices):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                
                # Check if bi is redundant
                if (ksub_masks[bi] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(bi)
            
            active_indices = new_active
            
            if not changed:
                break

        return active_indices

    def local_search(blocks_idx, deadline):
        """Try to improve the covering by swapping or removing blocks.
        
        Returns improved list of block indices.
        """
        if not blocks_idx:
            return blocks_idx
        
        active = list(blocks_idx)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each block
            for i in range(len(active)):
                if time.time() > deadline:
                    break
                # Check if active[i] is redundant
                other_mask = 0
                for j in range(len(active)):
                    if j != i:
                        other_mask |= ksub_masks[active[j]]
                if (ksub_masks[active[i]] & ~other_mask) == 0:
                    active.pop(i)
                    improved = True
                    break  # Restart scan after removal
            
            # Try swapping: replace one block with another that covers more uncovered
            if not improved:
                # Find current uncovered
                current_cover = 0
                for bi in active:
                    current_cover |= ksub_masks[bi]
                uncovered = full_mask & ~current_cover
                if uncovered == 0:
                    break  # Full coverage, no need to swap
                
                # Try adding a block that covers uncovered, then removing a redundant one
                best_gain = 0
                best_add_j = -1
                best_remove_i = -1
                
                for add_j in range(n_ksubs):
                    if add_j in active:
                        continue
                    gain = (uncovered & ksub_masks[add_j]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add_j = add_j
                
                if best_add_j != -1 and best_gain > 0:
                    # Try to find a block to remove that becomes redundant after adding best_add_j
                    new_active = active + [best_add_j]
                    for rem_i in range(len(new_active)):
                        other_mask = 0
                        for rem_j in range(len(new_active)):
                            if rem_j != rem_i:
                                other_mask |= ksub_masks[new_active[rem_j]]
                        if (ksub_masks[new_active[rem_i]] & ~other_mask) == 0:
                            # Can remove rem_i
                            active = new_active
                            active.pop(rem_i)
                            improved = True
                            break
                    if improved:
                        break
        
        return active

    best_blocks_idx = None
    best_len = float('inf')

    # Phase 1: Pure Greedy (baseline)
    used_set = set()
    res_idx, _ = greedy_cover(full_mask, n_tsubs, used_set, deadline - 20.0, dual_weights=None)
    if res_idx is not None:
        cleaned_idx = remove_redundant(res_idx, deadline - 15.0)
        # Verify validity
        final_mask = 0
        for idx in cleaned_idx:
            final_mask |= ksub_masks[idx]
        if final_mask == full_mask:
            best_blocks_idx = cleaned_idx
            best_len = len(cleaned_idx)
            # Try local search on this solution
            if time.time() < deadline - 10.0:
                improved = local_search(cleaned_idx, deadline - 10.0)
                final_mask2 = 0
                for idx in improved:
                    final_mask2 |= ksub_masks[idx]
                if final_mask2 == full_mask and len(improved) < best_len:
                    best_blocks_idx = improved
                    best_len = len(improved)

    # Phase 2: Lagrangian-guided greedy with dual weight updates
    if time.time() < deadline - 15.0:
        # Initialize dual weights
        dual_weights = [0.0] * n_tsubs
        step_size = 1.0
        
        # Multiple restarts with different initial dual weights or step sizes
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Reset dual weights with some randomization for diversity
            if r == 0:
                dual_weights = [0.0] * n_tsubs
                step_size = 1.0
            elif r == 1:
                dual_weights = [random.uniform(0.0, 0.5) for _ in range(n_tsubs)]
                step_size = 0.8
            else:
                dual_weights = [random.uniform(0.0, 1.0) for _ in range(n_tsubs)]
                step_size = 1.2
            
            # Run Lagrangian-guided greedy
            used_set = set()
            uncovered_mask = full_mask
            remaining = n_tsubs
            blocks_idx = []
            
            # Adaptive step size: decrease over iterations
            base_step = step_size
            
            while remaining > 0:
                if time.time() > deadline - 10.0:
                    break
                
                # Run greedy step with current dual weights
                step_result, new_uncovered = greedy_cover(uncovered_mask, remaining, used_set, deadline - 10.0, dual_weights=dual_weights)
                if step_result is None:
                    break
                
                if not step_result:
                    break
                
                # We only added one block per call to greedy_cover in this design
                # But greedy_cover runs until complete. Let's restructure.
                # Actually, let's just call it once and let it run to completion
                # or timeout.
                break  # This design is flawed; let's fix below
            
            # Better approach: single greedy run with dual weights, then update
            used_set = set()
            res_idx, final_uncovered = greedy_cover(full_mask, n_tsubs, used_set, deadline - 10.0, dual_weights=dual_weights)
            if res_idx is not None:
                cleaned_idx = remove_redundant(res_idx, deadline - 10.0)
                final_mask = 0
                for idx in cleaned_idx:
                    final_mask |= ksub_masks[idx]
                if final_mask == full_mask:
                    if len(cleaned_idx) < best_len:
                        best_blocks_idx = cleaned_idx
                        best_len = len(cleaned_idx)
                    # Try local search
                    if time.time() < deadline - 10.0:
                        improved = local_search(cleaned_idx, deadline - 10.0)
                        final_mask2 = 0
                        for idx in improved:
                            final_mask2 |= ksub_masks[idx]
                        if final_mask2 == full_mask and len(improved) < best_len:
                            best_blocks_idx = improved
                            best_len = len(improved)
                
                # Update dual weights based on what's still uncovered
                if final_uncovered != 0:
                    dual_weights = update_dual_weights(final_uncovered, dual_weights, base_step, deadline - 10.0)

    # Phase 3: Randomized greedy restarts (keep from original for diversity)
    time_remaining = deadline - time.time()
    if time_remaining > 10.0 and best_blocks_idx is not None:
        max_restarts = 3
        for r in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Randomized Greedy
            uncovered_mask = full_mask
            remaining = n_tsubs
            used_set = set()
            blocks_idx = []
            
            while remaining > 0:
                if time.time() > deadline - 10.0:
                    break
                
                # Find top-k candidates
                top_k = 5
                gains = []
                for j in range(n_ksubs):
                    if j in used_set:
                        continue
                    common = uncovered_mask & ksub_masks[j]
                    gain = common.bit_count()
                    if gain > 0:
                        gains.append((gain, j))
                
                if not gains:
                    break
                    
                # Select from top-k
                gains.sort(reverse=True)
                top_indices = [g[1] for g in gains[:top_k]]
                
                # Weighted random selection
                weights = [g[0] for g in gains[:top_k]]
                total_w = sum(weights)
                rand_val = random.uniform(0, total_w)
                current_sum = 0
                selected_j = top_indices[0]
                for w, j in zip(weights, top_indices):
                    current_sum += w
                    if rand_val <= current_sum:
                        selected_j = j
                        break
                
                blocks_idx.append(selected_j)
                used_set.add(selected_j)
                uncovered_mask &= ~ksub_masks[selected_j]
                remaining = uncovered_mask.bit_count()
                
            # Verify and clean
            if len(blocks_idx) < best_len:
                cleaned_idx = remove_redundant(blocks_idx, deadline - 10.0)
                final_mask = 0
                for idx in cleaned_idx:
                    final_mask |= ksub_masks[idx]
                if final_mask == full_mask and len(cleaned_idx) < best_len:
                    best_blocks_idx = cleaned_idx
                    best_len = len(cleaned_idx)

    if best_blocks_idx is None:
        # Fallback: should not happen for v=19, k=7, t=4
        # Return a valid covering using pure greedy
        used_set = set()
        res_idx, _ = greedy_cover(full_mask, n_tsubs, used_set, deadline - 5.0, dual_weights=None)
        if res_idx is not None:
            best_blocks_idx = res_idx
        else:
            # Last resort: return empty (invalid, but better than exception)
            best_blocks_idx = []

    return [list(all_ksubs[idx]) for idx in best_blocks_idx]
