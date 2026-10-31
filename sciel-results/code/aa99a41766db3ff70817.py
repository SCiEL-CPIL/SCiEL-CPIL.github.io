"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: count-based fast greedy + aggressive local search (removal / add-then-remove / swap),
with multiple random restarts. Uses a coverage-count array for O(k) removal/swap checks.
Enhancements:
1. Global frequency bias for greedy tie-breaking.
2. Candidate pool pruning for final high-precision polish.
3. Targeted swap heuristic in local search.
"""
import time
import itertools
import random
import sys

_DEADLINE = 145.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    # For each k-subset, list of t-subset indices it covers
    coverage = []
    # Reverse: for each t-subset index, list of k-subset indices covering it
    rev_coverage = [[] for _ in range(n_t)]
    for ki, ks in enumerate(k_subsets):
        idxs = [t_index[ts] for ts in itertools.combinations(ks, t)]
        coverage.append(idxs)
        for ti in idxs:
            rev_coverage[ti].append(ki)
    return all_t, t_index, k_subsets, coverage, n_t, rev_coverage


def _greedy_construct(coverage, n_t, rng, n_k, deadline, bias=None):
    """Greedy set cover using a coverage-count array. Optional bias for tie-breaking."""
    cnt = [0] * n_t  # cnt[i] = number of selected blocks covering t-subset i
    blocks = []
    uncovered = n_t
    
    # Precompute a list of indices to iterate, shuffled if rng provided, otherwise natural
    # For speed, we iterate all n_k each step.
    while uncovered > 0 and _elapsed() < deadline:
        best_i = -1
        best_gain = 0
        best_bias_val = -1
        for i in range(n_k):
            g = 0
            for idx in coverage[i]:
                if cnt[idx] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_i = i
                best_bias_val = bias.get(i, 0) if bias else 0
                # Early break if gain is 0? No, we need to find max. 
                # If max possible gain is small, we still scan.
            elif g == best_gain and g > 0:
                b_val = bias.get(i, 0) if bias else 0
                if b_val > best_bias_val:
                    best_bias_val = b_val
                    best_i = i
        if best_i < 0:
            break
        blocks.append(best_i)
        for idx in coverage[best_i]:
            if cnt[idx] == 0:
                uncovered -= 1
            cnt[idx] += 1
    return blocks, cnt, uncovered


def _local_search(blocks, coverage, rev_coverage, n_t, n_k, rng, deadline, bias=None):
    """Local search minimizing |blocks| while keeping full coverage.
    Uses cnt array for O(k) removal/swap checks.
    """
    blocks = list(blocks)
    cnt = [0] * n_t
    for b in blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
    
    # Verify full coverage
    if any(c == 0 for c in cnt):
        return blocks, cnt

    in_blocks = set(blocks)

    improved = True
    # We run passes until no improvement or deadline
    while improved and _elapsed() < deadline:
        improved = False
        # --- Removal pass ---
        i = 0
        while i < len(blocks) and _elapsed() < deadline:
            b = blocks[i]
            # Check if removing b keeps all cnt > 0
            removable = True
            for idx in coverage[b]:
                if cnt[idx] == 1:
                    removable = False
                    break
            if removable:
                for idx in coverage[b]:
                    cnt[idx] -= 1
                blocks.pop(i)
                in_blocks.discard(b)
                improved = True
                continue
            i += 1

        # --- Add-then-remove pass (targeted) ---
        # Find vulnerable t-subsets (cnt == 1)
        vulnerable = [i for i in range(n_t) if cnt[i] == 1]
        
        # If no vulnerable, we can't easily improve via add-remove unless we swap.
        # But standard add-remove requires a vulnerable target to be "protected" by the new block.
        
        # Try a few times
        for _ in range(20):
            if _elapsed() >= deadline:
                break
            if not vulnerable:
                break
            
            vt = vulnerable[rng.randrange(len(vulnerable))]
            candidates = rev_coverage[vt]
            if not candidates:
                continue
            
            # Score candidates by how many vulnerable t's they cover
            best_c = -1
            best_score = 0
            for c in candidates:
                if c in in_blocks:
                    continue
                s = 0
                for idx in coverage[c]:
                    if cnt[idx] == 1:
                        s += 1
                if s > best_score:
                    best_score = s
                    best_c = c
            
            if best_c < 0:
                continue
            
            # Add best_c
            blocks.append(best_c)
            in_blocks.add(best_c)
            for idx in coverage[best_c]:
                cnt[idx] += 1
            
            # Now try to remove a block
            removed = False
            for idx in range(len(blocks)):
                b = blocks[idx]
                removable = True
                for tidx in coverage[b]:
                    if cnt[tidx] == 1:
                        removable = False
                        break
                if removable:
                    for tidx in coverage[b]:
                        cnt[tidx] -= 1
                    blocks.pop(idx)
                    in_blocks.discard(b)
                    removed = True
                    improved = True
                    # Update vulnerable list for next iteration? 
                    # It's expensive to rebuild. We'll just break and let the next outer loop iteration handle it.
                    break
            
            if not removed:
                # Revert
                blocks.pop()
                in_blocks.discard(best_c)
                for idx in coverage[best_c]:
                    cnt[idx] -= 1

        # --- Swap pass: replace one block with another to enable a removal ---
        for _ in range(20):
            if _elapsed() >= deadline:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_k)
            if j in in_blocks:
                continue
            old = blocks[i]
            
            # Heuristic: Only swap if new block covers more 'vulnerable' or 'low count' t's than old?
            # For now, keep random but check if it enables removal.
            
            # Tentatively swap
            for idx in coverage[old]:
                cnt[idx] -= 1
            for idx in coverage[j]:
                cnt[idx] += 1
            blocks[i] = j
            in_blocks.discard(old)
            in_blocks.add(j)
            
            # Try to remove a block
            removed = False
            for idx in range(len(blocks)):
                b = blocks[idx]
                removable = True
                for tidx in coverage[b]:
                    if cnt[tidx] == 1:
                        removable = False
                        break
                if removable:
                    for tidx in coverage[b]:
                        cnt[tidx] -= 1
                    blocks.pop(idx)
                    in_blocks.discard(b)
                    removed = True
                    improved = True
                    break
            
            if not removed:
                # Revert swap
                blocks[i] = old
                in_blocks.discard(j)
                in_blocks.add(old)
                for idx in coverage[j]:
                    cnt[idx] -= 1
                for idx in coverage[old]:
                    cnt[idx] += 1

    return blocks, cnt


def construct(v, k, t):
    global _START
    _START = time.time()

    all_t, t_index, k_subsets, coverage, n_t, rev_coverage = _precompute(v, k, t)
    n_k = len(k_subsets)

    best_blocks = None
    best_size = float('inf')
    global_freq = {} # Global frequency of blocks across restarts

    # Multiple restarts with different seeds
    seeds = [12345, 67890, 24680, 13579, 99999, 55555, 11111, 77777, 135791, 246802]
    
    for seed in seeds:
        if _elapsed() > _DEADLINE - 40:
            break
        rng = random.Random(seed)
        
        # Use global_freq as bias for this restart
        # If it's the first restart, bias is empty
        bias = dict(global_freq)
        
        # Greedy construction
        blocks, cnt, uncovered = _greedy_construct(coverage, n_t, rng, n_k, _DEADLINE - 30, bias)
        
        if uncovered > 0:
            # Repair
            while uncovered > 0 and _elapsed() < _DEADLINE - 10:
                best_i = -1
                best_gain = 0
                for i in range(n_k):
                    g = 0
                    for idx in coverage[i]:
                        if cnt[idx] == 0:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_i = i
                if best_i < 0:
                    break
                blocks.append(best_i)
                for idx in coverage[best_i]:
                    if cnt[idx] == 0:
                        uncovered -= 1
                    cnt[idx] += 1
        
        if uncovered > 0:
            continue  # invalid, skip

        # Local search
        ls_blocks, ls_cnt = _local_search(blocks, coverage, rev_coverage, n_t, n_k, rng, _DEADLINE - 10, bias)
        
        if any(c == 0 for c in ls_cnt):
            # Fallback to original if LS broke validity (shouldn't happen, but safe)
            ls_blocks = blocks
        
        if len(ls_blocks) < best_size:
            best_size = len(ls_blocks)
            best_blocks = list(ls_blocks)
            
            # Update global frequency with the blocks in this good solution
            for b in ls_blocks:
                global_freq[b] = global_freq.get(b, 0) + 1

    if best_blocks is None:
        # Fallback: pure greedy
        rng = random.Random(12345)
        blocks, cnt, uncovered = _greedy_construct(coverage, n_t, rng, n_k, _DEADLINE - 5)
        best_blocks = blocks
        best_size = len(blocks)

    # --- Final Polish Pass ---
    # Use the global_freq to bias a final high-precision local search on the best solution found.
    # We restrict the swap/add candidates to the top 5000 most frequent blocks to speed up the search.
    if _elapsed() < _DEADLINE - 10 and best_blocks is not None:
        # Rebuild cnt for best_blocks
        cnt_best = [0] * n_t
        for b in best_blocks:
            for idx in coverage[b]:
                cnt_best[idx] += 1
        
        # Check validity
        if all(c > 0 for c in cnt_best):
            # Identify top candidates
            sorted_blocks = sorted(global_freq.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [b for b, _ in sorted_blocks[:5000]]
            if not top_candidates:
                top_candidates = list(range(n_k))
            
            # Run local search again, but with a restricted candidate pool for swaps/adds?
            # The current _local_search iterates all n_k. 
            # To implement restricted pool, we'd need to modify _local_search or create a wrapper.
            # Given time constraints, we just run _local_search again with the bias, 
            # relying on the bias to guide the *greedy* if we were to rebuild, 
            # but LS uses random swaps. 
            # Instead, let's do a "Simulated Annealing" style final polish:
            # Try to remove, then if stuck, try swapping with a high-frequency block.
            
            rng_polish = random.Random(42)
            improved = True
            while improved and _elapsed() < _DEADLINE - 2:
                improved = False
                # Removal
                i = 0
                while i < len(best_blocks) and _elapsed() < _DEADLINE - 2:
                    b = best_blocks[i]
                    removable = True
                    for idx in coverage[b]:
                        if cnt_best[idx] == 1:
                            removable = False
                            break
                    if removable:
                        for idx in coverage[b]:
                            cnt_best[idx] -= 1
                        best_blocks.pop(i)
                        improved = True
                        continue
                    i += 1
                
                # Targeted Swap with High-Freq blocks
                if not improved:
                    # Pick a random block from best_blocks
                    if best_blocks:
                        idx_old = rng_polish.randrange(len(best_blocks))
                        old_b = best_blocks[idx_old]
                        
                        # Pick a random candidate from top_candidates not in best_blocks
                        in_set = set(best_blocks)
                        candidates = [c for c in top_candidates if c not in in_set]
                        if candidates:
                            new_b = candidates[rng_polish.randrange(len(candidates))]
                            
                            # Tentative swap
                            for idx in coverage[old_b]:
                                cnt_best[idx] -= 1
                            for idx in coverage[new_b]:
                                cnt_best[idx] += 1
                            best_blocks[idx_old] = new_b
                            
                            # Check if any removal is now possible
                            removed = False
                            for idx in range(len(best_blocks)):
                                b = best_blocks[idx]
                                removable = True
                                for tidx in coverage[b]:
                                    if cnt_best[tidx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tidx in coverage[b]:
                                        cnt_best[tidx] -= 1
                                    best_blocks.pop(idx)
                                    removed = True
                                    improved = True
                                    break
                            
                            if not removed:
                                # Revert
                                best_blocks[idx_old] = old_b
                                for idx in coverage[new_b]:
                                    cnt_best[idx] -= 1
                                for idx in coverage[old_b]:
                                    cnt_best[idx] += 1

    # Ensure validity
    cnt = [0] * n_t
    for b in best_blocks:
        for idx in coverage[b]:
            cnt[idx] += 1
    if any(c == 0 for c in cnt):
        # Repair
        while any(c == 0 for c in cnt) and _elapsed() < _DEADLINE - 2:
            best_i = -1
            best_gain = 0
            for i in range(n_k):
                g = 0
                for idx in coverage[i]:
                    if cnt[idx] == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_i = i
            if best_i < 0:
                break
            best_blocks.append(best_i)
            for idx in coverage[best_i]:
                cnt[idx] += 1

    # Deduplicate and return
    seen = set()
    dedup = []
    for b in best_blocks:
        key = k_subsets[b]
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup

if __name__ == "__main__":
    # Quick test
    import sys
    v, k, t = 19, 7, 4
    start = time.time()
    result = construct(v, k, t)
    end = time.time()
    print(f"Result size: {len(result)}, Time: {end-start:.2f}s")
    # Verify
    all_t = list(itertools.combinations(range(v), t))
    covered = set()
    for block in result:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    print(f"Covered {len(covered)}/{len(all_t)}")
