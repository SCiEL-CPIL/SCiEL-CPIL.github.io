"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

Structural redesign: separate block *generation* (fast bitset greedy) from block
*selection* (greedy set cover + block-exchange local search). A pre-generated
candidate pool lets local search trade blocks, something pure greedy cannot do.
"""
import time
import random
import itertools

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# Precompute all t-subsets and each block's coverage as an integer bitmask.
_T_SUBSETS = []
_T_INDEX = {}
_BLOCK_COV = {}
_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    t_subs = [tuple(c) for c in itertools.combinations(range(v), t)]
    index = {sub: i for i, sub in enumerate(t_subs)}
    n = len(t_subs)
    cov = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= (1 << index[sub])
        cov[blk] = m
    _CACHE[key] = (t_subs, index, n, cov)
    return _CACHE[key]


def _gen_block(uncovered, v, k, t, index, rng, t_subs):
    """Build one block that covers many currently-uncovered t-subsets."""
    # Find an anchor: the first t-subset (by index) that is uncovered
    anchor = None
    # To speed this up, we can just iterate through t_subs and check the bit
    for sub in t_subs:
        idx = index[sub]
        if (uncovered >> idx) & 1:
            anchor = sub
            break
    
    if anchor is None:
        return None
        
    block = set(anchor)
    # greedily add the point covering the most uncovered t-subsets
    while len(block) < k:
        best_x = None
        best_gain = -1
        sorted_block = sorted(block)
        for x in range(v):
            if x in block:
                continue
            gain = 0
            # Check all t-1 subsets of the current block to see if adding x creates a covered t-subset
            for rest in itertools.combinations(sorted_block, t - 1):
                # Create the new t-subset
                new_sub = tuple(sorted(list(rest) + [x]))
                idx = index[new_sub]
                if (uncovered >> idx) & 1:
                    gain += 1
            # small random tie-break for diversity
            gain += rng.random() * 1e-9
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None or best_gain <= 0:
            # fill remaining with arbitrary points
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.add(x)
            break
        block.add(best_x)
    return tuple(sorted(block))


def _greedy_select(cov_list, n, deadline, rng):
    """Greedy set cover over candidate blocks. Returns (selected_idx, uncovered)."""
    uncovered = (1 << n) - 1
    remaining = list(range(len(cov_list)))
    selected = []
    # Optimization: Keep a list of active candidates. 
    # For large n, bit_count is fast, but scanning all remaining every time is O(N^2).
    # Given N ~ 1500-2000, this is acceptable.
    while uncovered and remaining:
        if time.time() > deadline:
            break
        best_i = -1
        best_gain = 0
        # We can optimize by only checking candidates that actually cover something new
        for i in remaining:
            g = (cov_list[i] & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        uncovered &= ~cov_list[best_i]
        # Remove best_i from remaining
        # Using a set for remaining would be faster for removal, but we need to iterate.
        # List removal is O(N). 
        remaining.remove(best_i)
    return selected, uncovered


def _local_search(selected, cov_list, n, deadline, rng, max_rounds=4000):
    """Block-exchange local search: drop a redundant block, or swap one in for one out."""
    nblocks = len(cov_list)
    sel = list(selected)
    sel_set = set(sel)
    # coverage count of each t-subset by selected blocks
    count = [0] * n
    for i in sel:
        m = cov_list[i]
        while m:
            lb = m & (-m)
            count[lb.bit_length() - 1] += 1
            m &= m - 1

    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # 1) try dropping each selected block
        drop_idx = -1
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            # check no t-subset becomes uncovered
            ok = True
            while m:
                lb = m & (-m)
                if count[lb.bit_length() - 1] == 1:
                    ok = False
                    break
                m &= m - 1
            if ok:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            m = cov_list[i]
            while m:
                lb = m & (-m)
                count[lb.bit_length() - 1] -= 1
                m &= m - 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            continue
            
        # 2) try swapping: find a block j not in sel that covers the critical subsets of some i in sel
        # and allows dropping i (or another block). 
        # Simplified: For each i in sel, find critical subsets. If there exists j not in sel 
        # that covers all critical subsets of i, swap i->j. Then retry dropping.
        # To save time, we only do this if we haven't found a drop.
        swap_done = False
        for pos in range(len(sel)):
            i = sel[pos]
            m = cov_list[i]
            crit_mask = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if count[idx] == 1:
                    crit_mask |= lb
                m &= m - 1
            
            if crit_mask == 0:
                # This block is redundant, but we should have caught it in step 1?
                # No, step 1 checks if *any* subset has count 1. If crit_mask is 0, 
                # it means no subset has count 1 *for this block*, so it's safe to drop?
                # Wait, if count[idx] > 1 for all idx in block i, then dropping i leaves them covered.
                # So if crit_mask == 0, it is droppable. Why didn't step 1 catch it?
                # Step 1 checks `if count[...] == 1: ok=False`. If no 1s, ok=True.
                # So this case is handled by step 1.
                continue

            # Find j not in sel that covers crit_mask
            for j in range(nblocks):
                if j in sel_set:
                    continue
                mj = cov_list[j]
                if (mj & crit_mask) == crit_mask:
                    # Swap i with j
                    # Remove i
                    mi = cov_list[i]
                    while mi:
                        lb = mi & (-mi)
                        count[lb.bit_length() - 1] -= 1
                        mi &= mi - 1
                    # Add j
                    while mj:
                        lb = mj & (-mj)
                        count[lb.bit_length() - 1] += 1
                        mj &= mj - 1
                    sel[pos] = j
                    sel_set.remove(i)
                    sel_set.add(j)
                    improved = True
                    swap_done = True
                    break
            if swap_done:
                break
    return sel


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    t_subs, index, n, cov = _prepare(v, k, t)

    # ---- Phase 1: generate a diverse candidate pool of good blocks ----
    candidates = []
    cand_set = set()
    full = (1 << n) - 1
    uncovered = full
    gen_limit = 2000 # Reduced to allow more time for selection/search
    gens = 0
    
    # Time budget for generation: 40% of total
    gen_deadline = t0 + (deadline - t0) * 0.4
    
    while uncovered and gens < gen_limit and time.time() < gen_deadline:
        b = _gen_block(uncovered, v, k, t, index, rng, t_subs)
        if b is None:
            break
        if b not in cand_set:
            cand_set.add(b)
            candidates.append(b)
        uncovered &= ~cov[b]
        gens += 1
        
        # Add random blocks for diversity
        if gens % 50 == 0 and time.time() < gen_deadline:
            for _ in range(5):
                rb = tuple(sorted(rng.sample(range(v), k)))
                if rb not in cand_set:
                    cand_set.add(rb)
                    candidates.append(rb)

    # If still uncovered, keep generating until deadline - 10s
    while uncovered and time.time() < deadline - 10:
        b = _gen_block(uncovered, v, k, t, index, rng, t_subs)
        if b is None:
            break
        if b not in cand_set:
            cand_set.add(b)
            candidates.append(b)
        uncovered &= ~cov[b]

    cov_list = [cov[b] for b in candidates]

    # ---- Phase 2: greedy selection ----
    # Use 50% of remaining time for greedy
    sel_deadline = time.time() + (deadline - time.time()) * 0.5
    selected, unc = _greedy_select(cov_list, n, sel_deadline, rng)

    # ---- Phase 3: local search to reduce block count ----
    # Use remaining time for local search
    ls_deadline = deadline - 2.0
    selected = _local_search(selected, cov_list, n, ls_deadline, rng)

    # ---- Fallback: ensure completeness ----
    # If any t-subset is still uncovered, we MUST add blocks to cover them.
    # This ensures validity.
    if time.time() < deadline - 1:
        # Calculate current coverage
        covered_mask = 0
        for i in selected:
            covered_mask |= cov_list[i]
        
        uncovered_now = full & ~covered_mask
        
        if uncovered_now:
            # Greedily add blocks to cover the rest
            # We can use the candidates list, or even generate new ones if needed.
            # Candidates should be sufficient if generation was thorough.
            remaining_candidates = [j for j in range(len(cov_list)) if j not in selected]
            remaining_candidates.sort(key=lambda j: (cov_list[j] & uncovered_now).bit_count(), reverse=True)
            
            for j in remaining_candidates:
                if not uncovered_now:
                    break
                if cov_list[j] & uncovered_now:
                    if j not in selected:
                        selected.append(j)
                    uncovered_now &= ~cov_list[j]
            
            # If still uncovered (should be rare), generate new blocks
            while uncovered_now and time.time() < deadline - 0.5:
                b = _gen_block(uncovered_now, v, k, t, index, rng, t_subs)
                if b is None:
                    break
                if b not in cand_set:
                    cand_set.add(b)
                    candidates.append(b)
                    cov_list.append(cov[b])
                    j = len(cov_list) - 1
                else:
                    # Find index of b
                    # This is inefficient, but this path is rare.
                    # Better to just check all cov_list
                    j = -1
                    for idx, c in enumerate(cov_list):
                        if c == cov[b]:
                            j = idx
                            break
                    if j == -1:
                         continue
                
                if j not in selected:
                    selected.append(j)
                uncovered_now &= ~cov_list[j]

    result = [list(candidates[i]) for i in selected]
    return result
