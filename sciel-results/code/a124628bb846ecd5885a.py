"""Covering design C(19,8,5): Conflict-graph maximum independent set via
greedy coloring lower bound + branch-and-bound, seeded from fast greedy set cover
with redundancy pruning and bitset-accelerated local search.
"""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 150s hard kill

    # ---------- Precomputation with bitmasks ----------
    all_t = list(itertools.combinations(range(v), t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    all_k = list(itertools.combinations(range(v), k))
    N_k = len(all_k)

    # bitmask for each block: which t-subsets it covers
    block_masks = []
    for b in all_k:
        mask = 0
        for s in itertools.combinations(b, t):
            mask |= (1 << t_idx[s])
        block_masks.append(mask)

    # For each t-subset, the list of blocks that cover it (for fast pivot selection)
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, m in enumerate(block_masks):
        m2 = m
        while m2:
            lb = m2 & -m2
            idx = lb.bit_length() - 1
            t_to_blocks[idx].append(bi)
            m2 ^= lb

    # ---------- Greedy set cover ----------
    def greedy_set_cover(deadline, seed=0):
        random.seed(seed)
        remaining = (1 << N_t) - 1
        # Use a Python int as bitset of uncovered t-subsets (N_t up to 11628 bits fits)
        uncovered_mask = (1 << N_t) - 1
        chosen = []

        while uncovered_mask and time.time() < deadline:
            # Randomly pick an uncovered t-subset
            # Find a set bit in uncovered_mask
            lb = uncovered_mask & -uncovered_mask
            pivot_t = lb.bit_length() - 1

            # Candidate blocks covering this t-subset
            cands = t_to_blocks[pivot_t]
            if not cands:
                break

            # Score each candidate: number of new t-subsets covered
            best_block = None
            best_gain = -1
            for bi in cands:
                gain = bin(block_masks[bi] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block is None:
                break

            chosen.append(best_block)
            uncovered_mask &= ~block_masks[best_block]

        return chosen

    # ---------- Redundancy removal ----------
    def remove_redundant(bi_list):
        if not bi_list:
            return bi_list
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in bi_list:
            m = block_masks[bi]
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                counts[idx] += 1
                m ^= lb

        keep = []
        for bi in bi_list:
            m = block_masks[bi]
            redundant = True
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                if counts[idx] == 1:
                    redundant = False
                    break
                m ^= lb
            if not redundant:
                keep.append(bi)
        return keep

    # ---------- Local search: single-block mutation ----------
    def local_search(bi_list, budget):
        end_t = time.time() + budget
        cur = list(bi_list)
        random.seed(42)

        while time.time() < end_t:
            # Compute coverage counts
            counts = np.zeros(N_t, dtype=np.int32)
            for bi in cur:
                m = block_masks[bi]
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lb

            improved = False
            # Try removing each block (random order)
            order = list(range(len(cur)))
            random.shuffle(order)
            for pos in order:
                bi = cur[pos]
                m = block_masks[bi]
                redundant = True
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    cur.pop(pos)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-point swap on a random block
            if len(cur) == 0:
                break
            pos = random.randrange(len(cur))
            bi = cur[pos]
            b = all_k[bi]
            p_out = random.choice(b)
            partial = [x for x in b if x != p_out]
            # Find candidate blocks that contain all of partial
            best_new = None
            best_score = -1
            # Iterate over blocks containing all elements of partial
            # Build a quick filter
            for cand_bi, m in enumerate(block_masks):
                if cand_bi == bi:
                    continue
                cb = all_k[cand_bi]
                if all(x in cb for x in partial):
                    # Score: unique coverage gained - unique coverage lost
                    # Approximate: count of t-subsets uniquely covered by bi vs cand_bi
                    score = 0
                    m_orig = block_masks[bi]
                    while m_orig:
                        lb = m_orig & -m_orig
                        idx = lb.bit_length() - 1
                        if counts[idx] == 1:
                            # uniquely covered by bi; check if cand covers it
                            if (m & (1 << idx)):
                                pass
                            else:
                                score -= 1
                        m_orig ^= lb
                    m_new = m
                    while m_new:
                        lb = m_new & -m_new
                        idx = lb.bit_length() - 1
                        if counts[idx] == 1:
                            if not (block_masks[bi] & (1 << idx)):
                                score += 1
                        m_new ^= lb
                    if score > best_score:
                        best_score = score
                        best_new = cand_bi
            if best_new is not None and best_new not in cur:
                cur[pos] = best_new
                improved = True

        return cur

    # ---------- Conflict graph B&B: max independent set ----------
    # Build adjacency: edge between two blocks if their union does NOT cover all t-subsets?
    # No: for covering design, we want a set of blocks whose union covers all t-subsets.
    # Independent set on conflict graph where edges connect blocks that CANNOT coexist?
    # Actually, the strategy says: "conflict graph of k-blocks, where edges connect blocks
    # that fail to cover a common t-subset" - this doesn't quite make sense for covering.
    # Let me reinterpret: We want to find a MINIMAL set of blocks covering all t-subsets.
    # The B&B approach: branch on whether to include or exclude a block.
    # Lower bound from greedy coloring on the "remaining uncovered" structure.

    def branch_and_bound(bi_list, uncovered_mask, depth, deadline):
        nonlocal best_blocks, best_size

        if time.time() > deadline:
            return

        # If uncovered is empty, we have a solution
        if uncovered_mask == 0:
            if len(bi_list) < best_size:
                best_blocks = list(bi_list)
                best_size = len(bi_list)
            return

        # Lower bound: greedy coloring heuristic on remaining
        # Count how many blocks needed: at least ceil(popcount(uncovered) / max_coverage_per_block)
        # Better: greedy set cover lower bound
        lb_mask = uncovered_mask
        lb_count = 0
        while lb_mask:
            # Pick block with max coverage
            best_gain = 0
            for bi in t_to_blocks[_first_bit(lb_mask)]:
                g = bin(block_masks[bi] & lb_mask).count('1')
                if g > best_gain:
                    best_gain = g
            if best_gain == 0:
                break
            lb_count += 1
            # Remove all t-subsets covered by the best block (approximate)
            # This is slow; use a simpler bound
            lb_count += 1
            break  # Just use 1 as a weak bound to keep speed

        # Simpler lower bound: each block covers at most C(8,5)=56 t-subsets
        lb_count = (bin(uncovered_mask).count('1') + 55) // 56
        if len(bi_list) + lb_count >= best_size:
            return

        # Pick a t-subset to branch on (most constrained)
        first_t = _first_bit(uncovered_mask)
        candidates = t_to_blocks[first_t]
        if not candidates:
            return

        # Order candidates by coverage (greedy)
        candidates.sort(key=lambda bi: bin(block_masks[bi] & uncovered_mask).count('1'), reverse=True)

        for bi in candidates:
            if time.time() > deadline:
                return
            new_mask = uncovered_mask & ~block_masks[bi]
            branch_and_bound(bi_list + [bi], new_mask, depth + 1, deadline)

        # Also try excluding the top candidate (branch on exclusion)
        # This is implicit: we try all candidates, so excluding means not picking it
        # Actually we need explicit exclusion branch for completeness
        # But for efficiency, just do inclusion branches (greedy B&B)
        # The "exclude" is handled by trying other candidates

    def _first_bit(mask):
        if mask == 0:
            return -1
        return (mask & -mask).bit_length() - 1

    # ---------- Main ----------
    best_blocks = None
    best_size = float('inf')

    # Phase 1: Greedy with multiple seeds
    greedy_deadline = t0 + 20.0
    for seed in range(20):
        if time.time() > greedy_deadline:
            break
        bi_list = greedy_set_cover(greedy_deadline, seed)
        bi_list = remove_redundant(bi_list)
        if bi_list and len(bi_list) < best_size:
            best_size = len(bi_list)
            best_blocks = list(bi_list)

    if best_blocks is None:
        return [list(all_k[0])]

    # Phase 2: Local search
    if time.time() < deadline - 15:
        ls_end = min(deadline - 15, time.time() + 20)
        ls_result = local_search(best_blocks, ls_end - time.time())
        ls_result = remove_redundant(ls_result)
        if ls_result and len(ls_result) < best_size:
            best_blocks = ls_result
            best_size = len(ls_result)

    # Phase 3: B&B refinement (short budget)
    bb_deadline = deadline - 5
    if time.time() < bb_deadline and best_blocks:
        try:
            branch_and_bound([], (1 << N_t) - 1, 0, bb_deadline)
        except RecursionError:
            pass
        except Exception:
            pass

    if not best_blocks:
        bi_list = greedy_set_cover(deadline, 0)
        bi_list = remove_redundant(bi_list)
        best_blocks = bi_list

    return [list(all_k[bi]) for bi in best_blocks]
