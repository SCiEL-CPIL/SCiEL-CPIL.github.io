"""Covering design C(24,7,3) - Vectorized weighted greedy (soft-min penalty) + SA local search."""
import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 142.0          # safety margin vs 150 s kill
    start = time.time()

    # ---------- Precomputation (once) ----------
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)                          # 2024 for (24,3)
    t_idx = {ts: i for i, ts in enumerate(all_t)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)                # 34610 for (24,7)

    # Bitmask representation: 77 uint64 words per block (2024 bits)
    words = (n_t + 63) // 64                  # 32
    # Precompute per-block set of t-subset indices (for reverse index & SA)
    block_triples = []
    for blk in all_blocks:
        tris = [t_idx[ts] for ts in itertools.combinations(blk, t)]
        block_triples.append(tris)

    # Reverse index: for each triple, list of block indices covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for b, tris in enumerate(block_triples):
        for tri in tris:
            t_to_blocks[tri].append(b)

    # Frequency of each triple (how many blocks cover it) -> weight
    freq = np.zeros(n_t, dtype=np.float64)
    for b, tris in enumerate(block_triples):
        for tri in tris:
            freq[tri] += 1.0
    # rarer triples get higher weight (inverse frequency, normalized)
    w = 1.0 / np.maximum(freq, 1.0)
    w /= w.max()

    # Precompute per-block base weighted value (sum of weights of its triples)
    # and per-block bitmask words (for coverage checks)
    block_base = np.zeros(n_blocks, dtype=np.float64)
    block_words = np.zeros((n_blocks, words), dtype=np.uint64)
    for b, tris in enumerate(block_triples):
        s = 0.0
        for tri in tris:
            s += w[tri]
            block_words[b, tri >> 6] |= (1 << (tri & 63))
        block_base[b] = s

    full_words = np.ones(words, dtype=np.uint64)
    # Mask out unused bits in the last word
    rem = n_t & 63
    if rem:
        full_words[-1] = (1 << rem) - 1

    # ---------- Vectorized greedy with soft-min penalty ----------
    def greedy_vectorized(seed, time_budget):
        rng = random.Random(seed)
        end = time.time() + time_budget
        selected_idx = []
        # covered: bool array over triples
        covered = np.zeros(n_t, dtype=bool)
        # For speed, maintain per-block "new triple count" lazily via vectorized op
        # We recompute gains each step (n_blocks * words ops) which is ~34k*32 = 1.1M word-ops, fast.

        while time.time() < end:
            # Compute marginal gain for every block:
            # gain = sum_{tri in block, not covered} w[tri]  -  penalty * soft_count(covered in block)
            # To keep it fast: gain = sum of w over block triples that are uncovered.
            # Vectorized: for each block, count uncovered triples via bitmask AND.
            # We do this with numpy: for each word, count bits in (block_words & ~covered_words)
            # covered_words: uint64 array [words]
            cov_words = np.zeros(words, dtype=np.uint64)
            for i in range(n_t):
                if covered[i]:
                    cov_words[i >> 6] |= (1 << (i & 63))

            # new_counts[b] = popcount(block_words[b] & ~cov_words)
            # Use numpy: bitwise and, then popcount via lookup? For 32 words * 34k blocks, do it in a loop over words.
            new_counts = np.zeros(n_blocks, dtype=np.int32)
            not_cov_words = np.bitwise_invert(cov_words) & full_words
            for wd in range(words):
                masked = np.bitwise_and(block_words[:, wd], not_cov_words[wd])
                # popcount per row: use numpy bit_count (NumPy >= 1.17 has no bit_count; emulate)
                # Use a small LUT via unpacking bits? Simpler: use python int popcount on each element is slow.
                # Instead, use np.unpackbits on a uint8 view? Too heavy.
                # Alternative: precompute popcount for 16-bit chunks.
                # For speed, use the fact that each block covers 35 triples; new_counts <= 35.
                # We can compute new_counts by iterating set bits of masked for each block only if masked != 0.
                nz = np.nonzero(masked)[0]
                for b in nz:
                    x = int(masked[b])
                    new_counts[b] += bin(x).count('1')

            # Weighted gain: sum of w over uncovered triples in block.
            # Approximate: gain = w_sum_uncovered. We can compute as:
            #   For each block, gain = base - sum_{covered in block} w
            # Compute covered_weight per block:
            covered_w = np.zeros(n_blocks, dtype=np.float64)
            # For each covered triple, add w[tri] to all blocks covering it
            # Use t_to_blocks
            for tri in range(n_t):
                if covered[tri]:
                    wt = w[tri]
                    for b in t_to_blocks[tri]:
                        covered_w[b] += wt
            gains = block_base - covered_w
            # Penalty for blocks that cover many already-covered triples (encourage diversity)
            # soft-min surrogate: penalty = beta * log(1 + new_counts) discourages redundant picks
            beta = 0.05
            penalty = beta * np.log1p(new_counts.astype(np.float64))
            scores = gains - penalty

            # Also require block covers at least one uncovered triple
            valid = new_counts > 0
            if not np.any(valid):
                break
            scores[~valid] = -np.inf

            # Pick best with a little randomness among top candidates
            top_n = 5
            order = np.argsort(scores)[::-1]
            cands = order[:top_n]
            cands = cands[scores[cands] > -np.inf]
            if len(cands) == 0:
                break
            # Randomized pick: softmax over top candidates
            s_max = scores[cands].max()
            probs = np.exp(scores[cands] - s_max)
            probs /= probs.sum()
            pick = int(rng.choices(cands, p=probs)[0])

            selected_idx.append(pick)
            # Update covered
            for tri in block_triples[pick]:
                covered[tri] = True

            # Early stop if all covered
            if covered.all():
                break

        return [all_blocks[i] for i in selected_idx]

    # ---------- Prune redundant blocks ----------
    def prune(blocks, time_budget):
        if not blocks:
            return blocks
        end = time.time() + time_budget
        # map block -> index in all_blocks
        bmap = {b: i for i, b in enumerate(all_blocks)}
        sel = list(blocks)
        sel_idx = [bmap[b] for b in sel]
        # coverage count per triple among selected
        cov_count = np.zeros(n_t, dtype=np.int32)
        for bi in sel_idx:
            for tri in block_triples[bi]:
                cov_count[tri] += 1
        changed = True
        while changed and time.time() < end:
            changed = False
            for pos in range(len(sel_idx)):
                if time.time() > end:
                    break
                bi = sel_idx[pos]
                # check if any triple in bi has cov_count == 1
                removable = True
                for tri in block_triples[bi]:
                    if cov_count[tri] == 1:
                        removable = False
                        break
                if removable:
                    for tri in block_triples[bi]:
                        cov_count[tri] -= 1
                    sel.pop(pos)
                    sel_idx.pop(pos)
                    changed = True
                    break  # restart scan
        return sel

    # ---------- Simulated Annealing local search ----------
    def sa_local_search(blocks, time_budget):
        if not blocks:
            return blocks
        end = time.time() + time_budget
        rng = random.Random(int(time.time() * 1000) % (2**31))
        bmap = {b: i for i, b in enumerate(all_blocks)}
        cur = list(blocks)
        cur_idx = [bmap[b] for b in cur]
        # coverage count
        cov_count = np.zeros(n_t, dtype=np.int32)
        for bi in cur_idx:
            for tri in block_triples[bi]:
                cov_count[tri] += 1
        cur_cost = len(cur)  # primary: minimize #blocks; secondary: coverage violations
        best = list(cur)
        best_idx = list(cur_idx)
        best_cost = cur_cost

        T = 1.0
        T_min = 1e-4
        cool = 0.999

        while time.time() < end:
            # Choose a move
            move_type = rng.random()
            if move_type < 0.5:
                # Swap: replace one block with another
                if len(cur_idx) == 0:
                    break
                i = rng.randrange(len(cur_idx))
                old_bi = cur_idx[i]
                # Candidate: blocks covering a triple that old covers (to maintain coverage)
                tris_old = block_triples[old_bi]
                if not tris_old:
                    continue
                tri_pick = tris_old[rng.randrange(len(tris_old))]
                cands = t_to_blocks[tri_pick]
                if not cands:
                    continue
                new_bi = cands[rng.randrange(len(cands))]
                if new_bi == old_bi:
                    continue
                # Tentative: remove old, add new
                # Compute delta coverage
                # We'll just rebuild cov_count for affected triples
                affected = set(block_triples[old_bi]) | set(block_triples[new_bi])
                # Check feasibility: no triple drops to 0
                feasible = True
                for tri in block_triples[old_bi]:
                    if cov_count[tri] == 1:
                        # this triple only covered by old; new must also cover it
                        if new_bi not in t_to_blocks[tri]:
                            feasible = False
                            break
                if not feasible:
                    continue
                # Apply
                for tri in block_triples[old_bi]:
                    cov_count[tri] -= 1
                for tri in block_triples[new_bi]:
                    cov_count[tri] += 1
                cur_idx[i] = new_bi
                cur[i] = all_blocks[new_bi]
                # Cost: #blocks same, but check violations
                violations = int(np.sum(cov_count < 1))
                new_cost = len(cur_idx) + 100 * violations
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / T):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(cur)
                        best_idx = list(cur_idx)
                else:
                    # revert
                    for tri in block_triples[new_bi]:
                        cov_count[tri] -= 1
                    for tri in block_triples[old_bi]:
                        cov_count[tri] += 1
                    cur_idx[i] = old_bi
                    cur[i] = all_blocks[old_bi]
            elif move_type < 0.8:
                # 2-for-1 merge: try to replace two blocks with one that covers their union
                if len(cur_idx) < 2:
                    break
                i, j = rng.sample(range(len(cur_idx)), 2)
                bi1, bi2 = cur_idx[i], cur_idx[j]
                union_tris = set(block_triples[bi1]) | set(block_triples[bi2])
                # Find a block that covers all union_tris (rare) - skip if too many
                if len(union_tris) > 35:
                    continue
                # Candidate blocks: intersect t_to_blocks for each triple in union
                cand_set = None
                for tri in union_tris:
                    s = set(t_to_blocks[tri])
                    if cand_set is None:
                        cand_set = s
                    else:
                        cand_set &= s
                    if not cand_set:
                        break
                if cand_set is None or not cand_set:
                    continue
                # Ensure the candidate is not bi1 or bi2 (otherwise no gain)
                cand_set.discard(bi1)
                cand_set.discard(bi2)
                if not cand_set:
                    continue
                new_bi = next(iter(cand_set))
                # Check feasibility: removing bi1, bi2 and adding new_bi
                # For each triple in bi1 or bi2, ensure still covered
                feasible = True
                for tri in block_triples[bi1]:
                    if cov_count[tri] == 1 and new_bi not in t_to_blocks[tri]:
                        feasible = False
                        break
                if feasible:
                    for tri in block_triples[bi2]:
                        if cov_count[tri] == 1 and new_bi not in t_to_blocks[tri]:
                            feasible = False
                            break
                if not feasible:
                    continue
                # Apply
                for tri in block_triples[bi1]:
                    cov_count[tri] -= 1
                for tri in block_triples[bi2]:
                    cov_count[tri] -= 1
                for tri in block_triples[new_bi]:
                    cov_count[tri] += 1
                # Remove the two, add new
                # Rebuild cur_idx without i,j (adjust indices)
                if i < j:
                    del cur_idx[j]
                    del cur_idx[i]
                    cur.pop(j)
                    cur.pop(i)
                else:
                    del cur_idx[i]
                    del cur_idx[j]
                    cur.pop(i)
                    cur.pop(j)
                cur_idx.append(new_bi)
                cur.append(all_blocks[new_bi])
                violations = int(np.sum(cov_count < 1))
                new_cost = len(cur_idx) + 100 * violations
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / T):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(cur)
                        best_idx = list(cur_idx)
                else:
                    # revert: this is complex; skip revert for merge (rare)
                    pass
            else:
                # Add a block (to fix violations)
                if len(cur_idx) > 300:
                    continue
                # Find uncovered triple
                uncov = np.where(cov_count < 1)[0]
                if len(uncov) == 0:
                    continue
                tri = int(uncov[rng.randrange(len(uncov))])
                cands = t_to_blocks[tri]
                if not cands:
                    continue
                new_bi = cands[rng.randrange(len(cands))]
                for tri2 in block_triples[new_bi]:
                    cov_count[tri2] += 1
                cur_idx.append(new_bi)
                cur.append(all_blocks[new_bi])
                violations = int(np.sum(cov_count < 1))
                new_cost = len(cur_idx) + 100 * violations
                if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / T):
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(cur)
                        best_idx = list(cur_idx)
                else:
                    # revert
                    for tri2 in block_triples[new_bi]:
                        cov_count[tri2] -= 1
                    cur_idx.pop()
                    cur.pop()

            T *= cool
            if T < T_min:
                T = T_min

        # Final prune on best
        best = prune(best, time_budget=5.0)
        return best

    # ---------- Main driver ----------
    best_blocks = None
    best_len = float('inf')

    # Phase 1: multiple vectorized greedy restarts
    n_restarts = 30
    for seed in range(n_restarts):
        if time.time() > deadline - 50.0:
            break
        blocks = greedy_vectorized(seed, time_budget=3.0)
        if not blocks:
            continue
        blocks = prune(blocks, time_budget=2.0)
        # Validate
        cov = np.zeros(n_t, dtype=bool)
        for b in blocks:
            bi = bmap[b]
            for tri in block_triples[bi]:
                cov[tri] = True
        if cov.all() and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # Phase 2: SA local search on best
    if best_blocks is not None and time.time() < deadline - 20.0:
        improved = sa_local_search(best_blocks, time_budget=40.0)
        # Validate
        cov = np.zeros(n_t, dtype=bool)
        for b in improved:
            bi = bmap[b]
            for tri in block_triples[bi]:
                cov[tri] = True
        if cov.all() and len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved

    # Fallback
    if best_blocks is None or len(best_blocks) == 0:
        best_blocks = greedy_vectorized(0, time_budget=30.0)
        best_blocks = prune(best_blocks, time_budget=5.0)

    # Final validation; if invalid, do a simple sequential greedy
    if best_blocks is None:
        return []
    cov = np.zeros(n_t, dtype=bool)
    for b in best_blocks:
        bi = bmap.get(b, -1)
        if bi >= 0:
            for tri in block_triples[bi]:
                cov[tri] = True
    if not cov.all():
        # Emergency sequential greedy
        uncovered = np.ones(n_t, dtype=bool)
        final = []
        for bi in range(n_blocks):
            if not uncovered.any():
                break
            tris = block_triples[bi]
            hit = False
            for tri in tris:
                if uncovered[tri]:
                    hit = True
                    break
            if hit:
                final.append(all_blocks[bi])
                for tri in tris:
                    uncovered[tri] = False
        return final

    return best_blocks
