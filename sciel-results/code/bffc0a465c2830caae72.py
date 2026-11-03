"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    total_space = q ** n

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover masks as Python ints (bitmasks)
    cover_masks = []
    for w in all_words:
        mask = 1 << word_to_idx(w)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    mask |= (1 << word_to_idx(nw))
        cover_masks.append(mask)

    # Initial code from tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    base_code_idx = [word_to_idx(cw) for cw in initial]

    def compute_union(idx_list):
        """Compute union of cover masks using numpy for speed."""
        if not idx_list:
            return 0
        masks = [cover_masks[i] for i in idx_list]
        # Use Python int OR for exactness (numpy can't handle arbitrary precision well)
        union = 0
        for m in masks:
            union |= m
        return union

    def greedy_reduce(idx_list):
        if len(idx_list) <= 1:
            return idx_list[:]
        masks = [cover_masks[i] for i in idx_list]
        full_union = compute_union(idx_list)
        n_cw = len(idx_list)
        keep = [True] * n_cw
        for i in range(n_cw):
            if time.time() > deadline:
                break
            if not keep[i]:
                continue
            union_without = 0
            for j in range(n_cw):
                if j != i and keep[j]:
                    union_without |= masks[j]
            if union_without == full_union:
                keep[i] = False
        return [idx_list[i] for i in range(n_cw) if keep[i]]

    def local_search(idx_list, max_rounds=30):
        code = idx_list[:]
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            n_cw = len(code)
            if n_cw == 0:
                break
            masks = [cover_masks[i] for i in code]
            full_union = compute_union(code)
            
            # Phase 1: Remove redundant codewords
            for i in range(n_cw):
                if time.time() > deadline:
                    break
                union_without = 0
                for j in range(n_cw):
                    if j != i:
                        union_without |= masks[j]
                if union_without == full_union:
                    code.pop(i)
                    masks.pop(i)
                    full_union = union_without
                    improved = True
                    break
            if improved:
                continue
            
            # Phase 2: Swap optimization
            code_set = set(code)
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                union_without = 0
                for j in range(len(code)):
                    if j != i:
                        union_without |= masks[j]
                uncovered = full_union & ~union_without
                if uncovered == 0:
                    code.pop(i)
                    masks.pop(i)
                    improved = True
                    break
                best_gain = 0
                best_new_idx = -1
                best_new_mask = 0
                ci = code[i]
                w = all_words[ci]
                candidates = {ci}
                for pos in range(n):
                    for val in range(q):
                        if val != w[pos]:
                            nw = w[:]
                            nw[pos] = val
                            candidates.add(word_to_idx(nw))
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        for v1 in range(q):
                            if v1 == w[p1]:
                                continue
                            for v2 in range(q):
                                if v2 == w[p2]:
                                    continue
                                nw = w[:]
                                nw[p1] = v1
                                nw[p2] = v2
                                candidates.add(word_to_idx(nw))
                for cand in candidates:
                    if cand in code_set:
                        continue
                    m = cover_masks[cand]
                    new_union = union_without | m
                    if new_union != full_union:
                        continue
                    gain = bin(uncovered & m).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_new_idx = cand
                        best_new_mask = m
                if best_new_idx >= 0:
                    code_set.discard(code[i])
                    code[i] = best_new_idx
                    masks[i] = best_new_mask
                    code_set.add(best_new_idx)
                    improved = True
                    break
            if not improved:
                break
        return code

    def randomized_start(base_idx, seed):
        """Create a randomized variant of the base code."""
        rng = random.Random(seed)
        code = base_idx[:]
        # Randomly swap some codewords with nearby alternatives
        for _ in range(min(20, len(code) // 5)):
            if not code:
                break
            i = rng.randrange(len(code))
            w = all_words[code[i]]
            # Pick a random neighbor
            pos = rng.randrange(n)
            val = rng.randrange(q)
            if val == w[pos]:
                continue
            nw = w[:]
            nw[pos] = val
            new_idx = word_to_idx(nw)
            if new_idx not in code:
                code[i] = new_idx
        return code

    # Run multiple restarts
    best_code = None
    best_size = float('inf')
    
    # First pass: deterministic
    code_idx = greedy_reduce(base_code_idx[:])
    code_idx = local_search(code_idx)
    code_idx = greedy_reduce(code_idx)
    best_code = code_idx
    best_size = len(code_idx)
    
    # Additional randomized restarts if time permits
    seed = 42
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        rand_code = randomized_start(base_code_idx, seed + restart)
        rand_code = greedy_reduce(rand_code)
        rand_code = local_search(rand_code, max_rounds=15)
        rand_code = greedy_reduce(rand_code)
        
        if len(rand_code) < best_size:
            best_code = rand_code
            best_size = len(rand_code)
    
    # Final polish on the best code found
    if best_code is not None:
        best_code = local_search(best_code, max_rounds=20)
        best_code = greedy_reduce(best_code)
    
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
