"""SA / VNS covering design C(24,7,3) -- different family from greedy+LS."""
import itertools
import random
import time

import numpy as np


def _all_triples(v, t):
    return list(itertools.combinations(range(v), t))


def _triple_to_idx(triple, v, t):
    # Lexicographic rank of a t-combination from v elements
    # Precomputed via a table would be faster; here we use a dict built once.
    # We'll build the dict in the caller.
    raise NotImplementedError


def _build_triple_index(triples):
    return {tr: i for i, tr in enumerate(triples)}


def _cover_matrix(blocks, triples, t):
    """Return a boolean matrix (n_blocks x n_triples) where entry is True if
    the block covers the triple (i.e., triple is subset of block)."""
    n_blocks = len(blocks)
    n_triples = len(triples)
    # For each block, the set of triples it covers
    mat = np.zeros((n_blocks, n_triples), dtype=bool)
    for i, b in enumerate(blocks):
        bset = set(b)
        # enumerate all t-subsets of b
        for tr in itertools.combinations(b, t):
            # find index
            # We need a fast lookup. Precompute a dict.
            pass
    return mat


def _initial_greedy(v, k, t, rng, time_limit):
    """Fast greedy to get a decent starting point for SA."""
    deadline = time.time() + time_limit
    triples = _all_triples(v, t)
    triple_idx = _build_triple_index(triples)
    uncovered = np.ones(len(triples), dtype=bool)
    n_uncovered = len(triples)
    blocks = []

    # For speed, precompute for each point, the list of triple indices containing it
    # Actually, for gain computation we need: for a partial block P and candidate x,
    # how many uncovered triples contain x and (t-1) elements from P.
    # t=3: triples containing x and 2 elements from P (where those 2 are in P and x not in P)
    # i.e., for each pair (a,b) in P with a!=b, if x not in {a,b}, triple (x,a,b)
    
    # Precompute: for each pair (a,b), the triple index of (a,b,x) for each x not in {a,b}
    # That's C(v,2) * (v-2) entries. For v=24, that's 276*22 = 6072. Fine.
    pair_triple = {}
    for a in range(v):
        for b in range(a+1, v):
            for x in range(v):
                if x != a and x != b:
                    tr = tuple(sorted((a, b, x)))
                    pair_triple[(a, b, x)] = triple_idx[tr]
    
    # For gain(x, partial_block): sum over pairs (p1,p2) in partial_block of
    #   uncovered[pair_triple[(p1,p2,x)]] if x not in {p1,p2}
    
    while n_uncovered > 0 and time.time() < deadline:
        # Pick a random uncovered triple to seed the block
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        seed_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
        seed_triple = triples[seed_idx]
        block = list(seed_triple)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        p1, p2 = block[i], block[j]
                        if x != p1 and x != p2:
                            idx = pair_triple[(p1, p2, x)]
                            if uncovered[idx]:
                                gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and rng.random() < 0.2:
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        bset = set(block)
        for tr in itertools.combinations(block, t):
            idx = triple_idx[tr]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
    
    return blocks, triples, triple_idx, uncovered, n_uncovered


def _compute_uncovered(blocks, triple_idx, t):
    """Compute the uncovered boolean array and count."""
    # Use a set of covered triples
    covered_set = set()
    for b in blocks:
        for tr in itertools.combinations(b, t):
            covered_set.add(tr)
    # Map to indices
    uncovered = np.ones(len(triple_idx), dtype=bool)
    for tr, idx in triple_idx.items():
        if tr in covered_set:
            uncovered[idx] = False
    n_unc = int(np.sum(uncovered))
    return uncovered, n_unc


def _sa_search(blocks, v, k, t, rng, time_limit):
    """Simulated annealing over block multiset."""
    deadline = time.time() + time_limit
    triples = _all_triples(v, t)
    triple_idx = _build_triple_index(triples)
    n_triples = len(triples)
    
    # Initial state
    current_blocks = [list(b) for b in blocks]
    current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
    
    best_blocks = [list(b) for b in current_blocks]
    best_n_unc = current_n_unc
    
    # Precompute pair_triple for fast gain in moves
    pair_triple = {}
    for a in range(v):
        for b in range(a+1, v):
            for x in range(v):
                if x != a and x != b:
                    tr = tuple(sorted((a, b, x)))
                    pair_triple[(a, b, x)] = triple_idx[tr]
    
    # Temperature schedule
    t0 = 2.0
    t_end = 0.01
    # Number of steps: we'll run until deadline
    # Each step: generate a move, compute delta, accept/reject
    
    start_time = time.time()
    step = 0
    current_temp = t0
    # Estimate steps: aim for ~100k steps
    # But we'll just run until deadline and cool based on elapsed fraction
    
    while time.time() < deadline:
        elapsed_frac = (time.time() - start_time) / (deadline - start_time)
        current_temp = t0 * (t_end / t0) ** elapsed_frac
        
        n_blocks = len(current_blocks)
        move_type = rng.random()
        
        if n_blocks > 0 and move_type < 0.4:
            # Replace a random block
            i = rng.randint(0, n_blocks - 1)
            old_block = current_blocks[i]
            # New block: random k-subset, but biased toward covering uncovered triples
            # Pick a random uncovered triple as seed
            unc_idx = np.where(current_uncovered)[0]
            if len(unc_idx) == 0:
                new_block = sorted(rng.sample(range(v), k))
            else:
                seed_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
                seed_triple = triples[seed_idx]
                new_block = list(seed_triple)
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_block]
                    if not candidates:
                        break
                    new_block.append(rng.choice(candidates))
                new_block.sort()
            
            # Compute delta in uncovered count
            # Old block covers some triples; new block covers some.
            # Delta = (newly uncovered by removing old) - (newly covered by adding new)
            # More precisely: 
            #   old_covered = set of triples covered by old_block
            #   new_covered = set of triples covered by new_block
            #   triples that were only covered by old_block (not by other blocks) become uncovered
            #   triples that were uncovered but are covered by new_block become covered
            # This is expensive to compute exactly. Approximate:
            #   delta ≈ -|new_covered ∩ uncovered| + |old_covered ∩ (uncovered ∪ covered_by_old_only)|
            # For simplicity, compute the actual change by checking each triple in old/new
            
            old_covered = set(itertools.combinations(old_block, t))
            new_covered = set(itertools.combinations(new_block, t))
            
            # Triples that become uncovered: in old_covered, not in new_covered, and not covered by any other block
            # This requires checking other blocks. Expensive.
            # Approximation: just use the net change in the uncovered set assuming no other block covers them.
            # This is an approximation that may lead to accepting bad moves, but SA can handle noise.
            
            delta = 0
            for tr in old_covered:
                if tr not in new_covered:
                    # Was this triple covered by any other block?
                    # Check if it's in current_uncovered (it shouldn't be, since old_block covers it)
                    # Actually, current_uncovered[triple_idx[tr]] tells us if it's uncovered.
                    # If it's NOT uncovered, it means some other block also covers it, so removing old doesn't uncover it.
                    # If it IS uncovered... that can't happen since old_block covers it.
                    # So: if current_uncovered[triple_idx[tr]] is False, it's covered by someone else.
                    idx = triple_idx[tr]
                    if current_uncovered[idx]:
                        # This shouldn't happen, but if it does, removing old uncovers it
                        delta += 1
                    # If not uncovered, it's covered by others, so no change
            for tr in new_covered:
                if tr not in old_covered:
                    idx = triple_idx[tr]
                    if current_uncovered[idx]:
                        delta -= 1
            
            # Accept/reject
            if delta <= 0 or rng.random() < np.exp(-delta / current_temp):
                current_blocks[i] = new_block
                # Update uncovered
                # Recompute from scratch for correctness (expensive but safe)
                current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
                if current_n_unc < best_n_unc:
                    best_n_unc = current_n_unc
                    best_blocks = [list(b) for b in current_blocks]
                    
        elif n_blocks > 1 and move_type < 0.6:
            # Swap two blocks
            i, j = rng.sample(range(n_blocks), 2)
            # Generate new versions of both
            unc_idx = np.where(current_uncovered)[0]
            new_blocks = []
            for idx in [i, j]:
                if len(unc_idx) == 0:
                    nb = sorted(rng.sample(range(v), k))
                else:
                    seed_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
                    seed_triple = triples[seed_idx]
                    nb = list(seed_triple)
                    while len(nb) < k:
                        candidates = [x for x in range(v) if x not in nb]
                        if not candidates:
                            break
                        nb.append(rng.choice(candidates))
                    nb.sort()
                new_blocks.append(nb)
            
            # Approximate delta
            old_i = current_blocks[i]
            old_j = current_blocks[j]
            old_covered = set(itertools.combinations(old_i, t)) | set(itertools.combinations(old_j, t))
            new_covered = set(itertools.combinations(new_blocks[0], t)) | set(itertools.combinations(new_blocks[1], t))
            
            delta = 0
            for tr in old_covered:
                if tr not in new_covered:
                    idx = triple_idx[tr]
                    if current_uncovered[idx]:
                        delta += 1
            for tr in new_covered:
                if tr not in old_covered:
                    idx = triple_idx[tr]
                    if current_uncovered[idx]:
                        delta -= 1
            
            if delta <= 0 or rng.random() < np.exp(-delta / current_temp):
                current_blocks[i] = new_blocks[0]
                current_blocks[j] = new_blocks[1]
                current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
                if current_n_unc < best_n_unc:
                    best_n_unc = current_n_unc
                    best_blocks = [list(b) for b in current_blocks]
                    
        elif n_blocks > 0 and move_type < 0.8:
            # Remove a block (if it doesn't uncover any triple)
            i = rng.randint(0, n_blocks - 1)
            old_block = current_blocks[i]
            old_covered = set(itertools.combinations(old_block, t))
            # Check if any triple in old_covered is only covered by this block
            would_uncover = 0
            for tr in old_covered:
                idx = triple_idx[tr]
                if current_uncovered[idx]:
                    would_uncover += 1
                else:
                    # Check if covered by another block
                    # This is expensive; approximate by checking if it's in the uncovered set
                    # Since it's not uncovered, it's covered by at least one block. If it's only covered by this one,
                    # removing it would uncover it. We can't easily tell without checking all blocks.
                    # For now, assume it's safe (optimistic). This may lead to invalid states temporarily.
                    pass
            
            if would_uncover == 0:
                # Try removing
                delta = 1  # we reduce block count by 1, but uncovered count might increase
                # Actually, if we remove and it doesn't uncover anything, uncovered stays same, block count decreases.
                # Our objective is primarily uncovered count, secondarily block count.
                # Let's use a combined objective: uncovered_count + 0.1 * block_count
                # For now, just remove if it doesn't increase uncovered.
                current_blocks.pop(i)
                current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
                if current_n_unc <= best_n_unc:
                    if current_n_unc < best_n_unc or len(current_blocks) < len(best_blocks):
                        best_n_unc = current_n_unc
                        best_blocks = [list(b) for b in current_blocks]
            # If would_uncover > 0, don't remove (or accept with SA probability based on would_uncover)
            elif rng.random() < np.exp(-would_uncover / current_temp):
                current_blocks.pop(i)
                current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
                if current_n_unc < best_n_unc:
                    best_n_unc = current_n_unc
                    best_blocks = [list(b) for b in current_blocks]
                    
        else:
            # Add a new block
            unc_idx = np.where(current_uncovered)[0]
            if len(unc_idx) > 0:
                seed_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
                seed_triple = triples[seed_idx]
                new_block = list(seed_triple)
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_block]
                    if not candidates:
                        break
                    new_block.append(rng.choice(candidates))
                new_block.sort()
                current_blocks.append(new_block)
                current_uncovered, current_n_unc = _compute_uncovered(current_blocks, triple_idx, t)
                if current_n_unc < best_n_unc:
                    best_n_unc = current_n_unc
                    best_blocks = [list(b) for b in current_blocks]
                # If adding didn't help, remove it with some probability (or let SA handle it)
                # Actually, adding always helps or is neutral, so keep it.
        
        step += 1
    
    return best_blocks


def _remove_redundant(blocks, v, k, t):
    """Remove blocks that don't cover any unique triple."""
    triples = _all_triples(v, t)
    triple_idx = _build_triple_index(triples)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            other = blocks[:i] + blocks[i+1:]
            # Check if block i covers any triple not covered by others
            redundant = True
            for tr in itertools.combinations(blocks[i], t):
                covered = False
                for ob in other:
                    if set(tr).issubset(set(ob)):
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break
    return blocks


def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # safety margin for 150s kill
    
    # Phase 1: Greedy initialization
    greedy_time = total_budget * 0.3
    rng = random.Random(42)
    blocks, triples, triple_idx, uncovered, n_unc = _initial_greedy(v, k, t, rng, greedy_time)
    
    # Phase 2: SA search
    sa_time = total_budget * 0.6
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        rng_sa = random.Random(12345)
        sa_blocks = _sa_search(blocks, v, k, t, rng_sa, min(sa_time, remaining))
        if len(sa_blocks) <= len(blocks):
            blocks = sa_blocks
    
    # Phase 3: Remove redundant blocks
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 5.0:
        blocks = _remove_redundant(blocks, v, k, t)
    
    # Final check: ensure validity (all triples covered)
    # If not valid, fall back to a simple greedy
    final_uncovered, final_n_unc = _compute_uncovered(blocks, triple_idx, t)
    if final_n_unc > 0:
        # Need to cover remaining triples
        rng_fix = random.Random(999)
        deadline = start_time + total_budget - 5.0
        while final_n_unc > 0 and time.time() < deadline:
            unc_idx = np.where(final_uncovered)[0]
            seed_idx = unc_idx[0]
            seed_triple = triples[seed_idx]
            block = list(seed_triple)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                # Pick the candidate that covers the most uncovered triples
                best_x = None
                best_gain = -1
                for x in candidates:
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1, p2 = block[i], block[j]
                            if x != p1 and x != p2:
                                tr = tuple(sorted((x, p1, p2)))
                                idx = triple_idx[tr]
                                if final_uncovered[idx]:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            if len(block) < k:
                break
            block.sort()
            blocks.append(block)
            # Update uncovered
            bset = set(block)
            for tr in itertools.combinations(block, t):
                idx = triple_idx[tr]
                if final_uncovered[idx]:
                    final_uncovered[idx] = False
                    final_n_unc -= 1
    
    return blocks
