"""Covering design C(v,k,t) via strong-greedy max-gain block selection (bitset accelerated)."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s / 150s kill
    
    # Set random seed for reproducibility if needed, or just use system random
    # random.seed(42) 

    # Map every t-subset to an integer id; use a Python int as a bitset for covered.
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # pair_mask[x][a][b] = bitmask of t-subsets containing x, a, b (for t=3).
    # Generalize: for t-subsets, precompute mask of t-subsets containing a given (t-1)-tuple.
    # We'll build, for each point x, a dict from frozenset(other t-1 points) -> bitmask of
    # t-subsets containing x and those others.
    # For efficiency with t=3: mask3[x][a][b].
    # Build generic: masks[x] is a dict keyed by tuple of (t-1) other points -> bitmask.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    covered = 0  # bitset of covered t-subsets
    blocks = []

    # Helper to count bits, using bit_count if available (Python 3.10+)
    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    # For a partial block (list/set B) and candidate x not in B, compute gain =
    # number of uncovered t-subsets that would become covered by adding x.
    # A t-subset {x} ∪ S (|S|=t-1, S ⊆ B) is newly covered iff it contains x and S ⊆ B.
    def gain_of_add(x, B_set):
        g = 0
        # iterate over (t-1)-subsets of B
        B_list = list(B_set)
        for S in itertools.combinations(B_list, t - 1):
            key = tuple(S)
            m = masks[x].get(key)
            if m is not None:
                g += count_bits(m & ~covered)
        return g

    def build_block_from(seed_points):
        # Grow a block from an initial set to size k by max-gain greedy.
        B = set(seed_points)
        # List of points not in B
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in available:
                g = gain_of_add(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    def run_greedy(rng_seed):
        """Run one greedy pass with a given RNG seed; return (blocks, covered_mask)."""
        rng = random.Random(rng_seed)
        cov = 0
        blks = []
        full_mask = (1 << n_subsets) - 1
        
        while True:
            if time.time() > deadline:
                break
            
            uncov_mask = full_mask & ~cov
            if uncov_mask == 0:
                break
            
            uncovered_indices = []
            m = uncov_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                uncovered_indices.append(idx)
                m ^= lsb
            
            if not uncovered_indices:
                break
            
            sample_size = min(80, len(uncovered_indices))
            sampled_indices = rng.sample(uncovered_indices, sample_size)
            
            best_block = None
            best_ratio = -1.0
            
            for idx in sampled_indices:
                seed = all_subsets[idx]
                B = build_block_from(set(seed))
                if len(B) < k:
                    continue
                total = 0
                for s in itertools.combinations(sorted(B), t):
                    total += 1 if not (cov >> subset_to_index[s]) & 1 else 0
                if total > 0:
                    ratio = total / len(B)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_block = B
            
            if best_block is None:
                break
            
            bl = sorted(best_block)
            for s in itertools.combinations(bl, t):
                cov |= 1 << subset_to_index[s]
            blks.append(bl)
        
        return blks, cov

    # Try multiple restarts with different seeds, keep the best (fewest blocks)
    best_result = None
    best_count = float('inf')
    
    max_restarts = 3
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        blks, cov = run_greedy(42 + r * 1000)
        if len(blks) < best_count:
            best_count = len(blks)
            best_result = blks
        # If we've already hit a good count, reduce restarts
        if best_count <= 74:
            break
    
    if best_result is None:
        # Fallback: single deterministic run
        blks, cov = run_greedy(42)
        blocks = blks
    else:
        blocks = best_result

    # Post-processing: Remove redundant blocks (multiple passes)
    if blocks:
        for _pass in range(3):
            if time.time() > deadline - 2.0:
                break
            block_triplets = []
            for b in blocks:
                tps = [subset_to_index[s] for s in itertools.combinations(b, t)]
                block_triplets.append(tps)
            
            triplet_count = [0] * n_subsets
            for tps in block_triplets:
                for idx in tps:
                    triplet_count[idx] += 1
            
            new_blocks = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for idx in block_triplets[i]:
                    if triplet_count[idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(b)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
            else:
                break

    return blocks
