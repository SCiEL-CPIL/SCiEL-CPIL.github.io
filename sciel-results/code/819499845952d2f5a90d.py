"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

# Precompute all t-subsets as frozensets for fast membership
def _all_tsubsets(v, t):
    return set(itertools.combinations(range(v), t))


def _greedy_cover(v, k, t, seed=None, time_limit=110.0):
    """
    Randomized greedy covering with fast gain computation and redundant block removal.
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_frozen = [frozenset(sub) for sub in all_tsubs]
    num_tsubs = len(all_tsubs_frozen)

    # Map each t-subset frozenset to an index for fast array access
    tsub_index = {S: i for i, S in enumerate(all_tsubs_frozen)}

    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * num_tsubs
    blocks = []  # list of frozensets

    # Precompute for each (t-1)-subset R (as a sorted tuple), the list of t-subset indices
    # that contain R. This lets us compute gain quickly.
    # For t=3: (t-1)=2, so pairs. For each pair (a,b), the triples containing it are (a,b,x) for x not in {a,b}.
    # Precompute: pair_to_triples[(a,b)] = list of t-subset indices for triples containing pair (a,b)
    # Actually, we need: for a partial block P and candidate x, gain = # uncovered triples S
    # such that S = R ∪ {x} where R is a (t-1)-subset of P.
    # So for each (t-1)-subset R, we need to know which t-subsets S have S\{x} = R for some x.
    # Equivalently: for each R (sorted (t-1)-tuple), precompute list of (x, tsub_idx) where
    # tsub_idx corresponds to frozenset(R + (x,)).

    # For t=3: R is a pair (a,b). For each x not in {a,b}, triple (a,b,x) has index tsub_index[frozenset({a,b,x})].
    # Store: pair_triples[(a,b)] = [(x, idx), ...] for all x not in {a,b}

    pair_triples = {}
    for i, S in enumerate(all_tsubs_frozen):
        S_list = sorted(S)
        for j in range(t):
            # Remove S_list[j], the rest is the (t-1)-subset
            R = S_list[:j] + S_list[j+1:]
            x = S_list[j]
            key = tuple(R)
            if key not in pair_triples:
                pair_triples[key] = []
            pair_triples[key].append((x, i))

    # For fast lookup: given a frozenset P and candidate x, compute gain.
    # gain = sum over all (t-1)-subsets R of P: (number of uncovered triples R∪{x})
    # For each R, we look up pair_triples[R] and count entries (x', idx) where x'==x and coverage[idx]==0.

    # To make this faster, precompute for each pair R and each x, the list of t-subset indices.
    # pair_x_to_triples[(R, x)] = [idx, ...] where idx are t-subsets = R∪{x}
    # But for a given R and x, there's at most one triple R∪{x}. So:
    # pair_x_to_triple[(R, x)] = idx or -1

    # Actually simpler: for t=3, for each pair (a,b) and x not in {a,b},
    # there is exactly one triple {a,b,x}. So:
    pair_x_to_triple = {}
    for i, S in enumerate(all_tsubs_frozen):
        S_list = sorted(S)
        for j in range(t):
            R = tuple(S_list[:j] + S_list[j+1:])
            x = S_list[j]
            pair_x_to_triple[(R, x)] = i

    def compute_gain(P_frozen, x):
        """
        Compute the number of uncovered t-subsets that would be newly covered
        if x is added to the partial block P_frozen.
        """
        gain = 0
        P_list = sorted(P_frozen)
        for R in itertools.combinations(P_list, t - 1):
            key = (tuple(R), x)
            idx = pair_x_to_triple.get(key, -1)
            if idx >= 0 and coverage[idx] == 0:
                gain += 1
        return gain

    while True:
        if time.time() > deadline - 3:
            break

        # Find an uncovered t-subset to seed a new block
        uncovered_idx = -1
        for i in range(num_tsubs):
            if coverage[i] == 0:
                uncovered_idx = i
                break
        if uncovered_idx < 0:
            break  # All covered

        uncovered = all_tsubs_frozen[uncovered_idx]

        # Start a new block with a random point from uncovered
        block_points = set()
        start_point = rng.choice(list(uncovered))
        block_points.add(start_point)
        P_frozen = frozenset(block_points)

        # Greedily add points
        while len(block_points) < k:
            if time.time() > deadline - 3:
                break

            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_points:
                    continue
                g = compute_gain(P_frozen, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_points]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            block_points.add(best_x)
            P_frozen = frozenset(block_points)

        if len(block_points) < k:
            remaining = [x for x in range(v) if x not in block_points]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block_points) >= k:
                    break
                block_points.add(x)
            if len(block_points) < k:
                continue

        block_frozen = frozenset(block_points)
        blocks.append(block_frozen)

        # Update coverage for all t-subsets of this block
        for S in itertools.combinations(sorted(block_points), t):
            S_frozen = frozenset(S)
            idx = tsub_index[S_frozen]
            coverage[idx] += 1

    # Remove redundant blocks
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, B in enumerate(blocks):
        for S in itertools.combinations(sorted(B), t):
            S_frozen = frozenset(S)
            idx = tsub_index[S_frozen]
            tsub_to_blocks[idx].append(i)

    redundant = [False] * len(blocks)
    for i, B in enumerate(blocks):
        is_redundant = True
        for S in itertools.combinations(sorted(B), t):
            S_frozen = frozenset(S)
            idx = tsub_index[S_frozen]
            covering = tsub_to_blocks[idx]
            if len(covering) == 1 and covering[0] == i:
                is_redundant = False
                break
        if is_redundant:
            redundant[i] = True

    if any(redundant):
        blocks = [B for i, B in enumerate(blocks) if not redundant[i]]

    return [sorted(B) for B in blocks]


def _local_search(v, k, t, blocks, seed=None, time_limit=30.0):
    """
    Try to reduce block count by replacing blocks with better alternatives.
    Uses a pool of candidate blocks and tries swapping.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute t-subset indices
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_frozen = [frozenset(sub) for sub in all_tsubs]
    num_tsubs = len(all_tsubs_frozen)
    tsub_index = {S: i for i, S in enumerate(all_tsubs_frozen)}

    # Coverage counts
    coverage = [0] * num_tsubs
    for B in blocks:
        for S in itertools.combinations(B, t):
            idx = tsub_index[frozenset(S)]
            coverage[idx] += 1

    current_blocks = list(blocks)
    current_count = len(current_blocks)

    # Generate a pool of candidate blocks (random 7-subsets)
    pool_size = min(500, 20000)
    candidate_pool = set()
    for _ in range(pool_size):
        B = frozenset(rng.sample(range(v), k))
        candidate_pool.add(B)

    # Also add some structured candidates: for each uncovered triple, make a block containing it
    # (but we should already have full coverage)

    # Try local search: for each block, try replacing it with a candidate that covers
    # the same or more unique triples
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(len(current_blocks)):
            if time.time() > deadline - 2:
                break
            # Find triples uniquely covered by block i
            B_i = current_blocks[i]
            unique_triples = []
            for S in itertools.combinations(sorted(B_i), t):
                idx = tsub_index[frozenset(S)]
                if coverage[idx] == 1:
                    unique_triples.append(idx)

            if not unique_triples:
                # Block is redundant, remove it
                current_blocks.pop(i)
                for S in itertools.combinations(sorted(B_i), t):
                    idx = tsub_index[frozenset(S)]
                    coverage[idx] -= 1
                current_count -= 1
                improved = True
                break

            # Try to find a candidate block that covers all unique_triples
            # and has good overall coverage
            best_candidate = None
            best_score = -1
            for C in candidate_pool:
                # Check if C covers all unique_triples
                covers_all = True
                for idx in unique_triples:
                    S = all_tsubs_frozen[idx]
                    if not S.issubset(C):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                # Score: number of additional triples covered by C that are not covered by B_i
                score = 0
                for S in itertools.combinations(sorted(C), t):
                    idx = tsub_index[frozenset(S)]
                    if coverage[idx] == 1 and frozenset(S).issubset(B_i):
                        # This triple is uniquely covered by B_i, but C also covers it
                        # Net effect: coverage stays at 1 (B_i removed, C added)
                        pass
                    elif coverage[idx] == 0:
                        score += 10  # Bonus for covering new triples
                    elif coverage[idx] == 1:
                        score += 1  # Slight bonus for redundant coverage
                if score > best_score:
                    best_score = score
                    best_candidate = C

            if best_candidate is not None:
                # Replace block i with best_candidate
                # Update coverage: remove B_i's contributions
                for S in itertools.combinations(sorted(B_i), t):
                    idx = tsub_index[frozenset(S)]
                    coverage[idx] -= 1
                # Add C's contributions
                for S in itertools.combinations(sorted(best_candidate), t):
                    idx = tsub_index[frozenset(S)]
                    coverage[idx] += 1
                current_blocks[i] = best_candidate
                candidate_pool.discard(best_candidate)
                improved = True
                break

    return [sorted(B) for B in current_blocks]


def _verify_covering(v, k, t, blocks):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for B in blocks:
        for S in itertools.combinations(B, t):
            covered.add(S)
    all_tsubs = set(itertools.combinations(range(v), t))
    return covered == all_tsubs


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    total_deadline = start_time + 118.0  # Leave 2s margin for 120s limit

    # Try multiple random restarts
    best_blocks = None
    best_count = float('inf')

    # First, try a deterministic greedy (seed=0) with most of the time
    time_for_greedy = min(40.0, total_deadline - start_time - 20)
    blocks0 = _greedy_cover(v, k, t, seed=0, time_limit=time_for_greedy)
    if blocks0 and _verify_covering(v, k, t, blocks0):
        best_blocks = blocks0
        best_count = len(blocks0)

    # Apply local search to the best result so far
    if best_blocks and time.time() < total_deadline - 15:
        ls_time = min(20.0, total_deadline - time.time() - 10)
        improved = _local_search(v, k, t, best_blocks, seed=42, time_limit=ls_time)
        if improved and _verify_covering(v, k, t, improved):
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved

    # Try more random restarts if time permits
    seed = 1
    while time.time() < total_deadline - 12:
        remaining = total_deadline - time.time() - 5
        if remaining < 5:
            break
        # Use a fraction of the remaining time for each restart
        time_per_greedy = min(remaining * 0.6, 20.0)
        time_per_ls = min(remaining * 0.3, 10.0)
        if time_per_greedy < 3:
            break
        blocks = _greedy_cover(v, k, t, seed=seed, time_limit=time_per_greedy)
        if blocks and _verify_covering(v, k, t, blocks):
            # Apply local search
            if time.time() < total_deadline - 8 and time_per_ls > 2:
                improved = _local_search(v, k, t, blocks, seed=seed + 100, time_limit=time_per_ls)
                if improved and _verify_covering(v, k, t, improved):
                    if len(improved) < best_count:
                        best_count = len(improved)
                        best_blocks = improved
            else:
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks
        seed += 1

    if best_blocks is None:
        return []

    return best_blocks
# EVOLVE-BLOCK-END
