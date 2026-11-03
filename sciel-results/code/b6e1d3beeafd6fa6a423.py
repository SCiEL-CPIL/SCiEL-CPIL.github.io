# EVOLVE-BLOCK-START
import time
import itertools
import heapq
from collections import defaultdict

_TIME_BUDGET = 110.0  # safety margin under the 120 s kill


def _build_universe(v, t):
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    return list(itertools.combinations(sorted(block), t))


def _build_counts(universe, blocks):
    """Return (cover_counts, block_to_subs) for the given blocks."""
    cover_counts = defaultdict(int)
    block_to_subs = []
    for blk in blocks:
        subs = _block_covers(blk, t)
        block_to_subs.append(subs)
        for s in subs:
            cover_counts[s] += 1
    return cover_counts, block_to_subs


def _uncovered_list(universe, cover_counts):
    return [s for s in universe if cover_counts.get(s, 0) == 0]


def _greedy_construct(v, k, t, universe, time_limit):
    """Fast heap-based greedy set cover. Returns (blocks, cover_counts, block_to_subs)."""
    cover_counts = defaultdict(int)
    block_to_subs = []
    blocks = []
    uncovered = set(universe)

    # Candidate blocks: all k-subsets. Score = number of uncovered t-subsets it covers.
    # We precompute for each candidate the list of t-subsets it covers.
    candidate_to_subs = {}
    for cand in itertools.combinations(range(v), k):
        candidate_to_subs[cand] = _block_covers(cand, t)

    # Heap entries: (-score, tie, cand)
    tie = 0
    heap = []
    for cand, subs in candidate_to_subs.items():
        sc = sum(1 for s in subs if s in uncovered)
        if sc > 0:
            tie += 1
            heapq.heappush(heap, (-sc, tie, cand))

    while uncovered and heap and time.time() < time_limit:
        sc, _, cand = heapq.heappop(heap)
        if sc == 0:
            break
        # Verify still beneficial
        if not any(s in uncovered for s in candidate_to_subs[cand]):
            continue
        blocks.append(list(cand))
        subs = candidate_to_subs[cand]
        block_to_subs.append(subs)
        for s in subs:
            if s in uncovered:
                uncovered.discard(s)
                cover_counts[s] += 1
            else:
                cover_counts[s] += 1
        if not uncovered:
            break

        # Recompute scores for candidates that intersect uncovered
        for cand2, subs2 in candidate_to_subs.items():
            sc2 = sum(1 for s in subs2 if s in uncovered)
            if sc2 > 0:
                tie += 1
                heapq.heappush(heap, (-sc2, tie, cand2))

    return blocks, cover_counts, block_to_subs


def _local_search(v, k, t, universe, blocks, cover_counts, block_to_subs, time_limit):
    """1-block swap local search with keep-if-better rule."""
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() > time_limit:
                break
            # Remove block i
            subs_i = block_to_subs[i]
            for s in subs_i:
                cover_counts[s] -= 1
            uncovered_after = [s for s in subs_i if cover_counts[s] == 0]
            if not uncovered_after:
                # Removing i doesn't break coverage; keep it removed (improvement)
                blocks.pop(i)
                block_to_subs.pop(i)
                improved = True
                break
            # Find best replacement block covering max of uncovered_after
            best_cand = None
            best_gain = 0
            for cand in itertools.combinations(range(v), k):
                if cand == tuple(blocks[i]):
                    continue
                subs_c = _block_covers(cand, t)
                gain = sum(1 for s in subs_c if s in set(uncovered_after))
                # Also count extra coverage of other uncovered (none, by construction)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            # Restore block i
            for s in subs_i:
                cover_counts[s] += 1
            if best_cand is not None and best_gain >= len(uncovered_after):
                # Replace
                for s in subs_i:
                    cover_counts[s] -= 1
                blocks[i] = list(best_cand)
                subs_c = _block_covers(best_cand, t)
                block_to_subs[i] = subs_c
                for s in subs_c:
                    cover_counts[s] += 1
                improved = True
                break
    return blocks, cover_counts, block_to_subs


def _greedy_with_replacement(v, k, t, universe, time_limit):
    """Greedy that, when stuck, tries to replace a block to cover remaining uncovered."""
    blocks, cover_counts, block_to_subs = _greedy_construct(v, k, t, universe, time_limit)
    while True:
        uncovered = _uncovered_list(universe, cover_counts)
        if not uncovered:
            break
        if time.time() > time_limit:
            break
        # Try to find a single block covering the most uncovered
        best_cand = None
        best_gain = 0
        unc_set = set(uncovered)
        for cand in itertools.combinations(range(v), k):
            subs_c = _block_covers(cand, t)
            gain = sum(1 for s in subs_c if s in unc_set)
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand is None or best_gain == 0:
            break
        blocks.append(list(best_cand))
        subs_c = _block_covers(best_cand, t)
        block_to_subs.append(subs_c)
        for s in subs_c:
            cover_counts[s] += 1
    return blocks, cover_counts, block_to_subs


def construct(v, k, t):
    start = time.time()
    time_limit = start + _TIME_BUDGET
    universe = _build_universe(v, t)

    blocks, cover_counts, block_to_subs = _greedy_with_replacement(v, k, t, universe, time_limit)

    # Local search to reduce block count
    if time.time() < time_limit:
        blocks, cover_counts, block_to_subs = _local_search(
            v, k, t, universe, blocks, cover_counts, block_to_subs, time_limit
        )

    # Verify coverage; if incomplete, patch greedily
    if time.time() < time_limit:
        uncovered = _uncovered_list(universe, cover_counts)
        while uncovered and time.time() < time_limit:
            unc_set = set(uncovered)
            best_cand = None
            best_gain = 0
            for cand in itertools.combinations(range(v), k):
                subs_c = _block_covers(cand, t)
                gain = sum(1 for s in subs_c if s in unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            if best_cand is None or best_gain == 0:
                break
            blocks.append(list(best_cand))
            subs_c = _block_covers(best_cand, t)
            block_to_subs.append(subs_c)
            for s in subs_c:
                cover_counts[s] += 1
            uncovered = [s for s in uncovered if cover_counts[s] == 0]

    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(key))
    return unique_blocks
# EVOLVE-BLOCK-END
