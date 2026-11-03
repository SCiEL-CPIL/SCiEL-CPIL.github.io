# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute 3-subset ids for v=24
_V = 24
_K = 7
_T = 3
_SUBSET_IDS = {}
_ID_TO_SUBSET = []
for _c in itertools.combinations(range(_V), _T):
    _SUBSET_IDS[_c] = len(_SUBSET_IDS)
    _ID_TO_SUBSET.append(_c)
_NSUB = len(_SUBSET_IDS)  # 2024

# Bitmask of the 3-subsets contained in a 7-block
_BLOCK_MASKS = {}
for _b in itertools.combinations(range(_V), _K):
    _m = 0
    for _c in itertools.combinations(_b, _T):
        _m |= 1 << _SUBSET_IDS[_c]
    _BLOCK_MASKS[_b] = _m

# For greedy fill: given a (t-1)-subset tuple, list of (x, bitmask) candidates
def _fill_masks(part):
    """part: tuple of (t-1) points. Returns list of (x, mask_of_3subs_with_x)."""
    out = []
    for x in range(_V):
        if x in part:
            continue
        m = 0
        for c in itertools.combinations(part, _T - 1):
            m |= 1 << _SUBSET_IDS[tuple(sorted(c + (x,)))]
        out.append((x, m))
    return out


def _greedy_cover(rng):
    """Fast greedy construction -> list of 7-tuples covering all 3-subsets."""
    covered = 0
    blocks = []
    while covered != (1 << _NSUB) - 1:
        # pick a random uncovered subset
        mask = covered
        while True:
            i = rng.randrange(_NSUB)
            if not (mask >> i) & 1:
                break
        # seed: random 3-subset containing i
        c = _ID_TO_SUBSET[i]
        part = list(c)
        bm = 0
        for cc in itertools.combinations(part, _T):
            bm |= 1 << _SUBSET_IDS[cc]
        while len(part) < _K:
            best_x, best_gain = None, -1
            for x, fm in _fill_masks(tuple(part)):
                gain = (fm & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_fm = fm
            part.append(best_x)
            bm |= best_fm
        blocks.append(tuple(sorted(part)))
        covered |= bm
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    FULL = (1 << _NSUB) - 1

    # multiple greedy restarts, keep smallest
    best = None
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        cov = _greedy_cover(rng)
        if best is None or len(cov) < len(best):
            best = cov
    blocks = list(best)
    masks = [_BLOCK_MASKS[b] for b in blocks]
    covered = 0
    for m in masks:
        covered |= m
    best_n = len(blocks)
    best_blocks = list(blocks)

    def try_remove():
        nonlocal blocks, masks, covered, best_n, best_blocks
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        if masks[i] & ~covered == 0:
            # redundant
            del blocks[i]
            del masks[i]
            covered = 0
            for m in masks:
                covered |= m
            if len(blocks) < best_n:
                best_n = len(blocks)
                best_blocks = list(blocks)
            return True
        return False

    def two_block_swap():
        nonlocal blocks, masks, covered, best_n, best_blocks
        n = len(blocks)
        if n < 2:
            return
        i, j = rng.sample(range(n), 2)
        # exposed subsets: covered only by these two blocks
        rest = 0
        for idx, m in enumerate(masks):
            if idx != i and idx != j:
                rest |= m
        exposed = (masks[i] | masks[j]) & ~rest
        # greedy rebuild up to 2 blocks covering `exposed`
        rem = exposed
        newblocks = []
        for _ in range(2):
            if rem == 0:
                break
            # seed with random uncovered-in-rem subset
            mask = rem
            while True:
                id_ = rng.randrange(_NSUB)
                if (rem >> id_) & 1:
                    break
            c = _ID_TO_SUBSET[id_]
            part = list(c)
            bm = 0
            for cc in itertools.combinations(part, _T):
                bm |= 1 << _SUBSET_IDS[cc]
            while len(part) < _K:
                best_x, best_gain = None, -1
                for x, fm in _fill_masks(tuple(part)):
                    gain = (fm & rem).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_fm = fm
                if best_x is None:
                    break
                part.append(best_x)
                bm |= best_fm
            newblocks.append(tuple(sorted(part)))
            rem &= ~bm
        # apply only if fewer new blocks than removed (2) and fully covered
        if len(newblocks) < 2 and rem == 0:
            # remove i and j (higher index first)
            hi, lo = max(i, j), min(i, j)
            del blocks[hi]
            del masks[hi]
            del blocks[lo]
            del masks[lo]
            for nb in newblocks:
                blocks.append(nb)
                masks.append(_BLOCK_MASKS[nb])
            covered = 0
            for m in masks:
                covered |= m
            if len(blocks) < best_n:
                best_n = len(blocks)
                best_blocks = list(blocks)

    while time.time() < deadline:
        if try_remove():
            continue
        two_block_swap()
        # occasional full re-scan for redundancy after swaps
        if rng.random() < 0.2:
            try_remove()

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
