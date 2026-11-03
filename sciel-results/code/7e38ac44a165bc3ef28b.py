"""Optimized covering design C(v,k,t): vectorized greedy set cover."""
import time
import itertools
import numpy as np

DEADLINE = 150.0
MARGIN = 2.0  # seconds to leave before the hard limit


def _cover_bitset(combs, v, k):
    """Return an int bitset of size C(v,t) covering all t-subsets of a k-block.

    combs: generator of the C(v,t) t-subsets in lexicographic order.
    """
    from math import comb
    total = comb(v, t)
    # map each t-subset to its lex index
    # We need a fast way to compute lex rank of a t-subset of range(v).
    # Precompute factorial-based rank.
    # rank of sorted tuple c: sum over positions of C(c[i] - (i+1), t-i-1) style.
    # Standard combinadic rank:
    # rank(c) = C(c[t-1],1)+C(c[t-2],2)+...+C(c[0],t)  (for 0-indexed, c ascending)
    # Actually: rank = sum_{i} C(c[i], t-i)  where c is ascending 0-indexed.
    # Let's verify: for t=2, v=4, subset (0,1): C(1,2)+C(0,1)=0+0=0 -> index 0. (0,2): C(2,2)+C(0,1)=1 -> index1. Good.
    # (1,2): C(2,2)+C(1,1)=1+1=2. (0,3): C(3,2)+C(0,1)=3. (1,3): C(3,2)+C(1,1)=3+1=4. (2,3):C(3,2)+C(2,1)=3+2=5. Total 6=C(4,2). 
    # So rank(c)=sum_i C(c[i], t-i).
    # Precompute C(n, r) table.
    # We'll build a lookup: for each possible value 0..v-1 and each i (0..t-1), C(val, t-i).
    # Then rank = sum over i of lookup[c[i]][t-i].
    # Build table T[r][val] = C(val, r) for r in 1..t, val in 0..v-1.
    max_r = t
    # T[r][val]
    T = np.zeros((max_r + 1, v), dtype=np.int64)
    for r in range(1, max_r + 1):
        for val in range(r, v):
            T[r, val] = comb(val, r)
    # Now iterate combs, but we only need to iterate over the t-subsets OF the block.
    # But combs is the full C(v,t) generator; we don't want to iterate all. Instead iterate t-subsets of the block.
    from itertools import combinations
    bits = 0
    for sub in combinations(block, t):
        # sub is ascending tuple of length t
        rank = 0
        for i in range(t):
            rank += T[t - i, sub[i]]
        bits |= (1 << rank)
    return bits


def construct(v, k, t):
    t0 = time.time()
    from math import comb
    total = comb(v, t)
    # We need bitset operations. If total > 64 we can't use a single Python int efficiently?
    # Python int is arbitrary precision, so 1<<rank works for large rank. Bitwise OR on huge ints is O(n_words).
    # total = C(19,5)=11628. That's fine for Python big ints.
    # But doing this per block (335 blocks) each with C(8,5)=56 subsets -> 18760 rank computations + big-int ORs. Acceptable.

    # Precompute lex rank for every t-subset of range(v).
    # total up to 11628 -> store in a dict mapping tuple->rank, and rank->tuple.
    # Build via itertools.combinations.
    rank_of = {}
    subs_list = []
    for idx, sub in enumerate(itertools.combinations(range(v), t)):
        rank_of[sub] = idx
        subs_list.append(sub)
    # Also build for fast rank of a block's t-subsets: precompute T table.
    T = np.zeros((t + 1, v), dtype=np.int64)
    for r in range(1, t + 1):
        for val in range(r, v):
            T[r, val] = comb(val, r)

    def rank_fast(sub):
        # sub ascending tuple length t
        s = 0
        for i in range(t):
            s += T[t - i, sub[i]]
        return s

    # uncovered bitset: all bits set
    uncovered = (1 << total) - 1
    blocks = []
    covered_count = 0

    # Precompute, for each point x, the list of t-subsets containing x? That's C(v-1, t-1).
    # For gain computation we need: for candidate x not in block, count t-subsets of (block+{x}) that are uncovered.
    # A t-subset of block+{x} either contains x (then it's (t-1)-subset of block + x) or doesn't (t-subset of block, already covered by block? no, block not yet complete).
    # Since block currently has < k points, and we add x: new t-subsets formed = those containing x = choose t-1 from block.
    # So gain(x) = number of (t-1)-subsets S of block such that S+{x} is uncovered.
    # We can compute this by iterating C(|block|, t-1) subsets.
    # To speed: precompute for each point x a bitset of t-subsets containing x? That's big but doable: each is C(v-1,t-1) bits.
    # Actually we need intersection with uncovered. Let's precompute contain_x[x] = bitset of t-subsets that include x.
    contain_x = np.zeros(v, dtype=object)  # python ints
    for x in range(v):
        bits = 0
        for sub in subs_list:
            if x in sub:
                bits |= (1 << rank_of[sub])
        contain_x[x] = bits

    # Precompute, for a (t-1)-subset S, the bitset of t-subsets extending S by one point (i.e., S + {x} for x not in S).
    # That's the union over x of the single t-subset S+{x}. We can compute on the fly.

    while uncovered and (time.time() - t0) < DEADLINE - MARGIN:
        # pick a seed t-subset from uncovered: the lowest set bit.
        # lowest set bit index:
        # (uncovered & -uncovered).bit_length() - 1
        lowbit = uncovered & (-uncovered)
        seed_rank = lowbit.bit_length() - 1
        seed = subs_list[seed_rank]
        block = list(seed)

        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of (t-1)-subsets S of block with S+{x} uncovered
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(S + (x,)))
                    r = rank_of[sub]
                    if (uncovered >> r) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                # no candidate improves; pick any unused
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            block.append(best_x)

        block.sort()
        # compute covered bitset of this block
        bits = 0
        for sub in itertools.combinations(block, t):
            bits |= (1 << rank_of[sub])
        newly = bits & uncovered
        # count newly covered
        # popcount
        covered_count += bin(newly).count('1')
        uncovered &= ~bits
        blocks.append(block)

    return blocks
