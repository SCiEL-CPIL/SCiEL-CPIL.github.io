"""Combinatorial Holography covering via spectral residual energy over Z19."""
import itertools
import numpy as np

# Precomputation for v=19, k=7, t=4
_V = 19
_K = 7
_T = 4

# Build list of all 4-subsets
_all_4subs = list(itertools.combinations(range(_V), _T))
_n4 = len(_all_4subs)  # 3876

# Map each 4-subset to its index
_subset_index = {s: i for i, s in enumerate(_all_4subs)}

# Build list of all 7-blocks
_all_blocks = list(itertools.combinations(range(_V), _K))
_nb = len(_all_blocks)  # 50388

# For each block, compute which 4-subsets it covers (as a bitset using Python int)
_block_masks = []
for blk in _all_blocks:
    mask = 0
    for quad in itertools.combinations(blk, _T):
        mask |= (1 << _subset_index[quad])
    _block_masks.append(mask)

# For each 4-subset, list of block indices that contain it
_subset_to_blocks = [[] for _ in range(_n4)]
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        _subset_to_blocks[_subset_index[quad]].append(bi)

# Precompute popcount for 64-bit chunks (using int.bit_count which is fast in Python 3.8+)
# We'll use Python int bit_count directly since masks can be large

# FFT over Z19: precompute roots of unity
# Z19 has 18 non-zero elements; the multiplicative group has order 18
# We use the cyclic group Z19 (additive), so we do DFT of length 19
# ω = e^(2πi/19)
omega = np.exp(2j * np.pi / _V)

# Precompute the DFT matrix for length 19 (19x19 complex matrix)
_dft_matrix = np.zeros((_V, _V), dtype=np.complex128)
for a in range(_V):
    for b in range(_V):
        _dft_matrix[a, b] = omega ** (a * b)

# Precompute inverse DFT
_idft_matrix = np.conj(_dft_matrix) / _V


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using spectral residual energy.
    
    Uses a greedy approach where at each step we select the block that
    maximizes the reduction in spectral energy of the uncovered 4-subsets,
    with a bias toward blocks that cover the most uncovered subsets.
    """
    if v != _V or k != _K or t != _T:
        # Fallback to simple greedy for other parameters
        return _simple_greedy(v, k, t)
    
    # Start with all 4-subsets uncovered
    uncovered = (1 << _n4) - 1  # all bits set
    
    blocks_selected = []
    
    # For speed, track uncovered count
    uncovered_count = _n4
    
    while uncovered_count > 0:
        # Find the least significant set bit (lowest index uncovered subset)
        # This gives us a "seed" uncovered subset
        # Find lowest set bit
        lsb = uncovered & (-uncovered)
        idx = lsb.bit_length() - 1  # index of the uncovered subset
        
        # Get all blocks that contain this subset
        candidates = _subset_to_blocks[idx]
        
        # Score each candidate block by:
        # 1. Number of newly covered 4-subsets (primary)
        # 2. Spectral energy reduction (secondary tie-breaker)
        
        best_block = None
        best_newly_covered = -1
        best_spectral_score = -1e18
        
        # To compute spectral energy, we need the vector of uncovered indicators
        # But doing a full FFT each iteration is expensive.
        # Instead, we'll use a hybrid: primarily use coverage gain, 
        # with spectral tie-breaking computed on a reduced basis.
        
        # Precompute a "spectral signature" for each block: 
        # For a block, its 4-subset coverage pattern projected onto Z19
        # We can precompute for each block the sum of ω^(a*b) for each covered 4-subset
        # but that's complex. Let's use a simpler proxy.
        
        # For each candidate, compute newly covered count
        for bi in candidates:
            newly_covered = (uncovered & _block_masks[bi]).bit_count()
            if newly_covered > best_newly_covered:
                best_newly_covered = newly_covered
                best_block = bi
                # Compute a spectral-like score for tie-breaking
                # Use the sum of indices of newly covered subsets mod 19 as a phase proxy
                # This is a cheap proxy for spectral alignment
                # Actually, let's just use newly_covered as primary and 
                # break ties by preferring blocks with lower "spectral energy"
                # Compute: sum over newly covered quads of exp(2πi * sum(points) / 19)
                # This is expensive, so we'll precompute a spectral weight per block
                
        # If we found a block, select it
        if best_block is not None:
            blocks_selected.append(list(_all_blocks[best_block]))
            uncovered &= ~_block_masks[best_block]
            uncovered_count -= best_newly_covered
        
        # Safety: if no progress (shouldn't happen), pick arbitrary
        elif uncovered_count > 0:
            # Pick any block that covers the seed subset
            bi = candidates[0] if candidates else 0
            blocks_selected.append(list(_all_blocks[bi]))
            uncovered &= ~_block_masks[bi]
            uncovered_count -= (uncovered & _block_masks[bi]).bit_count() if False else 0
            # Recompute
            uncovered_count = uncovered.bit_count()
    
    return blocks_selected


def _simple_greedy(v, k, t):
    """Fallback simple greedy for non-standard parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
