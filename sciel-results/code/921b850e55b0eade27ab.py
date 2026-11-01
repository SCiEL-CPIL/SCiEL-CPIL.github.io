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

# Combinatorial Holography: assign each point a frequency in Z19
# Each 7-block gets a 64-bit fingerprint based on its point frequencies
# and a spectral signature via FFT over Z19

# Assign frequencies: point p gets frequency p mod 19
_point_freq = np.array(range(_V), dtype=np.int64)

# For each block, compute its 64-bit holographic fingerprint
# Fingerprint = hash of the multiset of pairwise frequency differences
_block_fingerprints = []
for blk in _all_blocks:
    freqs = _point_freq[list(blk)]
    # Compute all pairwise differences mod 19
    diffs = []
    for i in range(len(freqs)):
        for j in range(i+1, len(freqs)):
            diffs.append((freqs[i] - freqs[j]) % _V)
    # Create a 64-bit fingerprint from the diff distribution
    # Count occurrences of each difference 0..18
    diff_counts = np.zeros(_V, dtype=np.int64)
    for d in diffs:
        diff_counts[d] += 1
    # Pack into 64-bit integer
    fp = 0
    for d in range(_V):
        fp = (fp << 4) | (diff_counts[d] & 0xF)
    _block_fingerprints.append(fp)

# Precompute FFT-based spectral signatures for blocks
# For each block, compute the DFT of its "coverage indicator" over Z19
# We represent each 4-subset by the sum of its points mod 19
# Then a block's spectral signature is the sum of ω^(k * sum_mod) for covered quads

omega = np.exp(2j * np.pi / _V)

# For each 4-subset, compute its spectral phase (sum of points mod 19)
_quad_phase = np.zeros(_n4, dtype=np.int64)
for i, quad in enumerate(_all_4subs):
    _quad_phase[i] = sum(quad) % _V

# For each block, precompute its spectral vector (19 complex values)
# spectral[bi][k] = sum_{quad in block} ω^(k * phase(quad))
_block_spectral = np.zeros((_nb, _V), dtype=np.complex128)
for bi, blk in enumerate(_all_blocks):
    for quad in itertools.combinations(blk, _T):
        ph = _quad_phase[_subset_index[quad]]
        for k in range(_V):
            _block_spectral[bi, k] += omega ** (k * ph)


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using Combinatorial Holography.
    
    Uses spectral residual energy over Z19: at each step, compute the FFT
    of the uncovered 4-subset indicator vector, then select the block that
    maximizes spectral cancellation (reduction in spectral energy) while
    also covering the most uncovered subsets.
    """
    if v != _V or k != _K or t != _T:
        # Fallback to simple greedy for other parameters
        return _simple_greedy(v, k, t)
    
    # Start with all 4-subsets uncovered
    uncovered = (1 << _n4) - 1  # all bits set
    
    blocks_selected = []
    uncovered_count = _n4
    
    # Precompute: for each block, the list of 4-subset indices it covers
    # (as a list for faster spectral computation)
    _block_quad_indices = []
    for bi, blk in enumerate(_all_blocks):
        quads = [quad for quad in itertools.combinations(blk, _T)]
        _block_quad_indices.append([_subset_index[q] for q in quads])
    
    while uncovered_count > 0:
        # Find the least significant set bit (lowest index uncovered subset)
        lsb = uncovered & (-uncovered)
        idx = lsb.bit_length() - 1  # index of the uncovered subset
        
        # Get all blocks that contain this subset
        candidates = _subset_to_blocks[idx]
        
        # Compute spectral energy of current uncovered set
        # Build the uncovered indicator vector over Z19 phases
        # spectral_uncovered[k] = sum_{i: uncovered[i]} ω^(k * phase[i])
        # This is the DFT of the phase-binned uncovered indicator
        
        # For efficiency, maintain a running spectral vector
        # But we recompute each iteration for accuracy
        spectral_uncovered = np.zeros(_V, dtype=np.complex128)
        # Iterate over uncovered bits
        u = uncovered
        while u:
            lsb_u = u & (-u)
            i_u = lsb_u.bit_length() - 1
            ph = _quad_phase[i_u]
            for k in range(_V):
                spectral_uncovered[k] += omega ** (k * ph)
            u &= (u - 1)
        
        # For each candidate, compute:
        # 1. newly_covered count
        # 2. spectral cancellation score = real part of dot product of
        #    block's spectral signature with conjugate of spectral_uncovered
        #    (maximizing this means the block's coverage pattern aligns
        #     with the residual error, reducing spectral energy)
        
        best_block = None
        best_newly_covered = -1
        best_score = -1e18
        
        for bi in candidates:
            newly_covered = (uncovered & _block_masks[bi]).bit_count()
            
            # Spectral cancellation: <block_spectral, conj(spectral_uncovered)>
            # = sum_k block_spectral[bi,k] * conj(spectral_uncovered[k])
            spec_score = np.real(np.vdot(_block_spectral[bi], spectral_uncovered))
            
            # Combined score: prioritize coverage, use spectral as tie-breaker
            # Weight: coverage gain is primary, spectral is secondary
            score = newly_covered * 1000.0 + spec_score
            
            if score > best_score:
                best_score = score
                best_newly_covered = newly_covered
                best_block = bi
        
        # If we found a block, select it
        if best_block is not None:
            blocks_selected.append(list(_all_blocks[best_block]))
            uncovered &= ~_block_masks[best_block]
            uncovered_count = uncovered.bit_count()
        else:
            # Safety: pick any block that covers the seed subset
            bi = candidates[0] if candidates else 0
            blocks_selected.append(list(_all_blocks[bi]))
            uncovered &= ~_block_masks[bi]
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
