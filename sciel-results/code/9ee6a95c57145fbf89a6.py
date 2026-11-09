"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import comb

N = 18
NUM_BOOLS = 1 << N  # 262144
NUM_WORDS = NUM_BOOLS // 64  # 4096

# Precompute initial masks and expected masks once at module level
_INITIAL_MASKS = None
_EXPECTED_MASKS = None

def _precompute_masks(n=N):
    """Precompute initial and expected masks for verification."""
    global _INITIAL_MASKS, _EXPECTED_MASKS
    if _INITIAL_MASKS is not None:
        return
    
    num_words = (1 << n) // 64
    
    # Initialize masks: mask[i] has bit k set iff (k >> i) & 1
    _INITIAL_MASKS = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for w in range(num_words):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            _INITIAL_MASKS[i, w] = word
    
    # Expected masks: wire i should have exactly {k : popcount(k) >= n - i}
    # Precompute expected bit counts
    _EXPECTED_MASKS = np.zeros(n, dtype=np.int64)
    for i in range(n):
        count = 0
        for j in range(n - i, n + 1):
            count += comb(n, j)
        _EXPECTED_MASKS[i] = count

def _verify_fast(comps, n=N):
    """Fast verification using precomputed masks. Returns True if correct."""
    _precompute_masks(n)
    num_words = (1 << n) // 64
    
    # Copy initial masks
    masks = _INITIAL_MASKS.copy()
    
    # Apply comparators
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    
    # Check bit counts match expected using vectorized popcount
    for i in range(n):
        unpacked = np.unpackbits(masks[i].astype(np.uint8).reshape(-1, 8), axis=1)
        bit_count = unpacked.sum()
        if bit_count != _EXPECTED_MASKS[i]:
            return False
    
    # Also check monotonicity as a quick sanity check
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    
    return True

def _build_ooe_network(n):
    """Odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _build_bubble_sort_network(n):
    """Bubble sort network for n wires."""
    comps = []
    for i in range(n - 1):
        for j in range(n - 1 - i):
            comps.append((j, j + 1))
    return comps

def _build_bitonic_sort_network(n):
    """Bitonic sort network for n wires (power of 2)."""
    if n <= 1:
        return []
    if n & (n - 1) != 0:
        # Not a power of 2, use odd-even transposition
        return _build_ooe_network(n)
    
    comps = []
    def _bitonic(comps, low, k, direction):
        if k <= 1:
            return
        half = k // 2
        _bitonic(comps, low, half, 1)
        _bitonic(comps, low + half, half, 0)
        for i in range(half):
            if direction == 1:
                comps.append((low + i, low + half + i))
            else:
                comps.append((low + half + i, low + i))
    
    _bitonic(comps, 0, n, 1)
    return comps

def _comparator_redundancy_score(comps, n=N):
    """
    Score each comparator by how likely it is to be redundant.
    Higher score = more likely redundant.
    """
    scores = []
    seen_pairs = {}
    n = len(comps)
    
    for idx, (a, b) in enumerate(comps):
        score = 0.0
        # Later comparators are more likely redundant
        score += 0.3 * (idx / n)
        
        # If we've seen this pair before, it's more likely redundant
        if (a, b) in seen_pairs:
            score += 0.4
            seen_pairs[(a, b)] += 1
        else:
            seen_pairs[(a, b)] = 1
        
        # Adjacent wire comparators are more fundamental (less likely redundant)
        if b == a + 1:
            score -= 0.3
        elif b == a + 2:
            score -= 0.1
        else:
            score += 0.05 * (b - a - 1)
        
        # Wires near the ends are less critical for sorting
        min_wire = min(a, b)
        max_wire = max(a, b)
        if min_wire >= N - 3 or max_wire <= 2:
            score += 0.1
        
        scores.append(score)
    
    return scores

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8 second safety margin before 90s kill
    
    n = N
    
    # Precompute masks
    _precompute_masks(n)
    
    # Start with odd-even transposition
    comps = _build_ooe_network(n)
    
    # Verify initial network
    if not _verify_fast(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    # Phase 1: Aggressive removal with smart ordering
    improved = True
    max_removal_rounds = 500
    round_count = 0
    
    while improved and time.time() < deadline and round_count < max_removal_rounds:
        improved = False
        round_count += 1
        
        # Score comparators for removal priority
        scores = _comparator_redundancy_score(comps, n)
        
        # Try removals in order of highest score first
        order = sorted(range(len(comps)), key=lambda i: scores[i], reverse=True)
        
        for idx in order:
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                improved = True
                best = list(comps)
                best_len = len(comps)
                break  # Restart from beginning after a removal
    
    # Phase 2: Swap-based local search to escape local optima
    # Inspired by the inspiration's swap-then-remove approach
    import random
    rng = random.Random(42)
    
    for _round in range(200):
        if time.time() > deadline:
            break
        found_improvement = False
        
        # Try random swaps to find new configurations
        for _attempt in range(30):
            if time.time() > deadline:
                break
            if len(comps) < 2:
                break
            
            # Pick two random positions to swap
            i, j = rng.sample(range(len(comps)), 2)
            trial = comps[:]
            trial[i], trial[j] = trial[j], trial[i]
            
            if _verify_fast(trial, n):
                # After a valid swap, try to remove a comparator
                for k in range(len(trial)):
                    if time.time() > deadline:
                        break
                    trial2 = trial[:k] + trial[k+1:]
                    if _verify_fast(trial2, n):
                        comps = trial2
                        best = list(comps)
                        best_len = len(comps)
                        found_improvement = True
                        break
                if found_improvement:
                    break
        if not found_improvement:
            # Try a few more random removals
            for _attempt in range(20):
                if time.time() > deadline:
                    break
                idx = rng.randint(0, len(comps) - 1)
                trial = comps[:idx] + comps[idx+1:]
                if _verify_fast(trial, n):
                    comps = trial
                    best = list(comps)
                    best_len = len(comps)
                    found_improvement = True
                    break
            if not found_improvement:
                break
    
    # Phase 3: Try replacing pairs of comparators with single comparators
    for _round in range(50):
        if time.time() > deadline:
            break
        found_replacement = False
        for i in range(len(comps)):
            if time.time() > deadline:
                break
            for j in range(i + 1, min(i + 5, len(comps))):
                if time.time() > deadline:
                    break
                a1, b1 = comps[i]
                a2, b2 = comps[j]
                
                # Only try if they share a wire or are close
                if not (b1 == a2 or b2 == a1 or (abs(a1 - a2) <= 2 and abs(b1 - b2) <= 2)):
                    continue
                
                # Try various single comparators that might replace both
                candidates = []
                if b1 == a2:  # Chain: (a1,b1) then (b1,b2)
                    candidates.append((a1, b2))
                if b2 == a1:  # Chain: (a2,b2) then (a1,b1) 
                    candidates.append((a2, b1))
                if a1 == a2:  # Same source
                    candidates.append((a1, max(b1, b2)))
                if b1 == b2:  # Same target
                    candidates.append((min(a1, a2), b1))
                
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    if cand[0] >= cand[1]:
                        continue
                    # Build trial: remove i and j, add cand at position i
                    trial = comps[:i] + [cand] + comps[i+1:j] + comps[j+1:]
                    if _verify_fast(trial, n):
                        comps = trial
                        best = list(comps)
                        best_len = len(comps)
                        found_replacement = True
                        break
                if found_replacement:
                    break
            if found_replacement:
                break
        if not found_replacement:
            break
    
    # Phase 4: Final cleanup - try more aggressive removals with different orderings
    for _round in range(100):
        if time.time() > deadline:
            break
        found_removal = False
        
        # Try removals in reverse order (earlier comparators first)
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                best = list(comps)
                best_len = len(comps)
                found_removal = True
                break
        
        if not found_removal:
            # Try removals in middle positions first
            mid = len(comps) // 2
            for offset in range(1, len(comps) // 2 + 1):
                if time.time() > deadline:
                    break
                for idx in [mid - offset, mid + offset]:
                    if 0 <= idx < len(comps):
                        trial = comps[:idx] + comps[idx+1:]
                        if _verify_fast(trial, n):
                            comps = trial
                            best = list(comps)
                            best_len = len(comps)
                            found_removal = True
                            break
                if found_removal:
                    break
            if not found_removal:
                break
    
    return best
# EVOLVE-BLOCK-END
