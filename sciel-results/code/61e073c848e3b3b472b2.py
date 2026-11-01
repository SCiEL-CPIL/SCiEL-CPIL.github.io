"""Stochastic local search for covering design C(24,7,3)."""
import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def _precompute_triples(v, t):
    """Precompute all t-subsets and map each to an index."""
    triples = list(itertools.combinations(range(v), t))
    triple_idx = {ts: i for i, ts in enumerate(triples)}
    return triples, triple_idx, len(triples)

def _block_triple_indices(block, t, v):
    """Return list of triple indices covered by a block."""
    return [triple_idx[ts] for ts in itertools.combinations(block, t)]

def _build_coverage_counts(blocks, triples, t, v):
    """Build coverage count array: count[i] = number of blocks covering triple i."""
    n = len(triples)
    counts = np.zeros(n, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = triple_idx[ts]
            counts[idx] += 1
    return counts

def _count_uncovered(counts):
    """Number of triples with count == 0."""
    return int(np.sum(counts == 0))

def _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=5.0):
    """Fast greedy to get a starting population."""
    deadline = time.time() + time_limit
    uncovered = set(range(len(triples)))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick random uncovered triple
        ts_idx = rng.choice(list(uncovered))
        ts = triples[ts_idx]
        block = list(ts)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if triple_idx[tset] in uncovered:
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
        blocks.append(tuple(block))
        
        covered = set(triple_idx[ts] for ts in itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]

def _random_block(v, k, rng):
    """Generate a random k-subset."""
    return sorted(rng.sample(range(v), k))

def _sl_search(v, k, t, triples, triple_idx, initial_blocks, rng, time_limit):
    """
    Stochastic local search with simulated annealing.
    State: list of blocks (fixed size after init).
    Move: replace one random block with a new random block.
    Energy: number of uncovered triples.
    """
    deadline = time.time() + time_limit
    n_triples = len(triples)
    
    # Start from initial blocks
    blocks = [list(b) for b in initial_blocks]
    n_blocks = len(blocks)
    
    # Build coverage counts
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = triple_idx[ts]
            counts[idx] += 1
    
    current_energy = _count_uncovered(counts)
    best_energy = current_energy
    best_blocks = [list(b) for b in blocks]
    
    # Annealing parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute triple indices for all possible blocks? No, too many.
    # Instead, compute on the fly.
    
    step = 0
    while time.time() < deadline and T > min_T:
        # Choose a random block index to replace
        idx = rng.randint(0, n_blocks - 1)
        old_block = blocks[idx]
        
        # Generate new random block
        new_block = _random_block(v, k, rng)
        
        # Compute energy delta
        # Remove old block's contribution
        old_contrib = np.zeros(n_triples, dtype=np.int32)
        for ts in itertools.combinations(old_block, t):
            old_contrib[triple_idx[ts]] += 1
        
        new_contrib = np.zeros(n_triples, dtype=np.int32)
        for ts in itertools.combinations(new_block, t):
            new_contrib[triple_idx[ts]] += 1
        
        # Delta in uncovered count
        # A triple becomes uncovered if count goes from 1 to 0
        # A triple becomes covered if count goes from 0 to 1
        delta = 0
        for i in range(n_triples):
            old_c = counts[i] - old_contrib[i]
            new_c = old_c + new_contrib[i]
            if old_c == 0 and new_c > 0:
                delta -= 1  # newly covered
            elif old_c > 0 and new_c == 0:
                delta += 1  # newly uncovered
        
        # Acceptance
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            # Apply move
            counts -= old_contrib
            counts += new_contrib
            blocks[idx] = new_block
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = [list(b) for b in blocks]
        
        T *= cooling
        step += 1
    
    return best_blocks, best_energy

def _multi_start_sl(v, k, t, triples, triple_idx, rng, time_limit):
    """Run multiple SL instances with different initializations."""
    deadline = time.time() + time_limit
    best_blocks = None
    best_energy = float('inf')
    
    # Number of restarts based on time
    # Each SL run takes ~10-20s, so do 2-3 restarts
    num_restarts = 3
    per_restart = (time_limit - 5.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        # Greedy init
        init_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, time_limit=min(5.0, per_restart * 0.3))
        
        # SL refinement
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        
        sl_blocks, sl_energy = _sl_search(v, k, t, triples, triple_idx, init_blocks, rng, remaining)
        
        if sl_energy < best_energy:
            best_energy = sl_energy
            best_blocks = sl_blocks
    
    return best_blocks, best_energy

def _remove_redundant(blocks, triples, triple_idx, v, k, t, rng, time_limit):
    """Remove redundant blocks to reduce population size."""
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    while time.time() < deadline:
        n = len(blocks)
        if n == 0:
            break
        
        # Build coverage counts
        n_triples = len(triples)
        counts = np.zeros(n_triples, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[triple_idx[ts]] += 1
        
        # Find redundant block
        redundant_idx = -1
        for i in range(n):
            b = blocks[i]
            for ts in itertools.combinations(b, t):
                idx = triple_idx[ts]
                if counts[idx] == 1:
                    break
            else:
                redundant_idx = i
                break
        
        if redundant_idx == -1:
            break
        
        # Remove it
        blocks.pop(redundant_idx)
    
    return blocks

def _is_valid_covering(blocks, v, k, t, triples, triple_idx):
    """Check validity."""
    n_triples = len(triples)
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[triple_idx[ts]] += 1
    return int(np.sum(counts == 0)) == 0

def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # Safety margin for 150s kill
    
    # Precompute
    triples, triple_idx, n_triples = _precompute_triples(v, t)
    
    rng = random.Random(42)
    
    # Phase 1: Multi-start SL
    best_blocks, best_energy = _multi_start_sl(v, k, t, triples, triple_idx, rng, total_budget * 0.75)
    
    if best_blocks is None:
        # Fallback: pure greedy
        best_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, 10.0)
        best_energy = _count_uncovered(_build_coverage_counts(best_blocks, triples, t, v))
    
    # Phase 2: Remove redundant blocks
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 2.0:
        best_blocks = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, rng, remaining)
    
    # Phase 3: Final SL polish if time remains
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 5.0:
        polish_blocks, polish_energy = _sl_search(v, k, t, triples, triple_idx, best_blocks, rng, remaining)
        if polish_energy < best_energy:
            best_blocks = polish_blocks
            best_energy = polish_energy
        # Remove redundant again
        best_blocks = _remove_redundant(best_blocks, triples, triple_idx, v, k, t, rng, min(5.0, total_budget - (time.time() - start_time) - 5.0))
    
    # Ensure validity
    if not _is_valid_covering(best_blocks, v, k, t, triples, triple_idx):
        # Fallback to greedy
        best_blocks = _greedy_init(v, k, t, triples, triple_idx, rng, 10.0)
    
    return best_blocks
# EVOLVE-BLOCK-END
