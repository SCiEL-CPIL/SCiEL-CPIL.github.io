import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets, all k-blocks, and mappings."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, which t-sets it covers
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(frozenset(tset_indices))
    
    # For each t-set, which blocks cover it
    tset_to_blocks = defaultdict(list)
    for b_idx, ts_set in enumerate(block_tsets):
        for ts_idx in ts_set:
            tset_to_blocks[ts_idx].append(b_idx)
    
    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks

def _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, time_limit):
    """Fast greedy set cover initialization."""
    deadline = time.time() + time_limit
    uncovered = np.ones(n_tsets, dtype=bool)
    blocks_chosen = []
    
    while np.any(uncovered) and time.time() < deadline:
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-set
        rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        
        # Find best block covering this t-set
        best_block = -1
        best_gain = -1
        for b_idx in tset_to_blocks[rand_idx]:
            gain = sum(1 for ts_idx in block_tsets[b_idx] if uncovered[ts_idx])
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks_chosen.append(best_block)
        for ts_idx in block_tsets[best_block]:
            uncovered[ts_idx] = False
            
    return blocks_chosen

def _remove_redundant(blocks_idx, block_tsets, n_tsets, time_limit):
    """Remove redundant blocks from a covering."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    
    if not current:
        return current
    
    # Count coverage
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            # Check if any t-set is only covered by this block
            redundant = True
            for ts_idx in block_tsets[b_idx]:
                if covered[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                del current[i]
                for ts_idx in block_tsets[b_idx]:
                    covered[ts_idx] -= 1
                changed = True
                break
    
    return current

def _simulated_annealing(blocks_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, n_tsets, rng, time_limit):
    """
    Simulated annealing on a fixed-size set of blocks.
    State: M blocks chosen from n_blocks.
    Energy: number of uncovered t-sets (lower is better).
    Moves: replace one block with a different one.
    """
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    M = len(current)
    
    if M == 0:
        return current
    
    # Coverage count
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    
    current_uncovered = n_tsets - int(np.count_nonzero(covered > 0))
    best_uncovered = current_uncovered
    best_blocks = current[:]
    
    # Temperature schedule
    t0 = 1.0  # initial temperature (accept any move with delta <= t0)
    t_end = 0.001
    cooling = 0.9995  # per step
    
    temperature = t0
    
    while time.time() < deadline and temperature > t_end:
        # Pick a position to change
        pos = rng.randint(0, M - 1)
        old_b = current[pos]
        
        # Pick a new block (different from old)
        new_b = rng.randint(0, n_blocks - 1)
        if new_b == old_b:
            continue
        
        # Compute delta in uncovered count
        # Remove old_b
        for ts_idx in block_tsets[old_b]:
            covered[ts_idx] -= 1
        # Add new_b
        for ts_idx in block_tsets[new_b]:
            covered[ts_idx] += 1
        
        new_uncovered = n_tsets - int(np.count_nonzero(covered > 0))
        delta = new_uncovered - current_uncovered
        
        # Acceptance
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            # Accept
            current[pos] = new_b
            current_uncovered = new_uncovered
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_blocks = current[:]
        else:
            # Reject: undo
            for ts_idx in block_tsets[new_b]:
                covered[ts_idx] -= 1
            for ts_idx in block_tsets[old_b]:
                covered[ts_idx] += 1
        
        temperature *= cooling
    
    # After SA, do local deletion to shrink
    best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, min(5.0, max(0.0, deadline - time.time())))
    
    return best_blocks

def _multi_start_sa(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, time_limit):
    """
    Run multiple SA iterations with different starting points.
    """
    deadline = time.time() + time_limit
    best_blocks = None
    best_size = float('inf')
    
    # First, get a good greedy start
    greedy_start = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 5.0)
    greedy_start = _remove_redundant(greedy_start, block_tsets, n_tsets, 2.0)
    
    if best_blocks is None or len(greedy_start) < best_size:
        best_blocks = greedy_start[:]
        best_size = len(greedy_start)
    
    # Now run SA from this start
    elapsed = time.time()
    sa_budget = deadline - time.time() - 3.0
    if sa_budget > 5.0:
        sa_result = _simulated_annealing(greedy_start, all_blocks, n_blocks, block_tsets, tset_to_blocks, n_tsets, rng, sa_budget)
        sa_result = _remove_redundant(sa_result, block_tsets, n_tsets, 2.0)
        if len(sa_result) < best_size:
            best_blocks = sa_result[:]
            best_size = len(sa_result)
    
    # Try a few more greedy starts with different seeds
    for seed in range(5):
        if time.time() > deadline - 5.0:
            break
        rng2 = random.Random(seed * 1000 + 42)
        greedy2 = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng2, 3.0)
        greedy2 = _remove_redundant(greedy2, block_tsets, n_tsets, 1.0)
        if len(greedy2) < best_size:
            best_blocks = greedy2[:]
            best_size = len(greedy2)
        
        # Quick SA from this start
        remaining = deadline - time.time() - 3.0
        if remaining > 3.0:
            sa2 = _simulated_annealing(greedy2, all_blocks, n_blocks, block_tsets, tset_to_blocks, n_tsets, rng, min(remaining - 3.0, 10.0))
            sa2 = _remove_redundant(sa2, block_tsets, n_tsets, 1.0)
            if len(sa2) < best_size:
                best_blocks = sa2[:]
                best_size = len(sa2)
    
    return best_blocks

def _verify_covering(blocks_idx, block_tsets, n_tsets):
    """Verify that the blocks cover all t-sets."""
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in blocks_idx:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    return np.all(covered > 0)

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit
    
    # Precompute
    all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks = _precompute(v, k, t)
    
    rng = random.Random(42)
    
    # Run multi-start SA
    best_blocks = _multi_start_sa(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, total_budget - 5.0)
    
    # Final verification and cleanup
    if best_blocks is None:
        # Fallback
        best_blocks = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 10.0)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 5.0)
    
    best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 2.0)
    
    if not _verify_covering(best_blocks, block_tsets, n_tsets):
        # Something went wrong; rebuild
        rng_fallback = random.Random(789)
        best_blocks = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng_fallback, 10.0)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 5.0)
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result

# EVOLVE-BLOCK-END
