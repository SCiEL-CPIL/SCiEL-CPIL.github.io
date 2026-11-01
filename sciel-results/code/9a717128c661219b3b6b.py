"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Greedy set cover with restarts and randomization for covering design C(v,k,t).
    Uses bitsets for efficiency. Multiple restarts with random ordering to escape local optima.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset
    # ksub_mask[j] is an integer where bit i is 1 if k-subset j covers t-subset i
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # Convert to numpy array for faster operations if possible, but Python int bit ops are very fast.
    # We'll stick to Python ints for masks as they are efficient for popcount (bit_count).
    
    best_blocks = None
    best_size = float('inf')
    
    # Number of restarts: dynamic based on time
    # First run is deterministic greedy. Subsequent runs use randomized greedy.
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Deterministic greedy
            order = list(range(n_ksubs))
        else:
            # Randomized greedy: shuffle candidate order
            order = list(range(n_ksubs))
            random.shuffle(order)
            
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline:
                break
                
            best_j = -1
            best_gain = 0
            
            for j in order:
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: random or first found? 
                    # In randomized restarts, the order is already shuffled, so first found is effectively random among max gain.
                    # But we can add a small random perturbation to break ties consistently within the run?
                    # No, simple max is fine. The shuffle handles diversity.
                    pass

            if best_j == -1 or best_gain == 0:
                # Stuck or no progress
                break

            block = list(all_ksubs[best_j])
            blocks.append(block)
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain
            
            # Early termination if we already have a better result than best known
            if len(blocks) > best_size:
                break

        if remaining == 0:
            if len(blocks) < best_size:
                best_size = len(blocks)
                best_blocks = blocks
                # Optional: if we hit the theoretical lower bound or a known good value, we could stop
                # Lower bound for C(19,7,4) is 131. If we hit 131, stop.
                if best_size <= 131:
                    break
        else:
            # If we didn't finish covering, this run is invalid.
            # But we keep looking for a valid one.
            pass

    # If no valid covering was found (unlikely given time), return the best valid one found
    # Or if the deterministic run finished, it should be valid.
    if best_blocks is None:
        # Fallback: run a final deterministic greedy if we have time, or return empty?
        # The problem says invalid scores 0. We must return a valid covering.
        # If we never found a valid one, it means we timed out in the middle of every run.
        # This is extremely unlikely for v=19.
        # Let's do one final deterministic run with no time limit check inside, just to be safe?
        # No, that might timeout. 
        # Better: if best_blocks is None, we failed. But the first run is usually very fast.
        # Let's assume best_blocks is not None.
        # If it is None, we return an empty list, which will be invalid (score 0).
        # But we can try to force a completion if we have time left.
        if time.time() < deadline - 5:
             # Try one more time with a tighter deadline? No, just return empty to avoid timeout.
             # Actually, if the first run didn't finish, something is wrong.
             # Let's just return whatever we have if it's valid? No, we don't track validity of partial.
             # We only update best_blocks if remaining == 0.
             # So if best_blocks is None, we have no valid covering.
             # Return empty list.
             return []

    return best_blocks
# EVOLVE-BLOCK-END
