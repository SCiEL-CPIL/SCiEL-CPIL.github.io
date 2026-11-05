"""
Strategy: Local Search with Insertion (1-move) + Aggressive Removal.
The Zero-One Principle allows bit-parallel verification of all 65,536 binary inputs
using numpy uint64 arrays. The current program only removes comparators from the
120-comparator Odd-Even Transposition baseline. This often gets stuck in local minima
where no single removal is possible, but a replacement (remove one, insert another)
could reduce size or improve structure.

This new strategy:
1. Uses the same bit-parallel verification (fast, ~1000s of checks/sec).
2. Alternates between:
   a. Greedy removal passes (backward and forward).
   b. Randomized 1-move replacements: pick a random comparator to remove,
      then try inserting a random new comparator at a random position.
      Accept if the new network is valid AND has the same or fewer comparators
      (since we remove one and add one, size stays same, but structure changes).
      Actually, to reduce size, we should only accept if we can then remove more.
      Better: Accept any valid replacement (size unchanged) to escape local minima,
      then run removal passes.
3. Also tries random insertions to see if they enable further removals (unlikely to
   help reduce size directly, but might change structure).

Key insight: The search space of valid sorting networks is large. Pure removal
from a fixed start is a "downhill" walk. Adding random valid moves (replacements)
allows "sideways" or even "uphill" (in terms of size, but here size is fixed in a
replacement) moves that can lead to new regions where removals are possible.

To strictly reduce size, we can:
- Do a replacement (size N -> N).
- Then immediately try to remove a comparator from the new network.
- If a removal is possible, accept the whole sequence (net size N-1).
- If not, we can still keep the replacement if we think it helps future removals,
  but to be safe and simple, we'll only accept replacements that lead to a
  subsequent removal OR we accept them with a probability and track the best.

Simpler and more effective:
- Maintain a current network.
- Loop:
  1. Try to remove any comparator (greedy pass). If success, continue.
  2. If no removal possible, try a random 1-move replacement:
     - Pick random index i to remove.
     - Pick random new comparator (a,b) and random position j to insert.
     - Check if the new network is valid.
     - If valid, accept it (size unchanged).
     - Then immediately try removal passes again.
  3. If stuck for a long time, restart with a slightly perturbed network or
     just stop.

This is a standard local search with a "sideways" move (replacement) to escape
local minima in the removal landscape.
"""

import numpy as np
import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
CHUNKS = NUM_INPUTS // 64  # 1024 chunks of 64 inputs each

_INIT_STATES = None

def _init_states():
    states = np.zeros((N, CHUNKS), dtype=np.uint64)
    for w in range(N):
        for c in range(CHUNKS):
            mask = 0
            for k in range(64):
                v = c * 64 + k
                if (v >> w) & 1:
                    mask |= (1 << k)
            states[w, c] = mask
    return states

def _get_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_states()
    return _INIT_STATES

def _verify_network_bit_parallel(comparators):
    """
    Verifies if the network sorts all 65536 binary inputs.
    Returns True if sorted, False otherwise.
    """
    states = _get_states()
    curr = states.copy()
    
    for i, j in comparators:
        min_val = curr[i] & curr[j]
        max_val = curr[i] | curr[j]
        curr[i] = min_val
        curr[j] = max_val
    
    for i in range(N - 1):
        if np.any(curr[i] & ~curr[i+1]):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with the 120-comparator Odd-Even Transposition network
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify baseline
    if not _verify_network_bit_parallel(network):
        return network

    best_network = network
    best_len = len(network)
    
    # Pre-compute all possible comparators for random insertion
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Local search loop
    # We alternate between removal passes and random replacements.
    
    no_improvement_count = 0
    max_no_improvement = 500  # If stuck for 500 replacements, try a bigger perturbation
    
    while time.time() < deadline:
        # 1. Greedy Removal Pass (Backward)
        progress = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify_network_bit_parallel(candidate):
                network = candidate
                best_network = network
                best_len = len(network)
                progress = True
                no_improvement_count = 0
                if best_len <= 60:
                    return best_network
        
        if progress:
            continue
        
        # 2. No removal possible. Try a random replacement.
        # Pick a random comparator to remove.
        if len(network) == 0:
            break
            
        remove_idx = random.randint(0, len(network) - 1)
        
        # Pick a random new comparator and random insertion position.
        new_comp = random.choice(all_comparators)
        insert_idx = random.randint(0, len(network))
        
        # Construct candidate: remove at remove_idx, insert at insert_idx
        # Note: if insert_idx > remove_idx, the insert index shifts by -1
        candidate = network[:]
        candidate.pop(remove_idx)
        # Adjust insert_idx if needed
        if insert_idx > remove_idx:
            insert_idx -= 1
        candidate.insert(insert_idx, new_comp)
        
        # Verify candidate
        if _verify_network_bit_parallel(candidate):
            # Accept the replacement (size unchanged)
            network = candidate
            no_improvement_count += 1
            
            # Immediately try to remove again from this new structure
            # This is done in the next loop iteration's step 1.
            
            # If we've been stuck for a long time, try a more aggressive perturbation:
            # Remove 2 random comparators and insert 2 random ones?
            # Or just restart from best_network with a different seed?
            if no_improvement_count > max_no_improvement:
                # Restart from best_network
                network = best_network[:]
                no_improvement_count = 0
                max_no_improvement = int(max_no_improvement * 1.5)
                if max_no_improvement > 5000:
                    break
        else:
            # Rejection: do nothing, just count as a step
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart from best_network
                network = best_network[:]
                no_improvement_count = 0
                max_no_improvement = int(max_no_improvement * 1.5)
                if max_no_improvement > 5000:
                    break

    return best_network

if __name__ == "__main__":
    net = construct()
    print(f"Construct returned {len(net)} comparators.")
    print(f"Valid: {_verify_network_bit_parallel(net)}")
