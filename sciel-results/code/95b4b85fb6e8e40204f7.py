import time
import random
import numpy as np
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

# Pre-compute verification data using NumPy for speed
# We represent the state of the network for all 131072 inputs as a matrix of shape (TOTAL_MASKS, N)
# or we can use the big integer trick but vectorized.
# Actually, the big integer trick is extremely fast in Python if we avoid slicing.
# Let's stick to the big integer trick but optimize the loop.
# The bottleneck in the current code is `new_net = current_net[:idx] + ...` and `_verify_network`.
# `_verify_network` iterates through the list.
# To speed up verification, we can pre-compute the "effect" of each comparator?
# No, the effect depends on the current state.
# However, we can verify a network by simulating it.
# A faster verification method:
# Instead of simulating the whole network every time, we can use a "incremental" approach?
# No, SA requires full verification for validity.

# Let's optimize the big integer verification.
# The current implementation:
# S = wi & ~wj
# wires[i] = wi ^ S
# wires[j] = wj ^ S
# This is O(1) per comparator.
# The loop is O(L) where L is length.
# The list slicing in SA is O(L).
# We can avoid list slicing by using a "dirty" flag or just copying the list (which is O(L) anyway).
# Copying a list of 100 tuples is fast.
# The main gain comes from running multiple islands in parallel? 
# Python GIL prevents true parallelism with threads.
# We can use multiprocessing? The overhead of pickling 100-tuple lists and big ints might be high, 
# but 60s is a long time.
# Alternatively, we can just run a much more aggressive local search with a better move set.
# The "4-Opt" move: swap (i,j) at pos p with (k,l) at pos q? 
# Or better: Insert a comparator at p, and delete one at q?
# The current SA only does Delete, Swap, Insert.
# "Add and Remove" (AR) move: Add a random comparator at a random position, and remove a random existing comparator.
# This allows changing the size by 0, but changing the structure significantly.
# Also, "Block Move": Move a block of comparators.

# Let's implement a robust Island Model with Multiprocessing.
# Each worker gets a fraction of the time.
# We will use 4 workers.

import multiprocessing as mp
import os

# Global variables for workers
_INIT_WIRES = None
_EXPECTED_WIRES = None

def _init_worker():
    global _INIT_WIRES, _EXPECTED_WIRES
    if _INIT_WIRES is not None:
        return
    
    _INIT_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INIT_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        S = wi & ~wj
        if S: # Optimization: only update if change
            wires[i] = wi ^ S
            wires[j] = wj ^ S
            
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_batcher_odd_merge(n):
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    net.extend(_build_batcher_odd_merge(n1))
    
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    for i in range(n1):
        net.append((i, i + n2))
        
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            net.append((2*i, 2*j))
            
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            net.append((2*i+1, 2*j+1))
            
    return net

def _island_search(initial_net, deadline, seed):
    """
    Runs a local search on a single island.
    Uses a population of 1 to avoid complexity, but with aggressive moves.
    Moves:
    1. Delete (always accept if valid)
    2. Insert (accept with prob based on SA)
    3. Swap (accept if valid, helps reordering)
    4. Add-Remove (Add one, Remove one, size constant, structure change)
    """
    current_net = initial_net[:]
    current_size = len(current_net)
    best_net = current_net[:]
    best_size = current_size
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    rng = random.Random(seed)
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        
        move_type = rng.random()
        
        if move_type < 0.5 and current_size > 0:
            # Delete
            idx = rng.randint(0, current_size - 1)
            # Optimization: avoid full list copy if possible? 
            # List copy is fast enough for ~100 items.
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(new_net):
                current_net = new_net
                current_size -= 1
                if current_size < best_size:
                    best_net = current_net[:]
                    best_size = current_size
        elif move_type < 0.7 and current_size >= 2:
            # Swap
            i, j = rng.sample(range(current_size), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if _verify_network(new_net):
                current_net = new_net
        elif move_type < 0.9:
            # Add-Remove (Size constant)
            if current_size > 0:
                # Pick a position to insert
                ins_idx = rng.randint(0, current_size)
                # Pick a comparator to insert
                comp = rng.choice(all_comps)
                # Pick a comparator to remove (from the new list, so indices shift)
                # Simulate insertion first to find valid removal index?
                # Or just pick a random index in current_net to remove, and insert elsewhere.
                rem_idx = rng.randint(0, current_size - 1)
                
                # Construct new net
                # If ins_idx > rem_idx, the inserted comp is at ins_idx, removed was at rem_idx
                # If ins_idx <= rem_idx, the removed comp shifts to rem_idx-1?
                # Easier: create a copy, remove, then insert.
                new_net = current_net[:]
                new_net.pop(rem_idx)
                # Adjust ins_idx if necessary? 
                # If we pop from rem_idx, indices > rem_idx shift left.
                # We want to insert at a position in the "new" list.
                # Let's just pick a random position in the new list.
                new_ins_idx = rng.randint(0, len(new_net))
                new_net.insert(new_ins_idx, comp)
                
                if _verify_network(new_net):
                    # Size is same. Accept to explore.
                    current_net = new_net
        else:
            # Insert (Size +1)
            if current_size < 4096:
                comp = rng.choice(all_comps)
                idx = rng.randint(0, current_size)
                new_net = current_net[:]
                new_net.insert(idx, comp)
                
                if _verify_network(new_net):
                    # Size increased. Accept with prob exp(-1/T)
                    if rng.random() < (2.718 ** (-1.0 / T)):
                        current_net = new_net
                        current_size += 1
                    # If not accepted, discard
                    
    return best_net

def _worker_process(args):
    initial_net, deadline, seed = args
    _init_worker()
    best_net = _island_search(initial_net, deadline, seed)
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Build initial network
    # Use Batcher Odd-Merge as it's smaller (105) than Odd-Even (136)
    best_net = _build_batcher_odd_merge(N)
    
    # Verify initial
    _init_worker() # Init globals in main process too for safety
    if not _verify_network(best_net):
        # Fallback to Odd-Even
        net = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                net.append((i, i + 1))
        best_net = net
    
    # Run a quick greedy deletion in main process to get a good starting point
    improved = True
    while improved and time.time() < deadline - 5: # Leave 5s for workers
        improved = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            new_net = best_net[:idx] + best_net[idx+1:]
            if _verify_network(new_net):
                best_net = new_net
                improved = True
                break
    
    # Launch Islands
    # Use 4 islands
    num_islands = 4
    # Adjust deadline for workers to be slightly before main deadline
    worker_deadline = deadline - 1.0 
    
    # Prepare arguments
    # Each island starts from the best found so far, but with different random seeds
    # To ensure diversity, we could also perturb the initial net for some islands
    args_list = []
    for i in range(num_islands):
        # Perturb initial net for islands > 0?
        # For simplicity, start all from best_net. The random seeds will diverge them.
        args_list.append((best_net, worker_deadline, 42 + i))
        
    # Use multiprocessing
    # Note: On some systems, mp might have overhead. 
    # If the environment doesn't support mp well, we should fallback.
    # But the task allows standard library.
    
    try:
        ctx = mp.get_context('spawn') # 'fork' is faster on Unix, 'spawn' is safer/cross-platform
        # Try fork first if available for speed
        if sys.platform.startswith('linux'):
             ctx = mp.get_context('fork')
        
        pool = ctx.Pool(num_islands)
        results = pool.map(_worker_process, args_list)
        pool.close()
        pool.join()
        
        # Find the best result
        current_best_size = len(best_net)
        for res in results:
            if res is not None and len(res) < current_best_size:
                # Verify result in main process just in case
                if _verify_network(res):
                    best_net = res
                    current_best_size = len(res)
                    
    except Exception as e:
        # Fallback: run sequential search if multiprocessing fails
        if time.time() < deadline:
            seq_deadline = deadline
            best_net = _island_search(best_net, seq_deadline, 999)
            
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
