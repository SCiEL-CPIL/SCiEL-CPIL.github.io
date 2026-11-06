import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55

def _bitonic_sort_network(n):
    """
    Generate a Bitonic Sort network for n wires.
    Note: Bitonic sort typically works on powers of 2.
    For n=18, we can construct it by merging bitonic networks or using a general recursive definition.
    However, a simpler dense network is often a good seed.
    Let's use a standard Bitonic Sort for 16 wires and merge the last 2? 
    Actually, for n=18, we can just use a recursive Batcher Mergesort or a Bitonic Sort adapted for non-powers of 2.
    
    A robust seed is the "Odd-Even Merge" based Batcher Mergesort.
    Let's implement a correct Batcher Odd-Even Mergesort for N=18.
    """
    if n <= 1:
        return []
    
    # Batcher Odd-Even Mergesort
    # Split into two halves
    k = n // 2
    left = _bitonic_sort_network(k)
    right = _bitonic_sort_network(n - k)
    
    # Shift right comparators
    right = [(i + k, j + k) for i, j in right]
    
    # Merge the two sorted halves using Odd-Even Merge
    merge = _odd_even_merge(k, n - k)
    
    return left + right + merge

def _odd_even_merge(n, m):
    """
    Batcher Odd-Even Merge for two sorted sequences of length n and m.
    Inputs are assumed to be on wires 0..n-1 and n..n+m-1.
    Returns comparators on wires 0..n+m-1.
    """
    if m == 0:
        return []
    if m == 1:
        # Merge n and 1
        # The comparators are (2*i, 2*i+1) for i in range(n)?
        # Let's verify for n=1, m=1: (0,1). Correct.
        # n=2, m=1: Wires 0,1 (A), 2 (B).
        # Standard O-E Merge for n=2, m=1:
        # Split A into A_even (wire 0), A_odd (wire 1).
        # Merge A_even (len 1) and B (len 1) -> M01 (wires 0,1? No, indices shift).
        
        # Actually, the recursive definition for O-E Merge (n, m) with n >= m:
        # 1. Split A into A0 (even indices) and A1 (odd indices).
        # 2. Merge A0 and B? No.
        # 
        # Reference:
        # O-E Merge(n, m):
        #   if m == 0: return []
        #   if m == 1:
        #      return [(2*i, 2*i+1) for i in range(n)] # This is for n=m?
        #      
        # Correct Recursive Step:
        # Split A into A_even (length ceil(n/2)) and A_odd (length floor(n/2)).
        # Split B into B_even (length ceil(m/2)) and B_odd (length floor(m/2)).
        # Merge A_even and B_even -> M_even
        # Merge A_odd and B_odd -> M_odd
        # Then merge M_even and M_odd? No, interleave them.
        
        # This is tricky to implement correctly from scratch in a short time.
        # Let's use a simpler, known dense network: **Odd-Even Transposition**.
        # It's 153 comparators. It's a valid seed.
        # To get a better seed, we can use **Bitonic Sort for 32 wires** and project?
        # No, that's complex.
        
        # Let's stick to Odd-Even Transposition for the seed, but the local search will improve it.
        # The main gain comes from the faster verifier and better search moves.
        return [] # Fallback to OET if this fails, but we won't call this if we use OET directly.

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Fast Verification Setup
    # We use a single array of uint64s. 
    # Each bit represents one of the 2^18 boolean inputs.
    # 2^18 = 262144 inputs.
    # 262144 / 64 = 4096 blocks.
    num_blocks = 2**N // 64
    # Total size: N * num_blocks
    # We pre-compute the initial state for each wire.
    # Wire w has bit b set if the b-th input has 1 at position w.
    
    # Instead of storing N arrays, we store one flat array.
    # Index: w * num_blocks + b
    initial_states = np.zeros(N * num_blocks, dtype=np.uint64)
    
    for w in range(N):
        for b in range(num_blocks):
            base = b * 64
            # For wire w, the bit is 1 if (input_index >> w) & 1
            # input_index = base + bit
            # We can generate the mask for this block.
            # If w < 6, the pattern repeats every 2^(w+1).
            # If w >= 6, the block is either all 0 or all 1.
            
            if w < 6:
                period = 1 << (w + 1)
                half = 1 << w
                mask = 0
                # Optimization: Pre-compute masks for small w?
                # Or just compute. 18 * 4096 = 73728 iterations. Fast enough.
                for bit in range(64):
                    idx = base + bit
                    if (idx >> w) & 1:
                        mask |= (1 << bit)
                initial_states[w * num_blocks + b] = mask
            else:
                # The bit w is constant within a block of 64 if w >= 6?
                # No, 2^w >= 64. So within a block of 64 consecutive integers,
                # the w-th bit is constant.
                if (base >> w) & 1:
                    initial_states[w * num_blocks + b] = (1 << 64) - 1
                else:
                    initial_states[w * num_blocks + b] = 0

    workspace = np.zeros(N * num_blocks, dtype=np.uint64)
    
    def verify(comparators):
        # Copy initial states to workspace
        workspace[:] = initial_states
        
        for i, j in comparators:
            # Compare-exchange wires i and j
            # min goes to i, max goes to j
            # In bitwise: min = A & B, max = A | B
            offset_i = i * num_blocks
            offset_j = j * num_blocks
            
            a = workspace[offset_i : offset_i + num_blocks]
            b = workspace[offset_j : offset_j + num_blocks]
            
            # We need to update workspace in place.
            # min_val = a & b
            # max_val = a | b
            # workspace[i] = min_val
            # workspace[j] = max_val
            
            # Slicing creates views in numpy, so this works.
            # But we must be careful with overlap if i==j (not allowed) or if views alias?
            # i != j, so no aliasing between i and j blocks.
            
            min_vals = a & b
            max_vals = a | b
            
            workspace[offset_i : offset_i + num_blocks] = min_vals
            workspace[offset_j : offset_j + num_blocks] = max_vals
        
        # Check if sorted:
        # For all w from 0 to N-2, workspace[w] <= workspace[w+1]
        # In bitwise: workspace[w] & ~workspace[w+1] should be 0.
        for w in range(N - 1):
            offset_w = w * num_blocks
            offset_w1 = (w + 1) * num_blocks
            # Check if any bit is 1 in workspace[w] but 0 in workspace[w+1]
            # i.e., workspace[w] has a 1 where workspace[w+1] has a 0.
            # violation = workspace[w] & ~workspace[w+1]
            # if any(violation): return False
            
            # Using np.any on uint64 array
            if np.any(workspace[offset_w : offset_w + num_blocks] & ~workspace[offset_w1 : offset_w1 + num_blocks]):
                return False
        return True

    def greedy_reduce(net):
        cur_net = list(net)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            idx = 0
            while idx < len(cur_net):
                if time.time() > deadline - 2:
                    return cur_net
                trial = cur_net[:idx] + cur_net[idx+1:]
                if verify(trial):
                    cur_net = trial
                    changed = True
                else:
                    idx += 1
        return cur_net

    def local_search(net):
        cur_net = list(net)
        cur_len = len(cur_net)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to remove
            # (Already done by greedy_reduce, but we can integrate)
            
            # Try 2-opt: Swap two comparators
            # Or relocate one comparator
            # Given time constraints, we will try random relocations and swaps.
            
            # To be efficient, we only accept if it reduces length (after reduction) or stays same but allows further reduction?
            # Actually, standard local search for sorting networks:
            # 1. Try to remove a comparator.
            # 2. If no removal, try to move a comparator to a new position.
            # 3. If no move, try to change a comparator's wires.
            
            # We will do a limited number of random moves.
            for _ in range(50): # Limit iterations per "improved" phase to avoid getting stuck
                if time.time() > deadline - 5:
                    return cur_net
                
                move_type = random.random()
                if move_type < 0.5 and len(cur_net) > 1:
                    # Relocate
                    i = random.randint(0, len(cur_net) - 1)
                    j = random.randint(0, len(cur_net) - 1)
                    if i == j: continue
                    trial = list(cur_net)
                    comp = trial.pop(i)
                    # Adjust index if j > i
                    if j > i:
                        j -= 1
                    trial.insert(j, comp)
                elif move_type < 0.8 and len(cur_net) > 0:
                    # Change wires of a comparator
                    i = random.randint(0, len(cur_net) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    trial = list(cur_net)
                    trial[i] = (a, b)
                else:
                    # Swap two comparators (2-opt)
                    if len(cur_net) < 2: continue
                    i, j = random.sample(range(len(cur_net)), 2)
                    trial = list(cur_net)
                    trial[i], trial[j] = trial[j], trial[i]
                
                # Reduce the trial network
                reduced = greedy_reduce(trial)
                
                if len(reduced) < len(cur_net):
                    cur_net = reduced
                    cur_len = len(cur_net)
                    improved = True
                    break # Restart the loop with new network
                elif len(reduced) == len(cur_net):
                    # Accept with some probability to escape local minima? 
                    # Or just accept if it's "equivalent" but different structure?
                    # For now, only accept if strictly better to keep it simple and fast.
                    pass
        
        return cur_net

    # Initial Network
    # Use Odd-Even Transposition
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    current_network = list(best_network)
    
    # Search Loop
    while time.time() < deadline - 10:
        # Perform local search on current network
        # We might want to restart from best_network occasionally
        if random.random() < 0.1:
            current_network = list(best_network)
            # Add some random noise
            K = random.randint(1, 3)
            for _ in range(K):
                if not current_network: break
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                current_network[idx] = (a, b)
        
        current_network = local_search(current_network)
        
        if len(current_network) < best_len:
            best_network = current_network
            best_len = len(best_network)
        
        # If we are very close to deadline, stop
        if time.time() > deadline - 2:
            break

    # Final reduction
    best_network = greedy_reduce(best_network)
    
    return best_network
