import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55

def _batcher_odd_even_merge(n, m):
    """
    Generate Batcher Odd-Even Merge network for merging two sorted sequences of length n and m.
    n and m should be such that n >= m.
    Returns a list of comparators.
    """
    if m == 0:
        return []
    if m == 1:
        # Interleave and sort adjacent pairs
        comps = []
        for i in range(n):
            # The standard odd-even merge for n and 1 is just comparing (2i, 2i+1) and (2i+1, 2i+2)?
            # Actually, for m=1, it's a single bitonic merge of size n+1?
            # Let's stick to the recursive definition.
            pass
            
    # Standard recursive definition:
    # 1. Split into two halves? No, Odd-Even Merge is specific.
    # The Odd-Even Merge algorithm:
    # 1. Split the input into two sorted sequences A (length n) and B (length m).
    # 2. Interleave them? No.
    
    # Let's use a known correct implementation for Batcher Odd-Even Merge.
    # From Knuth or standard references:
    # merge(n, m):
    #   if m == 0: return []
    #   if m == 1: return [(2i, 2i+1) for i in range(n)] # This is for n=m?
    
    # Actually, the simplest way to get a dense valid network is to use a Bitonic Sort for the next power of 2 (32) and then... 
    # But we need to sort exactly 18.
    
    # Let's use the recursive Batcher Mergesort structure which is robust.
    pass

def _batcher_mergesort(n):
    """
    Generate a Batcher Odd-Even Mergesort network for n wires.
    """
    if n <= 1:
        return []
    
    k = n // 2
    left = _batcher_mergesort(k)
    right = _batcher_mergesort(n - k)
    
    # Shift right comparators to start at k
    right = [(i + k, j + k) for i, j in right]
    
    # Merge left (0..k-1) and right (k..n-1)
    merge = _batcher_odd_even_merge(k, n - k)
    
    return left + right + merge

def _batcher_odd_even_merge(n, m):
    """
    Batcher Odd-Even Merge for two sorted sequences of length n and m.
    Assumes n >= m.
    """
    if m == 0:
        return []
    if m == 1:
        # The merge of n and 1 is a sequence of comparators
        # (0, 1), (2, 3), ..., (2n-2, 2n-1) ? No.
        # The output size is n+m = n+1.
        # The odd-even merge of n and 1:
        # The 0-th element of the merged sequence is min(A[0], B[0]).
        # The 1st is max(A[0], B[0]) or A[1]...
        
        # Actually, for m=1, the network is just comparing (2i, 2i+1) for i in 0..n-1?
        # Let's check N=2, M=1.
        # A=[a0], B=[b0]. Sorted A, Sorted B.
        # Merged: min(a0,b0), max(a0,b0).
        # Comparator (0,1).
        
        # For N=3, M=1.
        # A=[a0,a1,a2], B=[b0].
        # Merged size 4.
        # Odd-even merge:
        # Split A into A_even=[a0,a2] and A_odd=[a1].
        # Merge A_even with B? No.
        
        # Let's use a simpler, known dense network: **Bitonic Sort for 32 wires**.
        # We can generate a bitonic sort for 32 wires, and then... 
        # The problem is that a 32-wire network doesn't sort 18 wires if we just drop 14.
        # However, if we map the 18 inputs to the first 18 wires of a 32-wire network, 
        # and the other 14 wires are... we don't have 14 inputs.
        
        # Alternative: Use a known good network for 18 wires?
        # The task says "best known is 77".
        # We can try to find a network by starting with a dense one.
        
        # Let's try a different dense network: **Odd-Even Transposition** is 153.
        # **Bitonic Sort for 16 wires** is 60.
        # We can sort 16 wires, and then merge the 2 remaining wires into the 16?
        # This is complex.
        
        # Let's stick to the current seed (153) but improve the search.
        # The current score is 1.43 (size ~107).
        # If we can get to 100, score is 1.53.
        # If we can get to 90, score is 1.70.
        
        # The "Knowledge" says Batcher helped. Let's implement Batcher correctly.
        
        # Reference for Batcher Odd-Even Merge:
        # def odd_even_merge(n, m):
        #   if m == 0: return []
        #   if m == 1:
        #      return [(2*i, 2*i+1) for i in range(n)] # This is for N=M?
        
        # Let's use a simpler dense generator:
        # Generate a complete graph of comparators? No, too many.
        
        # Let's use the **AKS** or similar? No.
        
        # Let's use the current seed and improve the local search to be more aggressive.
        # The current local search is okay.
        
        # One improvement: **Parallel Greedy Reduction**.
        # Instead of removing one at a time, try to remove multiple independent comparators at once.
        
        # Another improvement: **Better Initialization**.
        # Let's use a **Bitonic Sort for 18 wires** by padding to 32, sorting, and then... 
        # Actually, we can construct a Bitonic Sort for 18 wires by treating it as a merge of two 9-wire bitonic sorts?
        # Bitonic sort requires power of 2.
        
        # Let's try **Heapsort** network? No.
        
        # Let's go back to the current program and optimize the `verify` and `greedy_reduce`.
        # The verify function is the bottleneck.
        # We can pre-compute the effect of each comparator? No, it depends on the state.
        
        # Let's try to speed up the local search by using a **coarser** verification for the first few steps?
        # No, correctness is strict.
        
        # Let's try to improve the **Initialization** to a **Random Dense Network**.
        # Generate a random network with 200 comparators.
        # Then reduce it.
        
        # Or, use the **Odd-Even Transposition** but shuffle it first?
        
        # The most promising lead is the **Batcher** initialization.
        # I will implement a correct Batcher Odd-Even Mergesort for N=18.
        
        pass

def _batcher_odd_even_merge(n, m):
    if m == 0:
        return []
    if m == 1:
        # Merge n and 1
        # The comparators are (2i, 2i+1) for i in range(n)?
        # Let's check n=1, m=1. (0,1). Correct.
        # n=2, m=1. A=[a0,a1], B=[b0].
        # Merged: [min(a0,b0), max(a0,b0) or a1?, ...]
        # The odd-even merge of 2 and 1:
        # Split A into A0=[a0], A1=[a1].
        # Merge A0 and B (size 1 and 1) -> M01 (size 2).
        # Merge A1 and empty? No.
        
        # Correct Recursive Definition:
        # merge(n, m):
        #   if m == 0: return []
        #   if m == 1:
        #      return [(2*i, 2*i+1) for i in range(n)] # This is WRONG for n>1.
        
        # Let's use a library or a known correct snippet.
        # Since I can't use external libs for network generation, I will write it carefully.
        
        # From "Sorting Networks" by Knuth:
        # Odd-Even Merge(n, m):
        # 1. If m == 0, return empty.
        # 2. If m == 1, return comparators (0,1), (2,3), ..., (2n-2, 2n-1).
        #    Wait, the indices are relative to the merged sequence of length n+m.
        #    The inputs are interleaved? No, the inputs are A[0..n-1] and B[0..m-1].
        #    The network operates on the combined wires.
        
        # Let's assume the inputs are on wires 0..n-1 and n..n+m-1.
        # The output is on wires 0..n+m-1.
        
        # For m=1:
        # The network compares (0, 1), (2, 3), ..., (2n-2, 2n-1).
        # This effectively merges A and B if A and B are sorted?
        # If A=[1,3], B=[2]. Merged wires: 1,2,3?
        # Wires: 0:1, 1:2, 2:3.
        # Comp (0,1): min(1,2)=1, max=2.
        # Comp (2,3): wire 3 is empty?
        
        # This is getting too complex to implement from scratch correctly in a short time.
        
        # I will stick to the **Odd-Even Transposition** seed but improve the **Local Search** and **Greedy Reduction**.
        # Specifically, I will add a **"Swap" move** that swaps two non-adjacent comparators, and a **"Relocate" move**.
        # I will also try to **parallelize** the greedy reduction by checking multiple removals in parallel (conceptually, by just doing more iterations).
        
        pass

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    for w in range(N):
        for b in range(num_blocks):
            base = b * 64
            if w < 6:
                period = 1 << (w + 1)
                half = 1 << w
                mask = 0
                for bit in range(64):
                    if ((base + bit) >> w) & 1:
                        mask |= (1 << bit)
                states[w][b] = mask
            else:
                if (base >> w) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def verify(comparators):
        for w in range(N):
            workspace[w][:] = states[w]
        
        for i, j in comparators:
            a = workspace[i]
            b = workspace[j]
            workspace[i] = a & b
            workspace[j] = a | b
        
        for w in range(N - 1):
            if np.any(workspace[w] & ~workspace[w + 1]):
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

    # Initial Network
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    current_network = list(best_network)
    stagnation_counter = 0
    max_stagnation = 100 # Reduced to allow more restarts
    
    while time.time() < deadline - 5:
        move_type = random.random()
        
        if move_type < 0.3:
            # Mutation
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                trial = list(current_network)
                trial[idx] = (a, b)
        elif move_type < 0.5:
            # Swap Adjacent
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                trial = list(current_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
        elif move_type < 0.7:
            # Insert
            idx = random.randint(0, len(current_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
        elif move_type < 0.85:
            # Slide
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                trial = list(current_network)
                if idx == 0:
                    trial[0], trial[1] = trial[1], trial[0]
                elif idx == len(trial) - 1:
                    trial[idx], trial[idx-1] = trial[idx-1], trial[idx]
                else:
                    if random.random() < 0.5:
                        trial[idx], trial[idx-1] = trial[idx-1], trial[idx]
                    else:
                        trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
        else:
            # Relocate
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                comp = current_network[idx]
                new_idx = random.randint(0, len(current_network) - 1)
                trial = list(current_network)
                del trial[idx]
                if new_idx > idx:
                    new_idx -= 1
                trial.insert(new_idx, comp)
        
        reduced = greedy_reduce(trial)
        
        if len(reduced) < len(best_network):
            best_network = reduced
            best_len = len(reduced)
            current_network = reduced
            stagnation_counter = 0
        elif len(reduced) < len(current_network):
            current_network = reduced
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick
                K = random.randint(3, 7)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                
                current_network = greedy_reduce(current_network)
                stagnation_counter = 0

    return best_network
