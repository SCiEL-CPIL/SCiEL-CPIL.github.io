import time
import random
import math
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    Returns a list of comparators (i, j) for n wires.
    """
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    
    # 1. Sort wires 0, 2, 4... (indices 0..n1-1 in sub-problem)
    sub_odd = _build_batcher_odd_merge(n1)
    for i, j in sub_odd:
        net.append((2*i, 2*j))
        
    # 2. Sort wires 1, 3, 5... (indices 0..n2-1 in sub-problem)
    sub_even = _build_batcher_odd_merge(n2)
    for i, j in sub_even:
        net.append((2*i+1, 2*j+1))
        
    # 3. Merge: compare (i, i+n2) for i in 0..n1-1
    for i in range(n1):
        net.append((i, i + n2))
        
    # 4. Sort even-indexed wires again (indices 0..n1-1)
    sub_even2 = _build_batcher_odd_merge(n1)
    for i, j in sub_even2:
        net.append((2*i, 2*j))
        
    # 5. Sort odd-indexed wires again (indices 0..n2-1)
    sub_odd2 = _build_batcher_odd_merge(n2)
    for i, j in sub_odd2:
        net.append((2*i+1, 2*j+1))
        
    return net

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_subnet(net, n_wires, offset=0):
    """
    Verify a sub-network of size n_wires, assuming inputs are bits 0..n_wires-1
    and outputs should be sorted relative to each other.
    This is a simplified check for local optimization: 
    We check if the sub-network sorts all 2^n_wires boolean inputs.
    """
    total_masks = 1 << n_wires
    initial_wires = [0] * n_wires
    expected_wires = [0] * n_wires
    
    # Precompute for this sub-size
    for i in range(n_wires):
        w = 0
        for k in range(total_masks):
            if (k >> i) & 1:
                w |= (1 << k)
        initial_wires[i] = w
        
    for i in range(n_wires):
        w = 0
        threshold = n_wires - i
        for k in range(total_masks):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        expected_wires[i] = w

    wires = initial_wires[:]
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    for i in range(n_wires):
        if wires[i] != expected_wires[i]:
            return False
    return True

def _optimize_subnet(net, n_wires, deadline):
    """
    Optimize a sub-network of size n_wires using local search.
    Returns the optimized net.
    """
    current_net = net[:]
    best_net = net[:]
    best_size = len(net)
    
    # Simple hill climbing: try to remove comparators
    # Since n_wires is small (8 or 9), verification is very fast.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_subnet(new_net, n_wires):
                current_net = new_net
                if len(current_net) < best_size:
                    best_net = current_net[:]
                    best_size = len(current_net)
                improved = True
                break
    
    return best_net

def _build_hierarchical_network(deadline):
    """
    1. Split 17 wires into 9 (0-8) and 8 (9-16).
    2. Build Batcher networks for each.
    3. Optimize each sub-network independently.
    4. Add a merge layer.
    5. Verify full network.
    """
    n1 = 9
    n2 = 8
    
    # Build base sub-networks
    net1 = _build_batcher_odd_merge(n1)
    net2 = _build_batcher_odd_merge(n2)
    
    # Optimize sub-networks
    # Allocate 40% of time to sub-optimization
    sub_deadline = time.time() + (deadline - time.time()) * 0.4
    
    opt_deadline_1 = time.time() + (sub_deadline - time.time()) * 0.5
    opt_net1 = _optimize_subnet(net1, n1, opt_deadline_1)
    
    opt_deadline_2 = sub_deadline
    opt_net2 = _optimize_subnet(net2, n2, opt_deadline_2)
    
    # Compose
    full_net = []
    # Map net1 (wires 0-8)
    for i, j in opt_net1:
        full_net.append((i, j))
        
    # Map net2 (wires 9-16)
    for i, j in opt_net2:
        full_net.append((i + n1, j + n1))
        
    # Merge Layer
    # Standard merge for bitonic merge would be complex.
    # Simple approach: Compare corresponding wires to ensure global order?
    # Actually, if we just sort them independently, the result is not globally sorted.
    # We need a merge network.
    # A simple merge network for 9 and 8 wires:
    # We can use a "compare-swap" layer that interleaves them?
    # Or just append a full sorting network for the remaining "mess"?
    # Better: Use a known small merge network or just append a Batcher merge?
    # Let's try a simple "zip" merge: compare (i, i+n2) for i in 0..n1-1? No, that's not a full merge.
    
    # To be safe and correct, let's append a full Batcher network for the combined 17 wires?
    # No, that defeats the purpose.
    
    # Alternative: Use the "Odd-Even Merge" structure for the whole 17 wires,
    # but optimize the leaves.
    # The current _build_batcher_odd_merge(17) already does this recursively.
    # The problem is that optimizing the leaves independently doesn't guarantee the merge is optimal.
    
    # Let's try a different composition:
    # 1. Sort 0-8
    # 2. Sort 9-16
    # 3. Merge 0-8 and 9-16 using a simple "bitonic merge" logic?
    # Bitonic merge requires the two halves to be sorted in opposite directions.
    # If both are sorted ascending, we can't just merge.
    
    # Let's stick to the full Batcher structure but optimize the recursive calls?
    # That's hard to do with the current _build_batcher_odd_merge.
    
    # Revised Strategy:
    # 1. Generate full Batcher network for 17.
    # 2. Identify the "leaf" comparators (those inside the smallest sub-networks).
    # 3. This is complex.
    
    # Let's try a simpler hierarchical approach:
    # Use the full Batcher network.
    # Then, perform local search on the FULL network, but use a "fast verification" 
    # that only checks a subset of inputs? No, correctness requires all.
    
    # Actually, the biggest win is just doing a very aggressive greedy removal on the full Batcher network.
    # The current program does this.
    # How to improve?
    # The current program's local search is "structured" but maybe too random.
    # Let's try a "simulated annealing" approach with a better cooling schedule
    # and a focus on "redundancy" detection.
    
    # Or, let's try to verify sub-parts of the network?
    # If a comparator (i, j) only affects wires i and j, and we can prove that
    # for all inputs, the state of wires i and j is already sorted before this comparator,
    # then it is redundant.
    # This is hard to prove globally.
    
    # Let's try a different decomposition:
    # Split 17 into 5, 6, 6?
    # Or 8, 9?
    # If we split into 8 and 9, sort them, then we need to merge.
    # A merge network for 8 and 9 wires can be constructed.
    # If we use a Batcher merge for 8+9, the total size is C(8) + C(9) + C_merge.
    # C(8) ~ 19, C(9) ~ 25. C_merge for 17 is roughly N^2/4 ~ 70?
    # Total ~ 114. Batcher 17 is ~ 71? No, Batcher 17 is ~ 65-70.
    # Wait, Batcher 17 is known to be around 65-70 comparators?
    # Actually, Batcher Odd-Merge for 16 is 64. For 17, it's a bit more.
    # Let's check the size of _build_batcher_odd_merge(17).
    
    # If the hierarchical approach (8+9+merge) results in a larger network, it's a bad start.
    # Batcher is already quite good.
    
    # So, the "Hierarchical" strategy might just be:
    # 1. Start with Batcher.
    # 2. Use a "divide and conquer" verification to speed up the search?
    # No, verification is already O(1) with bitmasks.
    
    # The bottleneck is the NUMBER of verifications we can do.
    # To do more, we need to make each verification faster?
    # It's already very fast.
    
    # Maybe the issue is the SEARCH ALGORITHM.
    # Greedy removal gets stuck.
    # SA with insertion helps escape but is slow.
    
    # Let's try "Random Restarts" with a tighter time budget per restart.
    # 1. Run greedy removal on Batcher.
    # 2. Run a short SA.
    # 3. Repeat with different random seeds?
    
    # Or, let's try to optimize the "Merge" step specifically?
    
    # Let's go back to the prompt's suggestion: "decomposing... into a hierarchy... allowing search to optimize smaller, faster-to-verify sub-problems independently".
    
    # If I build a network that is NOT a single Batcher, but a composition of two optimized sub-networks and a merge,
    # and if that composition is SMALLER than the standard Batcher, then I win.
    # But standard Batcher is likely near-optimal for its structure.
    
    # What if I use a "Shallow" network?
    # Depth is reported but not scored directly (score is count).
    # But shallow networks might have more redundancy?
    
    # Let's try a "Hybrid" approach:
    # 1. Build Batcher 17.
    # 2. Identify the "depth" of each comparator.
    # 3. Prioritize removal of comparators in the "middle" depths?
    
    # Actually, let's look at the "Inspiration 1": "Steepest-Descent Local Search".
    # The current program does "Greedy" then "SA".
    # Steepest Descent: At each step, try ALL possible single deletions and pick the one that works?
    # No, that's what greedy does (but it stops at the first one).
    # Steepest Descent: Try all deletions, if multiple work, pick the one that...?
    # They all have the same score (size-1).
    # So it's just random among the valid deletions.
    
    # Let's try "Simulated Annealing" with a much better move generator.
    # Moves:
    # 1. Delete a random comparator.
    # 2. Swap two comparators.
    # 3. Replace a comparator (i, j) with (i, k) or (k, j)?
    
    # Let's implement a robust SA with a good cooling schedule.
    
    pass

def _build_hierarchical_network(deadline):
    # This function is a placeholder for the strategy described in Step 1.
    # However, as reasoned above, simple decomposition might not yield a smaller network than Batcher.
    # Instead, we will use a "Multi-Start" approach with the Batcher network.
    
    best_net = _build_batcher_odd_merge(N)
    best_size = len(best_net)
    
    # Multi-Start Local Search
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        current_deadline = time.time() + time_per_restart
        
        # Start from Batcher
        current_net = _build_batcher_odd_merge(N)
        
        # Phase 1: Greedy Removal
        improved = True
        while improved and time.time() < current_deadline * 0.5:
            improved = False
            for idx in range(len(current_net) - 1, -1, -1):
                if time.time() > current_deadline * 0.5:
                    break
                new_net = current_net[:idx] + current_net[idx+1:]
                if _verify_network(new_net):
                    current_net = new_net
                    improved = True
                    break
        
        # Phase 2: SA
        T = 2.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < current_deadline and T > T_min:
            T *= cooling
            move_type = random.random()
            
            if move_type < 0.8 and len(current_net) > 0:
                idx = random.randint(0, len(current_net) - 1)
                new_net = current_net[:idx] + current_net[idx+1:]
                if _verify_network(new_net):
                    current_net = new_net
                    if len(current_net) < best_size:
                        best_net = current_net[:]
                        best_size = len(current_net)
            elif move_type < 0.95 and len(current_net) >= 2:
                i, j = random.sample(range(len(current_net)), 2)
                new_net = current_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
                if _verify_network(new_net):
                    current_net = new_net
            else:
                if len(current_net) < 4096:
                    comp = (random.randint(0, N-2), random.randint(1, N-1))
                    if comp[0] >= comp[1]:
                        comp = (comp[1], comp[0])
                    idx = random.randint(0, len(current_net))
                    new_net = current_net[:]
                    new_net.insert(idx, comp)
                    if _verify_network(new_net):
                        if random.random() < math.exp(-1.0 / T):
                            current_net = new_net

    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Use the multi-start strategy
    best_net = _build_hierarchical_network(deadline)
    
    # Final safety check
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
