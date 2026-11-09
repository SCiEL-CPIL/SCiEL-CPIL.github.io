import time
import random
import math
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.88

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

def _greedy_prune(net, deadline):
    """
    Systematic Greedy Pruning:
    Repeatedly scan the network and remove any comparator that is redundant.
    Since removing a comparator can make others redundant, we loop until no more
    can be removed or the deadline is reached.
    
    To avoid getting stuck in a specific order bias, we randomize the scan order
    in each pass, but the acceptance criterion is strictly: "does removal preserve sorting?".
    """
    current_net = net[:]
    
    while time.time() < deadline:
        improved = False
        # Randomize the order of indices to try to remove
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                return current_net
            
            # Try to remove current_net[idx]
            new_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network(new_net):
                current_net = new_net
                improved = True
                # Since the network changed, the indices list is now invalid.
                # Break out to restart the pass with a fresh random order.
                break
        
        if not improved:
            # No more comparators can be removed.
            break
            
    return current_net

def _steepest_descent(net, deadline):
    """
    Steepest Descent Local Search:
    1. Try all possible deletions. If any works, take the first one found (or best, but all deletions have same cost change).
    2. If no deletion works, try all possible swaps.
    3. If no swap works, try all possible insertions (to escape local minima if we allow size increase, but here we want to minimize size, so we usually stop).
    
    Actually, for size minimization, we primarily care about deletions.
    Swaps don't change size. Insertions increase size.
    So a pure "minimize size" local search is just "find a deletion".
    However, sometimes a swap is needed to enable a deletion.
    
    Strategy:
    Loop:
      1. Try all deletions. If one works, apply it and restart loop.
      2. If no deletion works, try all swaps. If one works AND enables a subsequent deletion, apply it?
         This is complex. 
         
    Alternative: 
    We can view this as a search for a minimal subsequence.
    A simpler robust approach:
    1. Greedy Prune (as above).
    2. If stuck, perform random swaps to perturb the network, then Greedy Prune again.
       This is effectively "Random Restarted Greedy Pruning".
    """
    best_net = net[:]
    best_size = len(best_net)
    
    current_net = net[:]
    
    while time.time() < deadline:
        # Phase 1: Greedy Prune
        current_net = _greedy_prune(current_net, deadline)
        
        if len(current_net) < best_size:
            best_net = current_net[:]
            best_size = len(best_net)
            
        # If we hit a local minimum (no more deletions possible), try to escape
        # by performing a random swap and then pruning again.
        if len(current_net) > 1:
            # Perform a random swap
            i, j = random.sample(range(len(current_net)), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
            # Check if the swapped network is valid (it must be, since swaps preserve the set of comparators, 
            # but the *order* matters for sorting networks? 
            # Wait, does order matter?
            # A sorting network is a sequence of comparators. Swapping two comparators changes the network.
            # It might become invalid.
            if _verify_network(new_net):
                current_net = new_net
            else:
                # If invalid, try a few more swaps or give up on this restart
                # To keep it simple, just revert and try a different restart strategy?
                # Or just break if we can't find a valid perturbation quickly?
                # Let's try 10 random swaps
                found_valid = False
                for _ in range(10):
                    i, j = random.sample(range(len(current_net)), 2)
                    new_net = current_net[:]
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    if _verify_network(new_net):
                        current_net = new_net
                        found_valid = True
                        break
                if not found_valid:
                    # Stuck hard. Break out of the loop? Or try a larger perturbation?
                    # For now, break to avoid wasting time.
                    break
        else:
            break
            
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with Batcher Odd-Merge
    best_net = _build_batcher_odd_merge(N)
    
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Run Steepest Descent with Random Restarts (via swaps)
    result_net = _steepest_descent(best_net, deadline)
    
    if len(result_net) < best_size:
        best_net = result_net
        best_size = len(result_net)
        
    # Final check
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
