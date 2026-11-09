import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.85

# Pre-compute initial and expected wires for verification
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

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    """
    if n <= 1:
        return []
    
    # Split into n1 (odd) and n2 (even)
    n1 = (n + 1) // 2
    n2 = n // 2
    
    # 1. Recursively sort the two halves
    net = []
    # First half (indices 0 to n1-1)
    net.extend(_build_batcher_odd_merge(n1))
    
    # Second half (indices n1 to n-1)
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    # 2. Merge the two sorted halves
    for i in range(n1):
        net.append((i, i + n2))
        
    # Recursively merge the even subsequence (size n1)
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            net.append((2*i, 2*j))
            
    # Recursively merge the odd subsequence (size n2)
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            net.append((2*i+1, 2*j+1))
            
    return net

def _local_search_remove(net, deadline):
    """
    Steepest descent local search: remove any comparator that is redundant.
    """
    current_net = net[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each comparator
        # We iterate in reverse to avoid index shifting issues during removal
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Create a copy without the comparator at idx
            new_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network(new_net):
                current_net = new_net
                improved = True
                # Restart the loop to catch new redundancies
                break
                
    return current_net

def _local_search_swap(net, deadline):
    """
    Local search that tries to swap adjacent comparators to find better ordering
    which might allow further removals.
    """
    current_net = net[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current_net) - 1):
            if time.time() > deadline:
                break
            # Try swapping idx and idx+1
            new_net = current_net[:]
            new_net[idx], new_net[idx+1] = new_net[idx+1], new_net[idx]
            
            # Check if this swap allows removal of a comparator elsewhere
            # Simple heuristic: just check if the swapped network is valid (it should be if original was)
            # But maybe the swap creates a redundancy?
            # Actually, swapping comparators in a valid sorting network keeps it valid.
            # The goal is to find an ordering that allows removal.
            # So we try to remove a comparator from the swapped version.
            for rem_idx in range(len(new_net)):
                test_net = new_net[:rem_idx] + new_net[rem_idx+1:]
                if _verify_network(test_net):
                    current_net = test_net
                    improved = True
                    break
            if improved:
                break
    return current_net

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    # Start with the Batcher Odd-Merge network
    best_net = _build_batcher_odd_merge(N)
    
    # Verify it
    if not _verify_network(best_net):
        # Fallback to seed if Batcher implementation has a bug
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Phase 1: Pure deletion on the initial network
    best_net = _local_search_remove(best_net, deadline)
    best_size = len(best_net)
    
    # Phase 2: Iterative improvement with random restarts/perturbations
    # Strategy: Add random comparators, then try to remove others.
    # This explores a wider search space than pure deletion.
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # We will run several rounds of "add then remove" search.
    # Each round:
    # 1. Start from best_net
    # 2. Add K random comparators
    # 3. Run local search remove
    # 4. If improved, update best_net
    
    K = 5 # Number of random comparators to add
    
    while time.time() < deadline:
        perturbed_net = best_net[:]
        # Add K random comparators
        for _ in range(K):
            if len(perturbed_net) >= 4096:
                break
            comp = rng.choice(all_comps)
            idx = rng.randint(0, len(perturbed_net))
            perturbed_net.insert(idx, comp)
            
        # Run local search remove
        candidate = _local_search_remove(perturbed_net, deadline)
        
        if len(candidate) < best_size:
            best_net = candidate
            best_size = len(candidate)
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
