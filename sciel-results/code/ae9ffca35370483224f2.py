import time
import random
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

def _simulated_annealing(net, deadline):
    """
    Simulated Annealing to minimize comparator count.
    Moves:
    1. Delete a comparator (if valid, always accept)
    2. Swap two comparators (if valid, accept based on SA criterion? No, size same, so accept if it leads to better 'order' or just random walk? 
       Actually, for size minimization, swapping doesn't change size. It helps escape local minima in the *structure*.
       However, since we only care about final size, and validity is binary, swapping is only useful if it enables subsequent deletions.
       Let's focus on:
       - Delete (size -1)
       - Insert (size +1)
       - Swap (size 0, changes structure)
       
    Strategy:
    - If we can delete, do it (greedy improvement).
    - If we cannot delete, try to swap or insert to escape the local minimum.
    - Accept worse solutions (larger size) with probability exp(-delta/T).
    """
    current_net = net[:]
    current_size = len(current_net)
    best_net = current_net[:]
    best_size = current_size
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Pre-compute a set for fast lookup? No, lists are fine for N=17 (136 pairs).
    
    while time.time() < deadline and T > T_min:
        T *= cooling
        
        # Try a random move
        move_type = random.random()
        
        # 60% chance to try deletion (if possible)
        # 30% chance to try swap
        # 10% chance to try insertion
        
        if move_type < 0.6 and current_size > 0:
            idx = random.randint(0, current_size - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
            if _verify_network(new_net):
                # Always accept improvement
                current_net = new_net
                current_size -= 1
                if current_size < best_size:
                    best_net = current_net[:]
                    best_size = current_size
        elif move_type < 0.9 and current_size >= 2:
            # Swap
            i, j = random.sample(range(current_size), 2)
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            
            # Swapping doesn't change size. We accept if it's valid.
            # To avoid cycling, we could check if it's different from last state, but simple SA handles this via temperature.
            # Since size is same, delta = 0. Accept with prob 1? Or just accept to explore?
            # Let's accept if valid.
            if _verify_network(new_net):
                current_net = new_net
                # No size change, so best_size doesn't update unless we find a deletion later
        else:
            # Insert
            if current_size < 4096:
                comp = random.choice(all_comps)
                idx = random.randint(0, current_size)
                new_net = current_net[:]
                new_net.insert(idx, comp)
                
                if _verify_network(new_net):
                    # Size increased by 1. Delta = 1.
                    # Accept with probability exp(-1/T)
                    if random.random() < (1.0 / T): # Approximation of exp(-1/T) for small T? No, use math.exp
                        # Actually, let's just use math.exp
                        import math
                        if random.random() < math.exp(-1.0 / T):
                            current_net = new_net
                            current_size += 1
                    # If not accepted, we discard new_net
                # If invalid, discard

    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    # Start with Batcher Odd-Merge
    best_net = _build_batcher_odd_merge(N)
    
    if not _verify_network(best_net):
        best_net = _build_seed()
        
    best_size = len(best_net)
    
    # Phase 1: Pure deletion (fast initial reduction)
    # Reuse the logic from previous local search but simpler
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_net = best_net[:idx] + best_net[idx+1:]
            if _verify_network(new_net):
                best_net = new_net
                improved = True
                break
    
    best_size = len(best_net)
    
    # Phase 2: Simulated Annealing
    # We run SA for the remaining time
    # We might run multiple SA runs with different seeds/temperatures if time permits
    # But one long run is usually better.
    
    # To improve the chance of finding a better basin, we can run SA starting from best_net
    # and keep the best result.
    
    # We'll run SA until deadline.
    sa_result = _simulated_annealing(best_net, deadline)
    
    if len(sa_result) < best_size:
        best_net = sa_result
        best_size = len(sa_result)
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
