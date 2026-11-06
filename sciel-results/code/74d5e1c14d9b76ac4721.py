import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    N = 18
    total_inputs = 1 << N
    
    # 1. Precompute init_wires
    init_wires = [0] * N
    for k in range(N):
        w = 0
        for i in range(total_inputs):
            if (i >> k) & 1:
                w |= (1 << i)
        init_wires[k] = w
        
    # 2. Generate Batcher's Network
    def batcher_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        # Split
        n1 = n // 2
        n2 = n - n1
        # Sort halves
        net1 = batcher_sort(n1)
        net2 = batcher_sort(n2)
        # Shift net2
        net2 = [(i + n1, j + n1) for i, j in net2]
        
        # Merge
        # For Batcher's merge, we need a specific merge network.
        # Let's use the standard recursive merge for 2k wires.
        # But n might not be 2k.
        # For n=18, n1=9, n2=9.
        # Merge 9 and 9.
        
        def merge(k, offset1, offset2):
            # Merges two sorted lists of length k
            # List 1 at offset1, List 2 at offset2
            # This is complex for arbitrary offsets.
            # Let's assume contiguous blocks for simplicity in the generator.
            pass
            
        # This is getting too complicated to implement Batcher's from scratch correctly in a short time.
        
        # Fallback: Use Odd-Even Transposition, which is definitely correct.
        # But it's large.
        
        # Let's use a "Knuth" network or similar?
        
        # I will use the Odd-Even Transposition network as the seed,
        # but I will improve the optimization loop significantly.
        
        comparators = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # 3. Verify Function
    def verify_network(comparators):
        wires = list(init_wires)
        for i, j in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for k in range(N - 1):
            if wires[k] & ~wires[k + 1]:
                return False
        return True

    # 4. Optimization
    best_net = batcher_sort(N)
    best_size = len(best_net)
    
    # Greedy Removal
    def greedy_remove(net):
        changed = True
        while changed:
            changed = False
            for i in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 5:
                    return net
                test = net[:i] + net[i+1:]
                if verify_network(test):
                    net = test
                    changed = True
                    break
        return net

    best_net = greedy_remove(best_net)
    best_size = len(best_net)
    
    # Simulated Annealing
    current_net = list(best_net)
    current_size = len(current_net)
    t = 100.0
    alpha = 0.999
    
    while time.time() < deadline - 2:
        # Move
        r = random.random()
        if r < 0.7:
            # Remove
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
        elif r < 0.9:
            # Swap
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            candidate = list(current_net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            # Insert
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j: continue
            if i > j: i, j = j, i
            pos = random.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            
        if not verify_network(candidate):
            continue
            
        new_size = len(candidate)
        delta = new_size - current_size
        
        if delta < 0:
            current_net = candidate
            current_size = new_size
            if new_size < best_size:
                best_net = list(current_net)
                best_size = new_size
                # Greedy burst
                best_net = greedy_remove(best_net)
                best_size = len(best_net)
                current_net = list(best_net)
                current_size = best_size
        else:
            if delta == 0:
                if random.random() < 0.5:
                    current_net = candidate
            else:
                prob = math.exp(-delta / t)
                if random.random() < prob:
                    current_net = candidate
                    current_size = new_size
                    
        t *= alpha
        if t < 1.0:
            t = 50.0
            current_net = list(best_net)
            current_size = best_size
            
    return best_net
