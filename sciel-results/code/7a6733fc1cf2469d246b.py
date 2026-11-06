import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # 1. Generate Batcher's Odd-Merge Network for N=16
    def sort(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        m = n // 2
        net1 = sort(m)
        net2 = [(i + m, j + m) for (i, j) in sort(m)]
        net3 = merge(n)
        return net1 + net2 + net3

    def merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        m = n // 2
        stage1 = [(i, i + m) for i in range(m)]
        
        even_net = []
        for (i, j) in merge(m):
            even_net.append((2*i, 2*j))
            
        odd_net = []
        for (i, j) in merge(m):
            odd_net.append((2*i + 1, 2*j + 1))
            
        return stage1 + even_net + odd_net

    network = sort(16)
    
    # 2. Verification
    def is_valid_fast(net):
        for x in range(65536):
            val = x
            for (i, j) in net:
                a = (val >> i) & 1
                b = (val >> j) & 1
                if a > b:
                    val ^= (1 << i) | (1 << j)
            if (val & (val + 1)) != 0:
                return False
        return True

    if not is_valid_fast(network):
        # Fallback to odd-even if Batcher generation failed
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    # 3. Local Search: Greedy Removal
    current_net = network[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order to avoid bias
        random.shuffle(current_net)
        for i in range(len(current_net)):
            if time.time() > deadline:
                break
            test_net = current_net[:i] + current_net[i+1:]
            if is_valid_fast(test_net):
                current_net = test_net
                improved = True
                break # Restart inner loop with new list
        
    return current_net
# EVOLVE-BLOCK-END
