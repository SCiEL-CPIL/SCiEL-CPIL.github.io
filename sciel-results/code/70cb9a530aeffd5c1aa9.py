import time
import random

N = 17

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Pre-compute target masks for sorted output
    # T[i] has bit k set if input k has more than i ones
    targets = [0] * N
    # W_init[i] has bit k set if the i-th bit of k is 1
    W_init = [0] * N
    
    for k in range(1 << N):
        cnt = k.bit_count()
        for i in range(N):
            if cnt > i:
                targets[i] |= (1 << k)
            if (k >> i) & 1:
                W_init[i] |= (1 << k)

    def is_valid(comparators):
        W = W_init[:]
        for i, j in comparators:
            wi = W[i]
            wj = W[j]
            W[i] = wi & wj
            W[j] = wi | wj
        
        for i in range(N):
            if W[i] != targets[i]:
                return False
        return True

    # Seed: Odd-even transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    if not is_valid(seed_network):
        # Fallback: Should not happen
        return seed_network
        
    best_network = seed_network[:]
    best_size = len(best_network)
    
    def greedy_remove(net):
        improved = True
        while improved:
            improved = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    return net
                candidate = net[:idx] + net[idx+1:]
                if is_valid(candidate):
                    net = candidate
                    improved = True
                    break
        return net

    best_network = greedy_remove(best_network)
    best_size = len(best_network)
    
    current_network = best_network[:]
    current_size = best_size
    
    def shares_wire(c1, c2):
        return c1[0] == c2[0] or c1[0] == c2[1] or c1[1] == c2[0] or c1[1] == c2[1]

    # Local Search
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.8 and len(current_network) > 0:
            # Try removal
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx+1:]
            if is_valid(candidate):
                current_network = candidate
                current_size = len(current_network)
                if current_size < best_size:
                    best_network = current_network[:]
                    best_size = current_size
        else:
            # Try swap of adjacent conflicting comparators
            if len(current_network) > 1:
                # Pick a random pair of adjacent indices
                idx1 = random.randint(0, len(current_network) - 2)
                idx2 = idx1 + 1
                c1 = current_network[idx1]
                c2 = current_network[idx2]
                
                # Only swap if they share a wire, otherwise it's a no-op
                if shares_wire(c1, c2):
                    candidate = current_network[:]
                    candidate[idx1], candidate[idx2] = c2, c1
                    if is_valid(candidate):
                        # Accept to explore
                        if candidate != current_network:
                            current_network = candidate
                            # Size is same, so best_size unchanged
                            # But we might be in a better spot

    return best_network
