import time

N = 17

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate Initial Network (Odd-Even Transposition)
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Pre-compute Initial Bitmasks for all 2^N inputs
    # Wire k has bit m set if the k-th bit of m is 1.
    # m ranges from 0 to 2^N - 1.
    initial_masks = [0] * N
    for m in range(1 << N):
        for k in range(N):
            if (m >> k) & 1:
                initial_masks[k] |= (1 << m)
                
    # 3. Verification Function
    def is_valid(net):
        # Copy masks
        masks = initial_masks[:]
        for i, j in net:
            # min to i, max to j
            # min(a, b) = a & b
            # max(a, b) = a | b
            new_i = masks[i] & masks[j]
            new_j = masks[i] | masks[j]
            masks[i] = new_i
            masks[j] = new_j
            
        # Check if sorted
        # For all k, if mask[k] has a 1, mask[k+1] must have a 1.
        # i.e., mask[k] & ~mask[k+1] == 0
        for k in range(N - 1):
            if (masks[k] & ~masks[k+1]) != 0:
                return False
        return True

    # 4. Greedy Pruning
    # We try to remove comparators.
    # Start from the end, as later stages are often redundant in transposition networks.
    changed = True
    while changed:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            # Temporarily remove
            removed = network.pop(idx)
            if is_valid(network):
                changed = True
                break
            else:
                # Restore
                network.insert(idx, removed)
                
    return network
# EVOLVE-BLOCK-END
