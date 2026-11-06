import time
import random

N = 17
NUM_INPUTS = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Initial network: Odd-Even Transposition
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    # Precompute targets
    # target[w] is a bitmask of length NUM_INPUTS
    # bit k is 1 if wire w is 1 for input k
    targets = [0] * N
    for k in range(NUM_INPUTS):
        # Count bits in k
        c = bin(k).count('1')
        # Ones are at the end (high indices)
        # Indices N-c to N-1 are 1
        for w in range(N - c, N):
            targets[w] |= (1 << k)
            
    # Check validity
    def is_valid(net):
        # state[w] is bitmask
        state = [0] * N
        for k in range(NUM_INPUTS):
            # Set initial state for input k
            # We can do this faster by iterating k and setting bits, 
            # but doing it per comparator is better.
            pass
        
        # Better: Initialize state for all k at once?
        # state[w] for input k is bit k of w.
        # So state[w] initially is just the w-th bit of the input vector for all k.
        # Input k is an integer. Bit w of k is (k >> w) & 1.
        # We want state[w] to have bit k = (k >> w) & 1.
        # This is equivalent to constructing an integer where bit k is set if bit w of k is set.
        
        # Let's build initial state array
        state = [0] * N
        for w in range(N):
            mask = 0
            for k in range(NUM_INPUTS):
                if (k >> w) & 1:
                    mask |= (1 << k)
            state[w] = mask
            
        for i, j in net:
            # min to i, max to j
            # new_i = state[i] & state[j]
            # new_j = state[i] | state[j]
            si = state[i]
            sj = state[j]
            state[i] = si & sj
            state[j] = si | sj
            
        for w in range(N):
            if state[w] != targets[w]:
                return False
        return True

    # Optimization
    best_net = net
    best_len = len(net)
    
    # We will use a loop that tries random moves
    # Moves: Delete, Swap, Move
    # We prioritize Delete.
    
    # To speed up, we can cache validity? No, network changes.
    
    # We'll run a simple hill climbing / simulated annealing hybrid.
    # Since we want to minimize size, we only accept if size decreases.
    # To escape plateaus, we might need to accept same size (swap/move) to enable future deletions.
    
    current_net = list(net)
    
    while time.time() < deadline:
        improved = False
        # Try deletions
        # Randomly shuffle indices to avoid bias
        indices = list(range(len(current_net)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            # Try removing idx
            new_net = current_net[:idx] + current_net[idx+1:]
            if is_valid(new_net):
                current_net = new_net
                improved = True
                if len(current_net) < best_len:
                    best_net = list(current_net)
                    best_len = len(current_net)
                break # Restart deletion loop with new length
                
        if improved:
            continue
            
        # If no deletion, try swaps or moves
        # Try a few random swaps
        if len(current_net) > 1:
            for _ in range(10): # Limit attempts per iteration to avoid hanging
                if time.time() > deadline:
                    break
                i, j = random.sample(range(len(current_net)), 2)
                # Swap i and j
                current_net[i], current_net[j] = current_net[j], current_net[i]
                if is_valid(current_net):
                    # Accepted, but size same. 
                    # This helps escape local minima for deletion.
                    pass
                else:
                    # Revert
                    current_net[i], current_net[j] = current_net[j], current_net[i]
                    
        # Try a move (remove and insert)
        if len(current_net) > 2:
            for _ in range(10):
                if time.time() > deadline:
                    break
                i = random.randint(0, len(current_net)-1)
                comp = current_net.pop(i)
                j = random.randint(0, len(current_net))
                current_net.insert(j, comp)
                if is_valid(current_net):
                    pass
                else:
                    current_net.pop(j)
                    current_net.insert(i, comp)

    return best_net
