import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin for 90s limit

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    best_len = len(best_network)
    
    # 2. Verification Function using bit-vector simulation
    # Precompute the initial bit-vectors for each wire
    # state[j] is a bitmask of length NUM_INPUTS where bit k is 1 if input k has 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check if sorted: wire k should be subset of wire k+1 (all 1s in k are also in k+1)
        # i.e., state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Greedy Removal Strategy:
    # The most effective way to reduce size is to repeatedly try removing comparators.
    # We do this in a "sweep" fashion, but randomized to avoid local minima.
    
    def try_remove_one(current_net):
        """Try to remove a random comparator. Return new net if valid, else None."""
        if len(current_net) == 0:
            return None
        idx = random.randint(0, len(current_net) - 1)
        candidate = current_net[:idx] + current_net[idx+1:]
        if is_valid(candidate):
            return candidate
        return None

    def greedy_shrink(current_net, max_iters=None):
        """Aggressively remove redundant comparators."""
        if max_iters is None:
            max_iters = len(current_net) * 5
        
        removed = 0
        while removed < max_iters:
            if time.time() > deadline:
                break
            new_net = try_remove_one(current_net)
            if new_net is not None:
                current_net = new_net
                removed += 1
            else:
                # If we fail to remove, try a few more random attempts before giving up on this pass
                # Actually, a single failure doesn't mean no more can be removed.
                # But for efficiency, we might want to restart the removal process or shuffle.
                # Let's just continue trying random indices.
                pass
        
        return current_net

    # Strategy: 
    # 1. Shrink the seed as much as possible.
    # 2. Then, use a "Grow-Shrink" or "Perturb-Shrink" loop.
    #    - Perturb: Add a random comparator or swap two comparators.
    #    - Shrink: Greedily remove as many as possible.
    #    - If the result is smaller than current best, accept.
    #    - If not, maybe keep it as current state if it's not too bad (simulated annealing style)
    #      or just discard. Given the time limit, simple hill climbing with random restarts
    #      or perturbations might be best.

    current_network = best_network[:]
    current_network = greedy_shrink(current_network)
    if len(current_network) < best_len:
        best_network = current_network[:]
        best_len = len(best_network)

    # Main Search Loop
    # We will alternate between:
    # A) Random removal attempts (local search)
    # B) Add a comparator, then shrink (escape local minima)
    # C) Swap two comparators, then shrink
    
    # Temperature for simulated annealing-like acceptance
    temp = 10.0
    min_temp = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        strategy = random.random()
        
        if strategy < 0.3:
            # Pure removal attempt
            new_net = try_remove_one(current_network)
            if new_net is not None:
                current_network = new_net
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(best_network)
                    # Reset temp on improvement?
                    temp = max(temp, 5.0)
            else:
                # No removal found, maybe cool down
                temp *= cooling
                
        elif strategy < 0.6:
            # Add a random comparator, then shrink
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                temp *= cooling
                continue
                
            pos = random.randint(0, len(current_network))
            candidate = current_network[:]
            candidate.insert(pos, (new_i, new_j))
            
            # Shrink the candidate
            candidate = greedy_shrink(candidate)
            
            # Acceptance criterion
            if len(candidate) < best_len:
                best_network = candidate[:]
                best_len = len(best_network)
                current_network = candidate[:]
                temp = max(temp, 5.0)
            elif len(candidate) <= len(current_network) or random.random() < math.exp(-(len(candidate) - len(current_network)) / max(temp, 0.1)):
                current_network = candidate[:]
                
            temp *= cooling

        elif strategy < 0.8:
            # Swap two comparators, then shrink
            if len(current_network) < 2:
                temp *= cooling
                continue
                
            i = random.randint(0, len(current_network) - 1)
            j = random.randint(0, len(current_network) - 1)
            if i == j:
                temp *= cooling
                continue
                
            candidate = current_network[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            candidate = greedy_shrink(candidate)
            
            if len(candidate) < best_len:
                best_network = candidate[:]
                best_len = len(best_network)
                current_network = candidate[:]
                temp = max(temp, 5.0)
            elif len(candidate) <= len(current_network) or random.random() < math.exp(-(len(candidate) - len(current_network)) / max(temp, 0.1)):
                current_network = candidate[:]
                
            temp *= cooling

        else:
            # Move a comparator to a new position (reorder), then shrink
            if len(current_network) < 1:
                temp *= cooling
                continue
                
            i = random.randint(0, len(current_network) - 1)
            j = random.randint(0, len(current_network) - 1)
            if i == j:
                temp *= cooling
                continue
                
            candidate = current_network[:]
            comp = candidate.pop(i)
            candidate.insert(j, comp)
            
            candidate = greedy_shrink(candidate)
            
            if len(candidate) < best_len:
                best_network = candidate[:]
                best_len = len(best_network)
                current_network = candidate[:]
                temp = max(temp, 5.0)
            elif len(candidate) <= len(current_network) or random.random() < math.exp(-(len(candidate) - len(current_network)) / max(temp, 0.1)):
                current_network = candidate[:]
                
            temp *= cooling

    # Final polish
    if time.time() < deadline:
        polished = greedy_shrink(best_network)
        if len(polished) < len(best_network):
            best_network = polished

    return best_network
# EVOLVE-BLOCK-END
