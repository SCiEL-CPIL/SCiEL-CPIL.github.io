import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # Precompute the initial state masks for all 2^N inputs
    # state[j] is a bitmask of length 2^N, where bit k is 1 if input k has a 1 on wire j
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
        
        # Check if sorted: state[k] should be a subset of state[k+1]
        # i.e., state[k] & ~state[k+1] should be 0
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Stagnation tracking
    stagnation_count = 0
    STAGNATION_THRESHOLD = 200
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995

    def greedy_remove(current_net):
        """
        Greedily removes redundant comparators.
        A comparator is redundant if, given the state before it, the values are already sorted (i <= j).
        We simulate the network, and if a comparator is a no-op, we mark it for removal.
        However, removing one might make another redundant. So we iterate.
        To be safe and fast, we do a single pass, removing any no-ops found.
        """
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            new_net = []
            state = initial_state[:]
            
            for idx, (i, j) in enumerate(current_net):
                # Check if this comparator is a no-op
                # If state[i] is a subset of state[j], then min is state[i] and max is state[j].
                # The comparator does nothing.
                if (state[i] & ~state[j]) == 0:
                    # It's a no-op. We can skip it (remove it).
                    removed = True
                    # We do NOT update state because it's a no-op.
                else:
                    # It's an active comparator. Keep it and update state.
                    new_net.append((i, j))
                    new_i = state[i] & state[j]
                    new_j = state[i] | state[j]
                    state[i] = new_i
                    state[j] = new_j
            
            # After the pass, we must verify that the reduced network is still valid.
            # It's possible that removing a no-op comparator earlier in the sequence
            # changes the state for subsequent comparators in a way that breaks the sort?
            # No, if a comparator is a no-op, it doesn't change the state.
            # So the state after the no-op is identical to the state before it.
            # Removing it does not change the state for any subsequent comparator.
            # Therefore, the resulting network is guaranteed to be valid if the original was valid.
            
            current_net = new_net
        
        return current_net

    def grow_and_shrink(current_net, num_adds):
        """Add random comparators, then aggressively remove redundancies."""
        candidate = current_net[:]
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return None
        
        optimized = greedy_remove(candidate)
        return optimized

    def swap_comparators(current_net):
        """Swap two comparators in the network."""
        if len(current_net) < 2:
            return None
        
        # Choose two indices to swap
        # Prefer swapping close indices to maintain local structure, but allow global swaps too
        idx1 = random.randint(0, len(current_net) - 1)
        idx2 = random.randint(0, len(current_net) - 1)
        if idx1 == idx2:
            return None
        
        candidate = current_net[:]
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        
        if is_valid(candidate):
            return candidate
        return None

    def kick_stagnation(current_net):
        """When stuck, add 2-4 comparators at random positions to escape local minima."""
        candidate = current_net[:]
        num_adds = random.choice([2, 3, 4])
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return None
        
        optimized = greedy_remove(candidate)
        return optimized

    def final_polish(current_net):
        """Final high-precision polish: try sequential removal from end to start for thorough local optimum."""
        if time.time() > deadline:
            return current_net
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_net = current_net[:]
            
            # Try removing each comparator from end to start
            for idx in range(len(new_net) - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                # Try removing comparator at idx
                candidate = new_net[:idx] + new_net[idx+1:]
                
                if is_valid(candidate):
                    new_net = candidate
                    improved = True
                    break  # Restart from beginning since network changed
            
            if improved:
                current_net = new_net
        
        return current_net

    # Main Search Loop
    while time.time() < deadline:
        # 1. Aggressively reduce current network
        optimized = greedy_remove(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        if time.time() > deadline:
            break

        # 2. Mutation Phase
        r = random.random()
        
        # If stuck, use stronger kicks
        if stagnation_count > STAGNATION_THRESHOLD:
            new_net = kick_stagnation(current_network)
            stagnation_count = 0
        elif r < 0.3:
            # Add and Shrink
            num_adds = random.choice([1, 2, 3])
            new_net = grow_and_shrink(current_network, num_adds)
        elif r < 0.6:
            # Swap
            new_net = swap_comparators(current_network)
        else:
            # Add and Shrink with larger growth
            num_adds = random.choice([2, 3, 4, 5])
            new_net = grow_and_shrink(current_network, num_adds)
        
        if new_net is not None:
            new_cost = len(new_net)
            delta = new_cost - current_cost
            
            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / T)
                accept = random.random() < accept_prob
            
            if accept:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
                    stagnation_count = 0
        
        # Cool down
        T = max(T * cooling_rate, T_min)

    # 3. Final polish pass
    polished = final_polish(best_network)
    if len(polished) < best_cost:
        best_network = polished
        best_cost = len(best_network)

    return best_network
# EVOLVE-BLOCK-END
