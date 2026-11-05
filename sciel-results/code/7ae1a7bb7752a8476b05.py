import numpy as np
import time
import random

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Setup Bit-Parallel State
    # We represent the state of all 65536 inputs for all 16 wires.
    # W is shape (N, MAX_INPUTS) of uint8.
    # W[w, k] is the value of wire w for input k (where k is 0..2^N-1).
    
    inputs = np.arange(MAX_INPUTS, dtype=np.uint64)
    W = np.empty((N, MAX_INPUTS), dtype=np.uint8)
    for w in range(N):
        W[w] = (inputs >> w) & 1
        
    # Precompute the "sorted" condition check?
    # A state is sorted if W[i, k] <= W[i+1, k] for all i, k.
    # We can compute a boolean mask of unsorted inputs.
    # unsorted_mask[k] = True if input k is not sorted.
    
    def apply_comparator(W_state, i, j):
        """
        Applies a comparator (i, j) to the bit-parallel state W_state.
        Returns the new state (or modifies in place if copy=True is handled by caller).
        For efficiency, we will do in-place updates on a copy.
        """
        a = W_state[i]
        b = W_state[j]
        # min(a, b) = a & b
        # max(a, b) = a | b
        W_state[i] = a & b
        W_state[j] = a | b

    def compute_cost(W_state):
        """
        Computes the number of unsorted inputs.
        Cost = sum_k (OR of (W[i,k] > W[i+1,k]) for i in 0..N-2)
        W[i,k] > W[i+1,k] is equivalent to (W[i,k] == 1) AND (W[i+1,k] == 0).
        So inv_mask = W[i] & (~W[i+1] & 1)
        total_unsorted = sum_k (any inv_mask over i)
        """
        # Vectorized check for inversions between adjacent wires
        # We need to know if ANY adjacent pair is inverted for each input k.
        # Let's compute the OR of all adjacent inversion masks.
        
        # To avoid creating too many intermediate large arrays, we can do this in a loop
        # or use a clever trick. 
        # inv_mask_total = zeros(MAX_INPUTS)
        # for i in range(N-1):
        #    inv_mask_total |= W_state[i] & (~W_state[i+1])
        
        # Since N is small (16), the loop is fine.
        unsorted = np.zeros(MAX_INPUTS, dtype=np.uint8)
        for i in range(N - 1):
            unsorted |= W_state[i] & (~W_state[i + 1])
            
        return np.sum(unsorted)

    def verify_network(net):
        """
        Verifies if the network sorts all inputs.
        Returns True if cost is 0.
        """
        test_W = W.copy()
        for comp in net:
            i, j = comp
            a = test_W[i]
            b = test_W[j]
            test_W[i] = a & b
            test_W[j] = a | b
        return compute_cost(test_W) == 0

    # 2. Initial Network: Odd-Even Transposition
    # This is a valid sorting network with 120 comparators.
    def get_odd_even_network():
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
        return net

    current_net = get_odd_even_network()
    current_cost = 0 # We know it's 0
    
    best_net = current_net.copy()
    best_cost = 0
    best_time = time.time()
    
    # 3. Simulated Annealing
    # State: current_net (list of tuples)
    # Cost: number of unsorted inputs (0 is optimal)
    # We want to minimize size (len(net)) subject to cost == 0.
    # However, SA usually minimizes a single objective.
    # We can use a combined objective: Cost + Lambda * Size.
    # Or, we can use a "repair" strategy: 
    # 1. Start with a valid network.
    # 2. Try to remove a random comparator.
    # 3. If cost > 0, try to add a random comparator to fix it?
    #    Or, accept the bad state with probability exp(-delta_E / T) where E includes cost penalty.
    
    # Let's use a penalty method:
    # Objective = Size + M * Cost
    # If we minimize this, we prefer smaller networks, but heavily penalize unsorted inputs.
    # As T decreases, we require Cost=0.
    
    M = 1000  # Penalty for unsorted inputs
    
    def get_objective(net):
        # This is expensive if we recompute cost from scratch every time.
        # But in SA, we need to evaluate neighbors.
        # Neighbor generation:
        # 1. Remove a comparator: new_net = net without k. Cost increases.
        # 2. Add a comparator: new_net = net with new (i,j). Cost decreases or stays same.
        # 3. Swap two comparators: Cost stays same (usually), Size stays same.
        
        # To be efficient, we should maintain the current state W_state.
        # But SA requires backtracking.
        # Given N=16 and MaxInputs=65536, W_state is 16*65536 bytes = 1MB.
        # Copying it is fast (~1ms).
        # Evaluating cost is ~16 * 65536 ops = 1M ops. Very fast in NumPy.
        # So we can afford to recompute cost for each neighbor.
        
        cost = 0
        if net:
            test_W = W.copy()
            for i, j in net:
                a = test_W[i]
                b = test_W[j]
                test_W[i] = a & b
                test_W[j] = a | b
            cost = compute_cost(test_W)
        else:
            # Empty network
            cost = compute_cost(W)
            
        return len(net) + M * cost

    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Remove a comparator
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net.copy()
            removed_comp = new_net.pop(idx)
            new_obj = get_objective(new_net)
            old_obj = get_objective(current_net) # This is wasteful, we should track it
            
            # Actually, let's track current_obj to avoid recomputing if we reject.
            # But we need to compute new_obj anyway.
            
            # To save time, let's pre-compute current_obj outside the loop?
            # No, because current_net changes.
            
            # Let's optimize get_objective by passing W_state if possible?
            # No, the structure of the network changes (length), so we must re-simulate.
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_net = new_net
                # We don't need to update global state variables if we recompute obj every time
        elif move_type < 0.8:
            # Add a comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Avoid duplicates? Not strictly necessary for SA, but good for size.
            # If (i,j) is already in net, adding it again does nothing to cost but increases size.
            # The objective penalizes size, so it will likely be rejected.
            
            new_net = current_net.copy()
            new_net.append((i, j))
            new_obj = get_objective(new_net)
            old_obj = get_objective(current_net)
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_net = new_net
        else:
            # Swap two comparators
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            new_net = current_net.copy()
            new_net[i], new_net[j] = new_net[j], new_net[i]
            new_obj = get_objective(new_net)
            old_obj = get_objective(current_net)
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_net = new_net
        
        # Update best
        # We only care about valid networks (cost == 0) for the final answer.
        # But during SA, we might explore invalid states.
        # We should track the best VALID network found.
        
        # Checking if current is valid and better than best
        # This is expensive. Let's do it periodically or only if size is small?
        # Actually, get_objective computes cost. If cost == 0, it's valid.
        # Let's modify get_objective to return (obj, cost).
        
        # Let's refactor to a helper that returns cost.
        
        # For now, let's just check every few steps if current_net is valid.
        # Or better:
        # If we accept a move, we have new_obj.
        # We need to know if new_cost == 0.
        
        # Let's rewrite the loop logic slightly to be more efficient.
        
        T *= cooling

    # After SA, we might be in an invalid state or a valid state.
    # We must return a valid network.
    # If best_net is valid, return it.
    # If current_net is valid and smaller than best_net, return current_net.
    
    # We need to ensure we have a valid network.
    # The initial one is valid.
    # If SA never found a smaller valid one, we return the initial one.
    
    # However, the above SA loop doesn't explicitly track "best valid".
    # Let's add that logic properly.
    
    # Re-implementing the loop with proper tracking for the final code block below.
    # The code above was a sketch. I will write the final clean version.
    
    return best_net

# EVOLVE-BLOCK-END

# Wait, I need to output the COMPLETE program. 
# I will refine the SA implementation to be robust and track the best valid solution.

def construct():
    start_time = time.time()
    
    inputs = np.arange(MAX_INPUTS, dtype=np.uint64)
    W = np.empty((N, MAX_INPUTS), dtype=np.uint8)
    for w in range(N):
        W[w] = (inputs >> w) & 1
        
    def compute_cost(W_state):
        unsorted = np.zeros(MAX_INPUTS, dtype=np.uint8)
        for i in range(N - 1):
            unsorted |= W_state[i] & (~W_state[i + 1])
        return np.sum(unsorted)

    def get_cost(net):
        if not net:
            return compute_cost(W)
        test_W = W.copy()
        for i, j in net:
            a = test_W[i]
            b = test_W[j]
            test_W[i] = a & b
            test_W[j] = a | b
        return compute_cost(test_W)

    def get_odd_even_network():
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
        return net

    current_net = get_odd_even_network()
    current_cost = 0
    best_net = current_net.copy()
    best_cost = 0
    
    T = 50.0
    T_min = 0.5
    cooling = 0.999
    
    M = 1000 # Penalty
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.7:
            # Remove
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net.copy()
            new_net.pop(idx)
            new_cost = get_cost(new_net)
            new_obj = len(new_net) + M * new_cost
            old_obj = len(current_net) + M * current_cost
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_net = new_net
                current_cost = new_cost
        else:
            # Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net = current_net.copy()
            new_net.append((i, j))
            new_cost = get_cost(new_net)
            new_obj = len(new_net) + M * new_cost
            old_obj = len(current_net) + M * current_cost
            
            delta = new_obj - old_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_net = new_net
                current_cost = new_cost
                
        # Update best valid
        if current_cost == 0 and len(current_net) < len(best_net):
            best_net = current_net.copy()
            best_cost = 0
            
        T *= cooling

    # Final verification
    if get_cost(best_net) != 0:
        # Fallback to odd-even
        best_net = get_odd_even_network()
        
    return best_net
