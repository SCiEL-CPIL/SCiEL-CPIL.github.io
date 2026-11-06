import time
import numpy as np

N = 18
DEADLINE = 85.0 # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # 1. Generate all 2^18 boolean inputs
    # Shape: (262144, 18)
    # We use uint8 for memory efficiency and speed
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1

    def is_valid(network):
        """
        Check if the network sorts all inputs.
        network: list of (i, j) tuples
        """
        if not network:
            return False # Empty network only valid if N<=1, but N=18
        
        state = all_inputs.copy()
        for i, j in network:
            # min goes to i, max goes to j
            # np.minimum and np.maximum are vectorized
            state[:, i], state[:, j] = np.minimum(state[:, i], state[:, j]), np.maximum(state[:, i], state[:, j])
        
        # Check if all rows are sorted
        # state[:, :-1] <= state[:, 1:]
        return np.all(state[:, :-1] <= state[:, 1:])

    # 2. Initial Network: Odd-Even Transposition
    # Phases 0 to N-1
    # Even phases (0, 2, ...): comparators (0,1), (2,3), ...
    # Odd phases (1, 3, ...): comparators (1,2), (3,4), ...
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network
    if not is_valid(network):
        # Should not happen, but fallback
        return network

    current_size = len(network)
    best_network = network.copy()
    best_size = current_size
    
    # 3. Greedy Removal
    # We repeatedly try to remove comparators.
    # To be efficient, we iterate through indices. If a removal is successful, we restart the index loop
    # because the list has changed.
    
    changed = True
    while changed and time.time() < DEADLINE - 10:
        changed = False
        # Randomize order to avoid bias? Or sequential?
        # Sequential is deterministic but might get stuck in specific patterns.
        # Random order is better for finding different removals.
        indices = np.random.permutation(len(network))
        for idx in indices:
            if time.time() > DEADLINE - 10:
                break
            # Try removing network[idx]
            new_network = network[:idx] + network[idx+1:]
            if is_valid(new_network):
                network = new_network
                current_size -= 1
                changed = True
                # Update best if improved
                if current_size < best_size:
                    best_size = current_size
                    best_network = network.copy()
                # Restart the permutation loop? 
                # Breaking and restarting the while loop is safer to handle index shifts.
                break

    # 4. Local Search (Simulated Annealing / Hill Climbing)
    # We are now at a local minimum where no single removal works.
    # We need to move to a different local minimum.
    # Moves:
    # - Swap two comparators
    # - Move one comparator to a different position
    # - (Optional) Replace one comparator with another (risky, might break validity)
    
    # We will use a simple hill-climbing with random restarts of the "move selection"
    # or just random valid moves.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    current_network = network
    current_size = len(current_network)
    
    no_improve_count = 0
    
    while time.time() < DEADLINE:
        if no_improve_count > 200:
            # We are stuck in a local minimum.
            # Perturb the network slightly to escape.
            # Strategy: Pick a random comparator and swap it with another random one.
            # Or, pick a random comparator and move it to a random position.
            # We accept the perturbation ONLY if it is valid.
            # If we can't find a valid perturbation, we might be in a very tight spot.
            
            # Try to find a valid swap
            found_perturbation = False
            for _ in range(50): # Try a few random swaps
                if len(current_network) < 2:
                    break
                i1, i2 = np.random.choice(len(current_network), 2, replace=False)
                # Perform swap
                current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
                
                if is_valid(current_network):
                    found_perturbation = True
                    no_improve_count = 0
                    # Don't update best size, size is same
                    break
                else:
                    # Revert
                    current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
            
            if not found_perturbation:
                # Try moving a comparator
                if len(current_network) > 0:
                    idx = np.random.randint(0, len(current_network))
                    comp = current_network[idx]
                    new_idx = np.random.randint(0, len(current_network))
                    # Remove and insert
                    current_network.pop(idx)
                    current_network.insert(new_idx, comp)
                    if is_valid(current_network):
                        no_improve_count = 0
                        found_perturbation = True
                    else:
                        # Revert
                        current_network.pop(new_idx)
                        current_network.insert(idx, comp)
                
            if not found_perturbation:
                # Stuck hard. Cool down and continue, or restart greedy removal?
                # Restarting greedy removal on the current network might find new removals if the structure changed?
                # But structure didn't change if we couldn't perturb.
                # Let's just cool down.
                T = max(T * 0.9, T_min)
                no_improve_count = 0 # Reset to avoid infinite loop of "stuck" checks

        # Choose a move
        # Bias towards removals if possible, but we already did greedy.
        # So we mostly do swaps/moves to explore.
        
        move_type = np.random.choice(['swap', 'move'], p=[0.7, 0.3])
        
        if move_type == 'swap':
            if len(current_network) < 2:
                continue
            i1, i2 = np.random.choice(len(current_network), 2, replace=False)
            current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
            
            if is_valid(current_network):
                # Valid move, size unchanged
                no_improve_count = 0
            else:
                # Revert
                current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
                no_improve_count += 1
                
        elif move_type == 'move':
            if len(current_network) < 2:
                continue
            i1 = np.random.randint(0, len(current_network))
            i2 = np.random.randint(0, len(current_network))
            if i1 == i2:
                continue
            comp = current_network[i1]
            current_network.pop(i1)
            current_network.insert(i2, comp)
            
            if is_valid(current_network):
                no_improve_count = 0
            else:
                # Revert
                current_network.pop(i2)
                current_network.insert(i1, comp)
                no_improve_count += 1
                
        # After a move, try to remove again?
        # This is crucial: a swap might enable a removal.
        # Let's do a quick greedy removal pass after every few successful moves.
        if no_improve_count == 0:
             # Try to remove one random comparator
             if len(current_network) > 0:
                 idx = np.random.randint(0, len(current_network))
                 new_network = current_network[:idx] + current_network[idx+1:]
                 if is_valid(new_network):
                     current_network = new_network
                     current_size -= 1
                     if current_size < best_size:
                         best_size = current_size
                         best_network = current_network.copy()
                     # Reset no_improve
                     no_improve_count = 0
                     # Continue loop to try more removals
                     continue

        # Cool down
        T *= alpha

    return best_network
