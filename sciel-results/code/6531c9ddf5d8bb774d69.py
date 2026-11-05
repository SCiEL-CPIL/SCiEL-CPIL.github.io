import time
import random
import math

N = 16

# Precompute initial bitsets for zero-one principle verification
# state[i] is a 65536-bit integer where bit k is 1 if input k has a 1 in wire i
_INITIAL_BITSETS = []
for i in range(N):
    bitset = 0
    for k in range(1 << N):
        if k & (1 << i):
            bitset |= (1 << k)
    _INITIAL_BITSETS.append(bitset)

def _verify_zero_one_fast(net):
    """
    Fast verification using bit-slicing.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not net:
        return True
    
    state = _INITIAL_BITSETS[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    # Check if sorted: for all adjacent wires i, i+1, 
    # if wire i has a 1, wire i+1 must also have a 1 in all inputs.
    # state[i] & ~state[i+1] should be 0.
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    """
    A known 60-comparator sorting network for 16 wires.
    """
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _odd_even_network():
    """
    Odd-even transposition network: 120 comparators.
    """
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial best network
    best_net = _base_60_network()
    best_len = len(best_net)
    
    if not _verify_zero_one_fast(best_net):
        # Fallback to odd-even if base is broken (shouldn't happen)
        best_net = _odd_even_network()
        best_len = len(best_net)
        
    rng = random.Random(42)
    
    # Simulated Annealing parameters
    # We want to explore the space of networks.
    # State: a list of comparators.
    # Cost: number of comparators (if valid) or a large penalty (if invalid).
    # However, SA usually requires evaluating the "energy" of the state.
    # Here, "energy" is not continuous. We can define energy as:
    # If valid: len(net)
    # If invalid: len(net) + penalty * (number of violations)
    # But calculating violations is expensive.
    
    # Alternative: Accept only valid states, but use a "restart" mechanism
    # or a "neighborhood" that includes invalid states with a high rejection rate.
    # Actually, standard SA on combinatorial problems often uses:
    # 1. Generate neighbor.
    # 2. If neighbor is valid, accept if better, or accept if worse with prob exp(-delta/T).
    # 3. If neighbor is invalid, reject immediately (or accept with very low prob).
    
    # Let's try a hybrid: Local Search with Random Restarts and Perturbations.
    # This is often more effective for discrete structures like sorting networks.
    
    num_restarts = 20
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start of a new session
        if restart == 0:
            current_net = best_net[:]
        else:
            # Start from best found so far, but perturb it
            # Or start from a random valid network? Hard to find random valid.
            # Let's perturb the best network.
            current_net = best_net[:]
            # Randomly swap a few comparators
            for _ in range(5):
                if len(current_net) > 1:
                    i, j = rng.sample(range(len(current_net)), 2)
                    current_net[i], current_net[j] = current_net[j], current_net[i]
            # Randomly remove a few comparators (might make it invalid)
            # We will try to fix it or revert if it doesn't lead to improvement.
            # Actually, let's just start from best and do local search.
            
        current_len = len(current_net)
        local_best = current_net[:]
        local_best_len = current_len
        
        session_start = time.time()
        session_deadline = session_start + time_per_restart
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < session_deadline and no_improve_count < max_no_improve:
            improved = False
            moves = []
            
            # 1. Try removing a comparator
            if len(current_net) > 1:
                idx = rng.randrange(len(current_net))
                moves.append(('remove', idx))
                
            # 2. Try swapping two comparators
            if len(current_net) > 1:
                i, j = rng.sample(range(len(current_net)), 2)
                moves.append(('swap', i, j))
                
            # 3. Try inserting a comparator (to escape local minima)
            if rng.random() < 0.1: # Low probability of insertion
                i, j = rng.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = rng.randrange(len(current_net) + 1)
                moves.append(('insert', i, j, pos))
                
            # 4. Try changing one endpoint of a comparator
            if len(current_net) > 0:
                idx = rng.randrange(len(current_net))
                old_i, old_j = current_net[idx]
                # Change i
                new_i = rng.randrange(N)
                if new_i != old_i:
                    if new_i < old_j:
                        moves.append(('mutate', idx, new_i, old_j))
                    elif new_i > old_j:
                        moves.append(('mutate', idx, old_j, new_i))
                # Change j
                new_j = rng.randrange(N)
                if new_j != old_j:
                    if old_i < new_j:
                        moves.append(('mutate', idx, old_i, new_j))
                    elif old_i > new_j:
                        moves.append(('mutate', idx, new_j, old_i))

            if not moves:
                break
                
            # Pick a random move
            move = rng.choice(moves)
            
            new_net = current_net[:]
            
            if move[0] == 'remove':
                idx = move[1]
                new_net = current_net[:idx] + current_net[idx+1:]
            elif move[0] == 'swap':
                i, j = move[1], move[2]
                new_net = current_net[:]
                new_net[i], new_net[j] = new_net[j], new_net[i]
            elif move[0] == 'insert':
                i, j, pos = move[1], move[2], move[3]
                new_net = current_net[:pos] + [(i, j)] + current_net[pos:]
            elif move[0] == 'mutate':
                idx, new_i, new_j = move[1], move[2], move[3]
                new_net = current_net[:]
                new_net[idx] = (new_i, new_j)
                
            # Verify new network
            if _verify_zero_one_fast(new_net):
                new_len = len(new_net)
                # Accept if better, or equal (for diversity), or worse but small (to escape?)
                # For minimization, we only accept if new_len < current_len
                # But to escape local minima, we might need to accept worse states temporarily.
                # However, since we restart, we can just do strict improvement.
                if new_len < current_len:
                    current_net = new_net
                    current_len = new_len
                    improved = True
                    no_improve_count = 0
                    if current_len < local_best_len:
                        local_best = current_net[:]
                        local_best_len = current_len
                        if current_len < best_len:
                            best_net = current_net[:]
                            best_len = current_len
                elif new_len == current_len:
                    # Accept equal length to move around the plateau
                    if rng.random() < 0.5:
                        current_net = new_net
                        # Don't reset no_improve_count for equal length? 
                        # Maybe increment slowly.
                        no_improve_count += 1
                else:
                    # Worse, reject
                    no_improve_count += 1
            else:
                # Invalid, reject
                no_improve_count += 1
                
        # End of session
        if local_best_len < best_len:
            best_net = local_best
            best_len = local_best_len

    return best_net

if __name__ == "__main__":
    # Quick self-test
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print(f"Valid: {_verify_zero_one_fast(net)}")
