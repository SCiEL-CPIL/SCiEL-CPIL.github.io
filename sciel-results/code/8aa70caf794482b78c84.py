import numpy as np
import time
import random

N = 18
M = 2**N

def generate_initial_states():
    idx = np.arange(M, dtype=np.uint32)
    states = np.zeros((N, M), dtype=np.uint8)
    for i in range(N):
        states[i] = (idx >> i) & 1
    return states

INITIAL_STATES = generate_initial_states()

def verify_network(net):
    states = INITIAL_STATES.copy()
    for i, j in net:
        a = states[i]
        b = states[j]
        min_ab = np.minimum(a, b)
        max_ab = np.maximum(a, b)
        states[i] = min_ab
        states[j] = max_ab
        
    diffs = np.diff(states, axis=0)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial network: Odd-Even Transposition
    current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    if not verify_network(current_net):
        return current_net

    best_net = current_net.copy()
    best_size = len(current_net)
    
    stuck_counter = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Removals
        # Iterate backwards
        for k in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline - 2.0: break
            candidate = current_net[:k] + current_net[k+1:]
            if verify_network(candidate):
                current_net = candidate
                improved = True
                break
        
        if improved:
            stuck_counter = 0
            if len(current_net) < best_size:
                best_net = current_net.copy()
                best_size = len(current_net)
            continue
        
        # 2. Try Moves
        if len(current_net) > 0:
            k = random.randint(0, len(current_net) - 1)
            comp = current_net[k]
            temp_net = current_net[:k] + current_net[k+1:]
            new_pos = random.randint(0, len(temp_net))
            candidate = temp_net[:new_pos] + [comp] + temp_net[new_pos:]
            
            if verify_network(candidate):
                current_net = candidate
                improved = True
                stuck_counter = 0
                continue
        
        # 3. Try Swapping Indices
        if len(current_net) > 0:
            k = random.randint(0, len(current_net) - 1)
            i, j = current_net[k]
            # Try changing j
            candidates = [x for x in range(N) if x != i]
            if candidates:
                new_j = random.choice(candidates)
                if new_j != j:
                    candidate_net = current_net.copy()
                    candidate_net[k] = (i, new_j)
                    if verify_network(candidate_net):
                        current_net = candidate_net
                        improved = True
                        stuck_counter = 0
                        continue
                        
            # Try changing i
            candidates = [x for x in range(N) if x != j]
            if candidates:
                new_i = random.choice(candidates)
                if new_i < j and new_i != i:
                    candidate_net = current_net.copy()
                    candidate_net[k] = (new_i, j)
                    if verify_network(candidate_net):
                        current_net = candidate_net
                        improved = True
                        stuck_counter = 0
                        continue
        
        # 4. If stuck, try adding a comparator
        if not improved:
            stuck_counter += 1
            # If stuck for a while, or if we are much larger than best, reset
            if stuck_counter > 50 or len(current_net) > best_size + 20:
                current_net = best_net.copy()
                stuck_counter = 0
                # Maybe perturb the best net slightly?
                # For now, just restart search from best
                continue
            
            # Add random comparator
            i = random.randint(0, N-2)
            j = random.randint(i+1, N-1)
            candidate = current_net + [(i, j)]
            if verify_network(candidate):
                current_net = candidate
                # Do not reset stuck_counter immediately, as we haven't improved size
                # But we did change state
                continue

    return best_net
