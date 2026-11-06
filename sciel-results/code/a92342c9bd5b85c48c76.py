import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

# Precompute full matrix of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _simulate(net):
    """
    Simulate the network and return the list of states after each comparator.
    states[k] is the state of wires after applying net[k].
    states has length len(net).
    """
    states = []
    W = W0_full.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        states.append(W.copy())
    return states

def _is_redundant(net, k, states):
    """
    Check if comparator at index k is redundant.
    states[k-1] is the state before applying net[k].
    """
    if k == 0:
        W_pre = W0_full
    else:
        W_pre = states[k-1]
    
    i, j = net[k]
    # Check if W_pre[:, i] <= W_pre[:, j] for all inputs
    # This is equivalent to: no input has 1 in i and 0 in j
    # i.e., (W_pre[:, i] & ~W_pre[:, j]).any() is False
    # Since values are 0/1, ~W_pre[:, j] is 1-W_pre[:, j]
    # Or simply: (W_pre[:, i] > W_pre[:, j]).any()
    return not np.any(W_pre[:, i] > W_pre[:, j])

def _prune(net):
    """
    Iteratively prune redundant comparators.
    """
    changed = True
    while changed:
        changed = False
        states = _simulate(net)
        # Scan from last to first
        for k in range(len(net) - 1, -1, -1):
            if _is_redundant(net, k, states):
                # Remove comparator k
                net = net[:k] + net[k+1:]
                changed = True
                break
    return net

def _verify_full(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    """
    if len(net) > MAX_COMPARATORS:
        return False
    if len(net) == 0:
        return True
    
    W = W0_full.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition(N)
    
    # Phase 1: Structural Pruning
    if time.time() < deadline - 10:
        net = _prune(net)
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 5:
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
            
            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.7:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Verify
            if not _verify_full(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
        
    return best_net
