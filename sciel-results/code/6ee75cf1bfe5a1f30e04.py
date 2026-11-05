import numpy as np
import time
import random
from itertools import product

N = 18
SIZE = 1 << N  # 262144

def _is_sorted(states):
    """Check if all 2^N inputs are sorted."""
    return not np.any(states[:-1] < states[1:])

def _apply_comparator(states, i, j):
    """Apply comparator (i, j) to states. Returns saved values for restoration."""
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _restore_comparator(states, i, j, si, sj):
    """Restore states after a comparator was applied."""
    states[i] = si
    states[j] = sj

def _seed_network():
    """Generate odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    """
    Constructs a sorting network using a combination of:
    1. Starting from odd-even transposition
    2. Aggressive greedy removal from both ends
    3. Targeted replacement moves (replace a comparator with nearby ones)
    4. Random local search with swap and removal
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    random.seed(12345)
    
    # Initialize states for all 2^N inputs
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    # Start with seed network
    current_network = _seed_network()
    best_network = current_network[:]
    best_len = len(current_network)
    
    # Aggressive greedy removal - try removing from end to start, then start to end
    def greedy_remove(network, states, direction='end'):
        """Try to remove comparators greedily."""
        net = network[:]
        removed = True
        while removed:
            removed = False
            if direction == 'end':
                indices = range(len(net) - 1, -1, -1)
            else:
                indices = range(len(net))
            for idx in indices:
                i, j = net[idx]
                si, sj = states[i].copy(), states[j].copy()
                states[i] = si & sj
                states[j] = si | sj
                if _is_sorted(states):
                    net.pop(idx)
                    removed = True
                else:
                    states[i] = si
                    states[j] = sj
        return net
    
    # Apply greedy removal from both ends
    current_network = greedy_remove(current_network, states, 'end')
    current_network = greedy_remove(current_network, states, 'start')
    if len(current_network) < best_len:
        best_network = current_network[:]
        best_len = len(current_network)
    
    # Now do randomized local search with targeted moves
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Try different move types with bias toward removal and targeted replacement
        move_type = random.choices(
            ['remove', 'targeted_replace', 'swap', 'insert_nearby'],
            weights=[0.3, 0.3, 0.2, 0.2]
        )[0]
        
        if move_type == 'remove':
            if len(current_network) <= best_len:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i, j = current_network[idx]
            
            si, sj = states[i].copy(), states[j].copy()
            states[i] = si & sj
            states[j] = si | sj
            
            if _is_sorted(states):
                current_network.pop(idx)
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(current_network)
            else:
                states[i] = si
                states[j] = sj
        
        elif move_type == 'targeted_replace':
            # Replace a comparator with one nearby (within 2-3 wires)
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            old_i, old_j = current_network[idx]
            
            # Generate candidate replacement: swap or nearby comparator
            candidates = []
            for offset in range(-2, 3):
                new_i = old_i + offset
                new_j = old_j + offset
                if 0 <= new_i < N - 1 and new_i + 1 <= new_j < N and new_i != old_i or new_j != old_j:
                    candidates.append((new_i, new_j))
            
            # Also try swapping the comparator
            if old_j - old_i > 1:
                candidates.append((old_i, old_j - 1))
                candidates.append((old_i + 1, old_j))
            
            random.shuffle(candidates)
            
            for ci, cj in candidates:
                if (ci, cj) == (old_i, old_j):
                    continue
                
                # Save state
                si_old, sj_old = states[old_i].copy(), states[old_j].copy()
                si_new, sj_new = states[ci].copy(), states[cj].copy()
                
                # Apply old removal and new addition
                states[old_i] = si_old
                states[old_j] = sj_old
                states[ci] = si_new & sj_new
                states[cj] = si_new | sj_new
                
                if _is_sorted(states):
                    # Valid move
                    current_network[idx] = (ci, cj)
                    if len(current_network) < best_len:
                        best_network = current_network[:]
                        best_len = len(current_network)
                    break
                else:
                    # Revert
                    states[old_i] = si_old
                    states[old_j] = sj_old
                    states[ci] = si_new
                    states[cj] = sj_new
        
        elif move_type == 'swap':
            if len(current_network) < 2:
                continue
            i1, i2 = random.sample(range(len(current_network)), 2)
            
            # Save state before swap
            net_copy = current_network[:]
            
            # Perform swap
            current_network[i1], current_network[i2] = current_network[i2], current_network[i1]
            
            # Verify from scratch
            temp_states = states.copy()
            for ci, cj in current_network:
                temp_si = temp_states[ci].copy()
                temp_sj = temp_states[cj].copy()
                temp_states[ci] = temp_si & temp_sj
                temp_states[cj] = temp_si | temp_sj
            
            if _is_sorted(temp_states):
                if len(current_network) < best_len:
                    best_network = current_network[:]
                    best_len = len(current_network)
            else:
                # Revert swap
                current_network[i1], current_network[i2] = net_copy[i1], net_copy[i2]
        
        elif move_type == 'insert_nearby':
            # Insert a comparator near an existing one, then try to remove something
            if len(current_network) == 0 or len(current_network) >= 4000:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i, j = current_network[idx]
            
            # Try inserting a nearby comparator
            candidates = []
            for offset in range(-2, 3):
                new_i = i + offset
                new_j = j + offset
                if 0 <= new_i < N - 1 and new_i + 1 <= new_j < N:
                    candidates.append((new_i, new_j))
            
            random.shuffle(candidates)
            
            for ci, cj in candidates:
                # Skip if already present
                if (ci, cj) in current_network:
                    continue
                
                pos = idx + random.randint(-1, 1)
                pos = max(0, min(pos, len(current_network)))
                
                new_network = current_network[:]
                new_network.insert(pos, (ci, cj))
                
                # Verify
                temp_states = states.copy()
                for net_i, net_j in new_network:
                    temp_si = temp_states[net_i].copy()
                    temp_sj = temp_states[net_j].copy()
                    temp_states[net_i] = temp_si & temp_sj
                    temp_states[net_j] = temp_si | temp_sj
                
                if _is_sorted(temp_states):
                    current_network = new_network
                    # Now try greedy removal to see if we can remove something else
                    temp_net = greedy_remove(current_network, temp_states, 'end')
                    if len(temp_net) < best_len:
                        best_network = temp_net[:]
                        best_len = len(temp_net)
                        current_network = temp_net[:]
                        states = temp_states
                    break
    
    # Final greedy pass
    if time.time() < deadline - 5:
        current_network = greedy_remove(current_network, states, 'end')
        current_network = greedy_remove(current_network, states, 'start')
        if len(current_network) < best_len:
            best_network = current_network[:]
            best_len = len(current_network)
    
    return best_network
