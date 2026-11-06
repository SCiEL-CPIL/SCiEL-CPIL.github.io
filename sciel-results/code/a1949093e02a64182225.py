import time
import random

N = 17
TOTAL_INPUTS = 1 << N

def _build_initial_masks():
    # Test input k has k ones at the left (indices 0..k-1)
    # Wire i has 1 in test input k if i < k
    # So bit k in mask[i] is 1 if i < k
    masks = []
    for i in range(N):
        m = 0
        for k in range(N):
            if i < k:
                m |= (1 << k)
        masks.append(m)
    return masks

def _build_target_masks():
    # Sorted output for test input k has k ones at the right (indices 17-k..16)
    # Wire i has 1 in sorted output k if i >= 17-k
    # So bit k in target[i] is 1 if i >= 17-k
    targets = []
    for i in range(N):
        m = 0
        for k in range(N):
            if i >= 17 - k:
                m |= (1 << k)
        targets.append(m)
    return targets

TARGET_MASKS = _build_target_masks()

def _is_sorted(masks):
    # Check if current masks match the target sorted masks
    return masks == TARGET_MASKS

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin

    # 1. Seed: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    initial_masks = _build_initial_masks()
    
    # Helper to check if network is valid
    def check_network(net):
        masks = initial_masks[:]
        for i, j in net:
            mi = masks[i]
            mj = masks[j]
            masks[i] = mi & mj
            masks[j] = mi | mj
        return _is_sorted(masks)
    
    def simulate(net):
        """Returns the final masks after applying net to initial_masks"""
        masks = initial_masks[:]
        for i, j in net:
            mi = masks[i]
            mj = masks[j]
            masks[i] = mi & mj
            masks[j] = mi | mj
        return masks

    # Helper to check if comparator k is redundant
    # Given states S_0 ... S_C
    # Redundancy of k: Apply C_{k+1}...C_{end} to S_k, check sorted.
    def is_redundant(k, states, net):
        if k == len(net) - 1:
            return _is_sorted(states[k])
        
        # Simulate tail
        masks = states[k][:]
        for i, j in net[k+1:]:
            mi = masks[i]
            mj = masks[j]
            masks[i] = mi & mj
            masks[j] = mi | mj
        return _is_sorted(masks)

    # 2. Greedy Deletion (Forward Scan)
    # Optimized: Pre-compute states. To check redundancy of k, we apply tail to states[k].
    # We can speed this up by caching the "tail effect" or just re-simulating the tail.
    # Given N=17 and C~100, simulation is very fast (17 ops * 100 comps = 1700 ops).
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        C = len(network)
        if C == 0: break
        
        # Compute states
        states = [initial_masks]
        current = initial_masks
        for comp in network:
            i, j = comp
            mi = current[i]
            mj = current[j]
            new_m = current[:]
            new_m[i] = mi & mj
            new_m[j] = mi | mj
            current = new_m
            states.append(current)
        
        # Scan from front to back
        for k in range(C):
            if time.time() >= deadline: break
            
            # Check redundancy of comparator k
            # If we remove k, the input to k+1 becomes states[k]
            # We need to check if applying network[k+1:] to states[k] results in sorted masks
            
            if k == C - 1:
                redundant = _is_sorted(states[k])
            else:
                # Simulate tail
                masks = states[k][:]
                for i, j in network[k+1:]:
                    mi = masks[i]
                    mj = masks[j]
                    masks[i] = mi & mj
                    masks[j] = mi | mj
                redundant = _is_sorted(masks)
            
            if redundant:
                network.pop(k)
                changed = True
                break # Restart scan from beginning as indices shifted

    best_network = network[:]
    best_len = len(network)
    
    # 3. Local Search
    # Focus on "Swap" and "Delete" moves, followed by Greedy Deletion.
    # Insertion is rarely useful for minimization unless it enables a larger deletion,
    # but Greedy Deletion handles deletions. We'll stick to Swap and Delete.
    
    for _ in range(20000): # Increase iterations, simulation is fast now
        if time.time() >= deadline - 2: break
        
        current_net = best_network[:]
        
        # Move: Swap two comparators
        if len(current_net) >= 2 and random.random() < 0.5:
            k1 = random.randint(0, len(current_net) - 1)
            k2 = random.randint(0, len(current_net) - 1)
            if k1 != k2:
                current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
        else:
            # Move: Delete a random comparator
            if len(current_net) > 0:
                k = random.randint(0, len(current_net) - 1)
                current_net.pop(k)
            else:
                continue
        
        # Check validity of the mutated network
        final_masks = simulate(current_net)
        if not _is_sorted(final_masks):
            continue
            
        # Run Greedy Deletion on the valid mutated network
        # This is the expensive part, but necessary to find the minimal form
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            C = len(current_net)
            if C == 0: break
            
            states = [initial_masks]
            current = initial_masks
            for comp in current_net:
                i, j = comp
                mi = current[i]
                mj = current[j]
                new_m = current[:]
                new_m[i] = mi & mj
                new_m[j] = mi | mj
                current = new_m
                states.append(current)
            
            for k in range(C):
                if time.time() >= deadline - 2: break
                
                if k == C - 1:
                    redundant = _is_sorted(states[k])
                else:
                    masks = states[k][:]
                    for i, j in current_net[k+1:]:
                        mi = masks[i]
                        mj = masks[j]
                        masks[i] = mi & mj
                        masks[j] = mi | mj
                    redundant = _is_sorted(masks)
                
                if redundant:
                    current_net.pop(k)
                    changed = True
                    break
        
        if len(current_net) < best_len:
            best_network = current_net
            best_len = len(current_net)
            
    # Final check
    final_masks = simulate(best_network)
    if _is_sorted(final_masks):
        return best_network
    else:
        # Fallback to seed (should not happen if seed is valid)
        return network
