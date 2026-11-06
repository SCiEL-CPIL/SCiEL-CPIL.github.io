import time
import random
from math import gcd
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # 5s safety margin

    n = limit
    # Nodes are 1..512.
    # We want to find a subset S of [1..512] such that lcm(a,b) <= 512 for all a,b in S.
    # This is a Maximum Clique problem on the graph G where (u,v) in E if lcm(u,v) <= 512.
    
    # Build adjacency matrix using NumPy for fast bitset operations
    # adj[i, j] = 1 if lcm(i+1, j+1) <= 512
    # We use 0-indexed for numpy, so node i corresponds to number i+1.
    
    # Precompute LCM matrix
    # This is 512x512, which is small.
    lcm_matrix = np.zeros((n, n), dtype=np.uint8)
    for u in range(n):
        num_u = u + 1
        for v in range(u + 1, n):
            num_v = v + 1
            g = gcd(num_u, num_v)
            lcm_val = (num_u * num_v) // g
            if lcm_val <= limit:
                lcm_matrix[u, v] = 1
                lcm_matrix[v, u] = 1
    # Diagonal is 1 (self-compatible)
    np.fill_diagonal(lcm_matrix, 1)
    
    # Convert to bitsets for fast intersection
    # bitsets[i] is a numpy array of uint8 representing the neighbors of i (including i)
    # Actually, let's use Python integers as bitsets if n <= 512? 
    # Python ints are arbitrary precision, so this is very fast for bitwise ops.
    
    bitsets = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if lcm_matrix[i, j] == 1:
                mask |= (1 << j)
        bitsets[i] = mask

    # Simulated Annealing
    # State: a set of nodes (represented as a boolean numpy array or a set of indices)
    # We represent the current solution as a set of indices.
    
    # Initial solution: Greedy
    # Sort by degree (number of neighbors)
    degrees = [bin(bitsets[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    current_set = set()
    for v in order:
        # Check if v is compatible with all in current_set
        # Using bitsets: 
        # intersection_mask = bitsets[v]
        # for u in current_set:
        #    if not (intersection_mask & (1 << u)): break
        # Faster: 
        # curr_mask = 0
        # for u in current_set: curr_mask |= (1 << u)
        # if (bitsets[v] & curr_mask) == curr_mask: add v
        
        curr_mask = 0
        for u in current_set:
            curr_mask |= (1 << u)
        
        if (bitsets[v] & curr_mask) == curr_mask:
            current_set.add(v)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Simulated Annealing Parameters
    # Temperature decay
    T_start = 10.0
    T_end = 0.1
    alpha = 0.99
    max_iters = 50000
    
    # Precompute neighbor lists for faster candidate generation?
    # neighbor_list[i] = list of neighbors of i
    neighbor_list = []
    for i in range(n):
        neighbors = []
        mask = bitsets[i]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neighbors.append(idx)
            mask ^= lsb
        neighbor_list.append(neighbors)

    current_size = len(current_set)
    in_set = [False] * n
    for v in current_set:
        in_set[v] = True
        
    # Mask of current set for fast check
    curr_mask = 0
    for v in current_set:
        curr_mask |= (1 << v)
        
    T = T_start
    iter_count = 0
    
    while time.time() < deadline and T > T_end:
        # Generate a random move:
        # 1. Remove a random element from current_set
        # 2. Add a random element not in current_set that is compatible with all remaining
        # 3. Swap: Remove u, Add v
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove
            if not current_set:
                T *= alpha
                continue
            u = random.choice(list(current_set))
            current_set.remove(u)
            in_set[u] = False
            curr_mask &= ~(1 << u)
            delta_size = -1
            # Energy change is 0 (we don't have a cost function other than size, but SA works on validity)
            # We just want to maximize size.
            # Acceptance: Always accept removal? Or accept with prob exp(-delta_E/T)?
            # Here "Energy" is -size. So removing increases energy (bad).
            # But we need to escape local optima.
            # Let's use a different approach: 
            # Proposal: Pick a random valid set? No.
            
            # Let's stick to standard SA on the objective function: Size.
            # But we must maintain validity.
            # If we remove, validity is maintained. Size decreases.
            # If we add, validity might be broken.
            
            # Better Move:
            # 1. Pick a random node v NOT in set.
            # 2. Check if v can be added. If yes, add it. (Improvement, always accept)
            # 3. If not, pick a random node u IN set.
            # 4. Check if (set - {u}) U {v} is valid. If yes, swap. (Neutral size, accept with prob exp(-0/T)=1? No, we want to explore)
            # 5. If not, remove u. (Worse, accept with prob exp(-1/T))
            
            # Let's re-evaluate the move generation.
            # Reset state for this iteration if we did a partial move
            # Actually, let's just do the move properly.
            
            # Undo removal
            current_set.add(u)
            in_set[u] = True
            curr_mask |= (1 << u)
            
        # Proper Move Generation
        # 1. Try Add
        # Pick random v not in set
        not_in_set = [i for i in range(n) if not in_set[i]]
        if not_in_set:
            v = random.choice(not_in_set)
            # Check if v compatible with all in set
            # (bitsets[v] & curr_mask) == curr_mask
            if (bitsets[v] & curr_mask) == curr_mask:
                # Accept
                current_set.add(v)
                in_set[v] = True
                curr_mask |= (1 << v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
            else:
                # 2. Try Swap
                if current_set:
                    u = random.choice(list(current_set))
                    # Check if (set - {u}) U {v} is valid
                    # New mask = curr_mask ^ (1 << u)
                    new_mask = curr_mask ^ (1 << u)
                    if (bitsets[v] & new_mask) == new_mask:
                        # Accept swap (size same)
                        # Probability 1? Or exp(-0/T)?
                        # To explore, we accept neutral moves with high prob
                        if random.random() < 0.5:
                            current_set.remove(u)
                            in_set[u] = False
                            current_set.add(v)
                            in_set[v] = True
                            curr_mask = new_mask | (1 << v)
                    else:
                        # 3. Try Remove (if swap failed or no v)
                        # Actually, if add failed and swap failed, we can try removing u to escape
                        # But removing reduces size.
                        # Accept with prob exp(-1/T)
                        if random.random() < np.exp(-1.0 / T):
                            current_set.remove(u)
                            in_set[u] = False
                            curr_mask ^= (1 << u)
                            current_size -= 1
                            
        T *= alpha
        iter_count += 1
        
    # Final Verification
    final_set = set(best_set)
    final_list = list(final_set)
    is_valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            a = final_list[i] + 1
            b = final_list[j] + 1
            g = gcd(a, b)
            if (a * b) // g > limit:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        return [1]
        
    return sorted([x + 1 for x in final_set])

construct()
