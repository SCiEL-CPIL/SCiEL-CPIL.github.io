import numpy as np
import time
import random

# Precomputation
N = 512
# For each x (0-indexed), store arrays of indices for the 3 other elements
# for each of the 4 roles.
# roles:
# 0: x is 4th (a, a+d, a+2d, x) -> need a, a+d, a+2d. a = x-3d.
# 1: x is 3rd (a, a+d, x, a+3d) -> need a, a+d, a+3d. a = x-2d.
# 2: x is 2nd (a, x, a+2d, a+3d) -> need a, a+2d, a+3d. a = x-d.
# 3: x is 1st (x, a+d, a+2d, a+3d) -> need a+d, a+2d, a+3d. a = x.

# We will store lists of numpy arrays for each x and role.
# Actually, just one array per role per x containing all indices?
# No, we need to check triples.
# We can store 3 arrays per role per x: idx1, idx2, idx3.

conflict_data = []

for x in range(1, N + 1):
    idx_x = x - 1
    role_data = []
    
    # Role 0: x is 4th
    # d such that x-3d >= 1
    max_d = (x - 1) // 3
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        i1 = x - 1 - 3*ds  # a
        i2 = x - 1 - 2*ds  # a+d
        i3 = x - 1 - 1*ds  # a+2d
        role_data.append((i1, i2, i3))
    else:
        role_data.append(None)
        
    # Role 1: x is 3rd
    # d such that x+2d <= N and x-2d >= 1
    max_d1 = (N - x) // 2
    max_d2 = (x - 1) // 2
    max_d = min(max_d1, max_d2)
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        i1 = x - 1 - 2*ds  # a
        i2 = x - 1 - 1*ds  # a+d
        i3 = x - 1 + 1*ds  # a+3d
        role_data.append((i1, i2, i3))
    else:
        role_data.append(None)
        
    # Role 2: x is 2nd
    # d such that x+3d <= N and x-3d >= 1
    max_d1 = (N - x) // 3
    max_d2 = (x - 1) // 3
    max_d = min(max_d1, max_d2)
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        i1 = x - 1 - 3*ds  # a
        i2 = x - 1 - 2*ds  # a+2d
        i3 = x - 1 + 1*ds  # a+3d
        role_data.append((i1, i2, i3))
    else:
        role_data.append(None)
        
    # Role 3: x is 1st
    # d such that x+3d <= N
    max_d = (N - x) // 3
    if max_d > 0:
        ds = np.arange(1, max_d + 1)
        i1 = x - 1 + 1*ds  # a+d
        i2 = x - 1 + 2*ds  # a+2d
        i3 = x - 1 + 3*ds  # a+3d
        role_data.append((i1, i2, i3))
    else:
        role_data.append(None)
        
    conflict_data.append(role_data)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_size = 0
    best_set = set()
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        order = list(range(N))
        random.shuffle(order)
        
        selected_bits = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            # Check conflicts
            can_add = True
            roles = conflict_data[idx]
            
            for r in range(4):
                data = roles[r]
                if data is not None:
                    i1, i2, i3 = data
                    # Check if any triple is fully selected
                    # selected_bits[i1] & selected_bits[i2] & selected_bits[i3]
                    if np.any(selected_bits[i1] & selected_bits[i2] & selected_bits[i3]):
                        can_add = False
                        break
                        
            if can_add:
                selected_bits[idx] = True
                count += 1
                
        if count > best_size:
            best_size = count
            best_indices = np.where(selected_bits)[0] + 1
            best_set = set(best_indices)
            
    if best_size == 0:
        # Fallback
        return [1]
        
    return sorted(best_set)
