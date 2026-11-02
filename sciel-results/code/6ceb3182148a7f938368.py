import time
import numpy as np
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate Vectors
    vectors_A = []
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors_A.append(v)
            
    vectors_B = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vectors_B.append(v)
            
    n_A = len(vectors_A)
    n_B = len(vectors_B)
    
    final_vectors_A = np.array(vectors_A, dtype=np.int64)
    V_B = np.array(vectors_B, dtype=np.int64)
    
    # 2. Build Conflict Graph for Type B
    dot_B = V_B @ V_B.T
    conflict_B = (dot_B > 2).astype(np.int8)
    np.fill_diagonal(conflict_B, 0)
    
    # Precompute neighbor lists for speed
    # Using lists of lists
    neighbors = []
    for i in range(n_B):
        neighbors.append(np.nonzero(conflict_B[i])[0])
        
    # 3. Local Search
    selected = np.zeros(n_B, dtype=bool)
    conflict_count = np.zeros(n_B, dtype=np.int32)
    available = set()
    
    # Greedy Init
    # Add all valid vectors
    for i in range(n_B):
        available.add(i)
        
    def polish():
        nonlocal available, selected, conflict_count
        while available:
            v = available.pop()
            if selected[v] or conflict_count[v] > 0:
                continue
            # Add v
            selected[v] = True
            for u in neighbors[v]:
                conflict_count[u] += 1
                if conflict_count[u] == 1 and not selected[u]:
                    available.discard(u)
                    
    polish()
    best_size = np.sum(selected)
    best_selected = selected.copy()
    
    # SA Loop
    while time.time() < deadline - 5:
        if not available:
            # Remove a random selected vector
            sel_indices = np.nonzero(selected)[0]
            if len(sel_indices) == 0:
                break
            v_remove = sel_indices[np.random.randint(len(sel_indices))]
            
            # Remove v_remove
            selected[v_remove] = False
            for u in neighbors[v_remove]:
                conflict_count[u] -= 1
                if conflict_count[u] == 0 and not selected[u]:
                    available.add(u)
            
            # Polish
            polish()
            
            current_size = np.sum(selected)
            if current_size > best_size:
                best_size = current_size
                best_selected = selected.copy()
        else:
            # If available is not empty, just polish (should be empty after previous polish)
            # This case shouldn't happen if polish is exhaustive, but just in case
            polish()
            
    # Final Polish
    selected = best_selected.copy()
    conflict_count = np.zeros(n_B, dtype=np.int32)
    available = set()
    for i in range(n_B):
        if not selected[i]:
            # Calculate initial conflict count
            # This is O(N*Degree) which is fine for one time
            pass
            
    # Recalculate conflict counts for best_selected
    for v in np.nonzero(selected)[0]:
        for u in neighbors[v]:
            conflict_count[u] += 1
            
    for i in range(n_B):
        if not selected[i] and conflict_count[i] == 0:
            available.add(i)
            
    polish()
    
    final_B = V_B[selected]
    
    if len(final_B) > 0:
        result = np.vstack((final_vectors_A, final_B))
    else:
        result = final_vectors_A
        
    return result

# EVOLVE-BLOCK-END
