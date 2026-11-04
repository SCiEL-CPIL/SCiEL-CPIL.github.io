import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Precomputation
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Conflict matrix
    # conflict[i, j] = True if intersection >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    rng = np.random.RandomState(42)
    
    def random_init():
        x = np.zeros(N_S, dtype=bool)
        order = rng.permutation(N_S)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    def local_search(x, max_iter=2000):
        current_x = x.copy()
        current_score = np.sum(current_x)
        best_x = current_x.copy()
        best_score = current_score
        
        # Precompute neighbors? Not strictly necessary with numpy matrix ops.
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Calculate blocked set
            in_set = np.where(current_x)[0]
            if len(in_set) > 0:
                blocked = np.any(conflict[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Move selection
            r = rng.random()
            
            if r < 0.4:
                # Add
                candidates = np.where(~(current_x | blocked))[0]
                if len(candidates) > 0:
                    v = int(rng.choice(candidates))
                    new_x = current_x.copy()
                    new_x[v] = True
                    new_score = current_score + 1
                    current_x = new_x
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_x = current_x.copy()
                        
            elif r < 0.7:
                # Swap
                if len(in_set) == 0:
                    continue
                u = int(rng.choice(in_set))
                
                # Candidates for v: not in x, not blocked by x \ {u}
                temp_x = current_x.copy()
                temp_x[u] = False
                in_set_no_u = np.where(temp_x)[0]
                
                if len(in_set_no_u) > 0:
                    blocked_no_u = np.any(conflict[in_set_no_u], axis=0)
                else:
                    blocked_no_u = np.zeros(N_S, dtype=bool)
                
                candidates = np.where(~(current_x | blocked_no_u))[0]
                if len(candidates) > 0:
                    v = int(rng.choice(candidates))
                    new_x = current_x.copy()
                    new_x[u] = False
                    new_x[v] = True
                    new_score = current_score
                    # Accept neutral or improving?
                    # For MIS, neutral swaps allow exploration.
                    # Use a simple acceptance: always accept if score >= current?
                    # Or use a temperature?
                    # Let's just accept to allow random walk.
                    current_x = new_x
                    # current_score unchanged
                    if new_score > best_score:
                        best_score = new_score
                        best_x = current_x.copy()
                        
            else:
                # Remove (to escape)
                if len(in_set) > 0:
                    v = int(rng.choice(in_set))
                    new_x = current_x.copy()
                    new_x[v] = False
                    new_score = current_score - 1
                    # Accept with probability based on temperature?
                    # Simple: accept if random < 0.5
                    if rng.random() < 0.5:
                        current_x = new_x
                        current_score = new_score

        # Final greedy polish
        in_set = np.where(current_x)[0]
        if len(in_set) > 0:
            blocked = np.any(conflict[in_set], axis=0)
        else:
            blocked = np.zeros(N_S, dtype=bool)
            
        changed = True
        while changed:
            changed = False
            candidates = np.where(~(current_x | blocked))[0]
            if len(candidates) > 0:
                v = int(rng.choice(candidates))
                current_x[v] = True
                current_score += 1
                blocked |= conflict[v]
                changed = True
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

        return best_x

    best_support_set = set()
    best_size = 0
    
    # Run multiple restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        x0 = random_init()
        x_best = local_search(x0, max_iter=1000)
        size = np.sum(x_best)
        
        if size > best_size:
            best_size = int(size)
            best_support_set = set(np.where(x_best)[0].tolist())
            
    # Construct vectors
    vectors = []
    
    # Type A
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
