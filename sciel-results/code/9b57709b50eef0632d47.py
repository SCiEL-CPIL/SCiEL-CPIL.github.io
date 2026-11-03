import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all weight 3 vectors
    # Support: choose 3 indices from 11
    # Signs: 2^3 = 8 patterns
    candidates = []
    indices = list(range(D))
    
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Precompute incompatibility matrix: two vectors are incompatible if their dot product > 1
    # For weight-3 vectors: ||u-v||^2 = 6 - 2*u.v >= 3 iff u.v <= 1
    # So incompatible if u.v >= 2
    dots = candidates @ candidates.T  # N x N matrix of dot products
    incompat = (dots > 1).astype(np.int8)
    np.fill_diagonal(incompat, 0)
    
    # Simulated annealing to find maximum independent set
    # State: binary vector x of length N, x[i]=1 means vector i is selected
    # Constraint: no two selected vectors are incompatible
    # Objective: maximize sum(x)
    
    best_x = None
    best_size = 0
    
    # Multiple restarts with different initial solutions
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with greedy solution
        x = np.zeros(N, dtype=np.int8)
        remaining = np.ones(N, dtype=bool)
        
        while remaining.any():
            # Pick a random remaining vector
            idx = rng.choice(np.where(remaining)[0])
            x[idx] = 1
            # Remove incompatible vectors from remaining
            remaining[incompat[idx] == 1] = False
            remaining[idx] = False
        
        current_size = x.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_x = x.copy()
        
        # Simulated annealing
        T_start = 1.0
        T_end = 0.01
        n_iterations = 5000
        T = T_start
        
        for it in range(n_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Cool down
            T = T_start * (T_end / T_start) ** (it / n_iterations)
            
            # Pick a random move: add or remove a vector
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add a vector
                # Find vectors not currently selected
                unselected = np.where(x == 0)[0]
                if len(unselected) == 0:
                    continue
                idx = rng.choice(unselected)
                
                # Check if adding idx would violate constraints
                # idx is compatible with all currently selected vectors if incompat[idx, x==1].sum() == 0
                if incompat[idx] @ x == 0:
                    x[idx] = 1
                    delta = 1
                else:
                    # Can't add without violating, skip
                    continue
            else:
                # Try to remove a vector
                selected_idx = np.where(x == 1)[0]
                if len(selected_idx) == 0:
                    continue
                idx = rng.choice(selected_idx)
                x[idx] = 0
                delta = -1
            
            # Accept or reject based on energy change
            # Energy = -size (we want to maximize size, so minimize -size)
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_size = x.sum()
                if current_size > best_size:
                    best_size = current_size
                    best_x = x.copy()
            else:
                # Reject move
                if move_type < 0.5:
                    x[idx] = 0  # undo add
                else:
                    x[idx] = 1  # undo remove
    
    if best_x is not None and best_size > 0:
        return candidates[best_x == 1]
    
    # Fallback: greedy
    selected = []
    for i in range(N):
        if time.time() > deadline - 5.0:
            break
        v = candidates[i]
        valid = True
        for u in selected:
            dot = np.dot(u, v)
            if dot > 1:
                valid = False
                break
        if valid:
            selected.append(v)
    
    if selected:
        return np.array(selected, dtype=np.int8)
    
    # Last resort: weight-2 vectors
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
    return np.array(vectors, dtype=np.int8)
