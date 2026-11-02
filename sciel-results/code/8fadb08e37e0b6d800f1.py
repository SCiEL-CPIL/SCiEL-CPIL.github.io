import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate Candidates
    # Weight 2, 3, 4
    candidates = []
    for w in [2, 3, 4]:
        for positions in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # 2. Compute Conflicts
    # R^2 is max norm sq = 4
    # Conflict if dist_sq < 4
    # dist_sq = n_i + n_j - 2 dot_ij
    norms_sq = np.sum(candidates**2, axis=1)
    dots = np.dot(candidates, candidates.T)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency list (lists for speed)
    neighbors = [np.where(conflicts[i])[0].tolist() for i in range(n_candidates)]
    
    # 3. Greedy Initialization
    def greedy_init():
        available = set(range(n_candidates))
        selected = set()
        # Randomize order for diversity
        indices = list(range(n_candidates))
        random.shuffle(indices)
        for v in indices:
            if v in available:
                # Check if v conflicts with any in selected
                # Optimized check:
                conflict_found = False
                for u in neighbors[v]:
                    if u in selected:
                        conflict_found = True
                        break
                if not conflict_found:
                    selected.add(v)
                    # Remove neighbors from available
                    for u in neighbors[v]:
                        if u in available:
                            available.remove(u)
                    available.remove(v)
        return selected

    # 4. Local Search / ILS
    def local_search(initial_set, max_iter=1000):
        S = set(initial_set)
        best_size = len(S)
        best_S = S.copy()
        
        # Track "cost" of removing? Just simple swap.
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to add a random vertex from a pool of "potential" vertices
            # Potentials are those not in S. 
            # To be efficient, we can maintain a list of candidates not in S.
            # But S changes.
            
            # Simple move: Pick a random v not in S.
            # If it fits, add it.
            # If it doesn't fit, try to swap with a neighbor u in S.
            
            # Optimization: Maintain list of non-selected? Too heavy.
            # Just sample random indices.
            
            v = random.randint(0, n_candidates - 1)
            if v in S:
                continue
            
            # Check if v can be added
            conflict_with_S = [u for u in neighbors[v] if u in S]
            
            if not conflict_with_S:
                S.add(v)
                improved = True
            else:
                # Try to remove one or more neighbors to add v
                # Heuristic: Remove the neighbor with the smallest "degree in S"
                # Or just try random swap.
                # Let's try removing 1 neighbor.
                u_to_remove = random.choice(conflict_with_S)
                
                # If we remove u_to_remove, does v fit?
                # v conflicts with other members of conflict_with_S?
                # Yes, by definition conflict_with_S is the set of ALL neighbors of v in S.
                # So removing one is not enough if |conflict_with_S| > 1.
                
                if len(conflict_with_S) == 1:
                    S.discard(u_to_remove)
                    S.add(v)
                    improved = True
                elif len(conflict_with_S) == 2:
                    # Try removing both?
                    u1, u2 = conflict_with_S
                    # Check if u1 and u2 are adjacent? Doesn't matter for v.
                    # But we lose 2, gain 1. Net -1. Bad unless it opens up more.
                    # Skip for now.
                    pass
            
            if improved:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                # Reset iteration count? No.
                # If we hit deadline, return best.
            else:
                # No improvement. Try a perturbation?
                # Remove a random small set of vertices to break local optimum.
                # This is the "ILS" part.
                if random.random() < 0.1:
                    # Perturb: remove 5% of S
                    num_remove = max(1, len(S) // 20)
                    to_remove = random.sample(list(S), num_remove)
                    for u in to_remove:
                        S.discard(u)
        
        return best_S, best_size

    # Run multiple restarts
    best_global_size = 0
    best_global_set = set()
    
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        S_init = greedy_init()
        
        # Local Search
        S_local, size_local = local_search(S_init)
        
        if size_local > best_global_size:
            best_global_size = size_local
            best_global_set = S_local
            # print(f"Restart {i}: Found {best_global_size}")
            
    if not best_global_set:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = candidates[list(best_global_set)]
    return result

# EVOLVE-BLOCK-END
