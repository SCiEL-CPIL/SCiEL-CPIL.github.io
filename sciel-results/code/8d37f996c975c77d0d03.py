import numpy as np
import itertools
import time
import sys

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all vectors of weight 1 with entry +/- 2"""
    candidates = []
    for i in range(D):
        for s in [2, -2]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def local_improve(conflicts, initial_set, deadline, rng):
    """
    Local search to improve the independent set.
    Tries to swap vertices to increase set size.
    """
    N = len(conflicts)
    current_set = set(initial_set)
    in_set = np.zeros(N, dtype=bool)
    for i in current_set:
        in_set[i] = True
    
    best_size = len(current_set)
    best_set = list(current_set)
    
    # Precompute neighbor lists for faster access
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any vertex that has no neighbors in set
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) == 0:
            break
            
        rng.shuffle(not_in_set)
        
        for u in not_in_set:
            if time.time() > deadline:
                break
                
            if not np.any(in_set[neighbors[u]]):
                in_set[u] = True
                current_set.add(u)
                best_size += 1
                best_set = list(current_set)
                improved = True
        
        if not improved:
            # Try 1-swap
            u = rng.integers(N)
            if in_set[u]:
                continue
                
            neigh_in_set = neighbors[u][in_set[neighbors[u]]]
            
            if len(neigh_in_set) == 1:
                v = neigh_in_set[0]
                
                in_set[v] = False
                current_set.remove(v)
                in_set[u] = True
                current_set.add(u)
                
                added_any = False
                not_in_set_now = np.where(~in_set)[0]
                rng.shuffle(not_in_set_now)
                
                for w in not_in_set_now:
                    if not np.any(in_set[neighbors[w]]):
                        in_set[w] = True
                        current_set.add(w)
                        added_any = True
                        
                if not added_any:
                    in_set[u] = False
                    current_set.remove(u)
                    in_set[v] = True
                    current_set.add(v)
                else:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    # 1. Generate Weight-4 vectors
    vectors_w4 = generate_weight_4_vectors()
    N = len(vectors_w4)
    
    if N == 0:
        # Fallback
        w1 = generate_weight_1_vectors()
        return w1.astype(np.int64)
        
    # 2. Build Conflict Graph for W4
    dots = vectors_w4 @ vectors_w4.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # 3. Run Greedy MIS multiple times with different seeds to find a good set
    # We have ~115s. Greedy is fast.
    # Let's run greedy for the first 60s, keeping the best result.
    greedy_deadline = t0 + 60
    
    while time.time() < greedy_deadline:
        current_set = greedy_mis(conflicts, greedy_deadline, rng)
        # Apply local improvement to the greedy result
        improved_set = local_improve(conflicts, current_set, greedy_deadline, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    # 4. Generate Weight-1 vectors
    vectors_w1 = generate_weight_1_vectors()
    
    # 5. Combine
    # The W1 vectors are compatible with all W4 vectors.
    # They are compatible with each other.
    # So we just take the union.
    
    if not best_set:
        # If MIS failed, just return W1
        return vectors_w1.astype(np.int64)
        
    result_vectors = np.vstack([vectors_w4[best_set], vectors_w1])
    
    # 6. Verification (Optional, but good for safety)
    # Check if min pairwise distance >= max norm
    # Max norm is 2 (squared 4).
    # We need d(u,v)^2 >= 4 for all u != v.
    # This is equivalent to u.v <= 2 for all u,v with norm 2.
    # For W1 and W4, we proved this.
    # For W1 and W1, we proved this.
    # For W4 and W4, MIS ensures this.
    
    return result_vectors.astype(np.int64)
