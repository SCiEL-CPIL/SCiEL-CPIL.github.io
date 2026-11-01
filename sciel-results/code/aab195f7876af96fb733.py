"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set, updating counts.
        Returns number of nodes added.
        """
        added = 0
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def rebuild_counts(in_set):
        """Rebuild conflict counts from scratch using matrix multiply."""
        return conflicts[in_set].sum(axis=0).astype(np.int32)

    def local_search(in_set):
        """
        Iterated local search: Try removing nodes (1 or 2) and greedily filling.
        Uses vectorized count updates for speed.
        """
        in_set = in_set.copy()
        counts = rebuild_counts(in_set)
            
        current_size = in_set.sum()
        
        # First, greedy fill to maximize current set
        added = greedy_fill(in_set, counts)
        current_size += added
        
        # Local search loop
        while time.time() < deadline - 5:
            improved = False
            in_set_indices = np.where(in_set)[0]
            rng.shuffle(in_set_indices)
            
            # Pass 1: single removals
            for v in in_set_indices:
                if not in_set[v]:
                    continue
                    
                # Try removing v
                in_set[v] = False
                for u in neighbors[v]:
                    counts[u] -= 1
                    
                # Greedy fill
                candidates = np.where((counts == 0) & (~in_set))[0]
                added = 0
                for u in candidates:
                    if counts[u] == 0 and not in_set[u]:
                        in_set[u] = True
                        added += 1
                        for w in neighbors[u]:
                            counts[w] += 1
                            
                new_size = in_set.sum()
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert
                    in_set[v] = True
                    counts = rebuild_counts(in_set)
            
            # Pass 2: 2-for-1 swaps (remove 2, fill)
            if time.time() < deadline - 5:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) >= 4:
                    n_pairs = min(200, len(in_set_indices) // 2)
                    for _ in range(n_pairs):
                        pair = rng.choice(in_set_indices, size=2, replace=False)
                        v1, v2 = int(pair[0]), int(pair[1])
                        if not in_set[v1] or not in_set[v2]:
                            continue
                        
                        # Remove both
                        in_set[v1] = False
                        in_set[v2] = False
                        for u in neighbors[v1]:
                            counts[u] -= 1
                        for u in neighbors[v2]:
                            counts[u] -= 1
                        # Fix double-counted neighbors
                        common = set(neighbors[v1]) & set(neighbors[v2])
                        for u in common:
                            counts[u] += 1
                        
                        # Greedy fill
                        candidates = np.where((counts == 0) & (~in_set))[0]
                        added = 0
                        for u in candidates:
                            if counts[u] == 0 and not in_set[u]:
                                in_set[u] = True
                                added += 1
                                for w in neighbors[u]:
                                    counts[w] += 1
                            
                        new_size = in_set.sum()
                        if new_size > current_size:
                            current_size = new_size
                            improved = True
                        else:
                            # Revert
                            in_set[v1] = True
                            in_set[v2] = True
                            counts = rebuild_counts(in_set)
            
            if not improved:
                break
                
        return in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
