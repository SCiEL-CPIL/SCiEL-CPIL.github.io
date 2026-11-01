"""Combined program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115

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
    
    # Convert to neighbor lists
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

    def local_search_v2(in_set):
        in_set = in_set.copy()
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        avail_set = set(np.where((counts == 0) & (~in_set))[0])
        
        def refill():
            nonlocal avail_set, counts, in_set
            added = []
            while avail_set:
                u = next(iter(avail_set))
                avail_set.discard(u)
                if counts[u] == 0 and not in_set[u]:
                    in_set[u] = True
                    added.append(u)
                    for w in neighbors[u]:
                        if not in_set[w]:
                            counts[w] += 1
                            if counts[w] == 1:
                                avail_set.discard(w)
            return added

        def remove_node(v):
            nonlocal avail_set, counts, in_set
            in_set[v] = False
            for w in neighbors[v]:
                counts[w] -= 1
                if counts[w] == 0 and not in_set[w]:
                    avail_set.add(w)

        def revert(added, v):
            nonlocal avail_set, counts, in_set
            for u in added:
                remove_node(u)
            # Re-add v
            in_set[v] = True
            for w in neighbors[v]:
                if not in_set[w]:
                    counts[w] += 1
                    if counts[w] == 1:
                        avail_set.discard(w)

        # Initial Fill
        refill()
        current_size = in_set.sum()
        
        while time.time() < deadline - 10:
            improved = False
            in_set_indices = np.where(in_set)[0]
            rng.shuffle(in_set_indices)
            
            for v in in_set_indices:
                if not in_set[v]:
                    continue
                    
                remove_node(v)
                added = refill()
                
                new_size = in_set.sum()
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    revert(added, v)
                    
            if not improved:
                # Perturbation
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 3:
                    to_remove = rng.choice(in_set_indices, size=2, replace=False)
                    for v in to_remove:
                        remove_node(v)
                    refill()
                    new_size = in_set.sum()
                    if new_size > current_size:
                        current_size = new_size
                        improved = True
                    else:
                        # Accept perturbation to escape local optimum
                        pass
                        
                if not improved:
                    break
                    
        return in_set

    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_min_degree()
        in_set = local_search_v2(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # Perturbation of best
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search_v2(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
