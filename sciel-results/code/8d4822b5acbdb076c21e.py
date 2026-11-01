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
    
    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N1 = len(vectors_v1)
    
    # Combine all vectors
    all_vectors = np.vstack([vectors_v4, vectors_v1])
    N = len(all_vectors)
    
    # Compute conflict matrix: conflict if dot > 2
    # Note: We only care about pairs (i, j) where i != j.
    dots = all_vectors @ all_vectors.T
    # Conflict if dot > 2. 
    # Also, we must ensure distinct vectors. The generation ensures distinctness.
    # However, if two vectors are identical, they are not distinct. 
    # Our generation creates distinct vectors.
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
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
                # Update degrees for remaining alive nodes
                # A node u loses a neighbor if v is removed or v is a neighbor of u
                # Here we remove chosen and its neighbors.
                # For u in rem, its degree decreases by the number of neighbors of u that are in to_remove
                # to_remove contains chosen and neighbors of chosen.
                # So for u, the removed neighbors are exactly the intersection of neighbors[u] and to_remove.
                # This is equivalent to conflicts[u, to_remove].sum()
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

    def calculate_conflicts(in_set):
        """Calculate total number of conflicting pairs in in_set."""
        # This is O(N^2) if done naively, but we can use the neighbor list.
        # Sum of counts for each node in set, divided by 2?
        # No, counts[u] is the number of neighbors of u in the set.
        # Sum of counts[u] for u in set is 2 * (number of edges in the subgraph induced by set).
        counts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
        # The number of conflicts is the sum of counts for nodes in the set, divided by 2?
        # Let S be the set. Edges are pairs (u,v) in S such that u~v.
        # Sum_{u in S} count(u) = Sum_{u in S} |N(u) \cap S| = 2 * |E(S)|.
        # So conflicts = sum(counts[in_set]) // 2
        return np.sum(counts[in_set]) // 2

    def local_search_sa(in_set):
        """
        Simulated Annealing based local search.
        Objective: Maximize Size - Lambda * Conflicts
        """
        in_set = in_set.copy()
        size = in_set.sum()
        conflicts = calculate_conflicts(in_set)
        
        best_in_set = in_set.copy()
        best_score = size - 10.0 * conflicts
        
        T = 5.0
        cooling = 0.999
        lambda_pen = 10.0
        
        # Pre-compute some stats for speed?
        # We need to update size and conflicts efficiently.
        
        while time.time() < deadline - 10 and T > 0.01:
            # Random move
            move_type = rng.integers(3) # 0: Add, 1: Remove, 2: Swap
            
            if move_type == 0:
                # Add a random node
                v = rng.integers(N)
                if in_set[v]:
                    continue
                # Check conflicts of v with current set
                nbrs = neighbors[v]
                conflict_count = np.count_nonzero(in_set[nbrs])
                
                delta_size = 1
                delta_conflicts = conflict_count
                delta_E = -(delta_size - lambda_pen * delta_conflicts)
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = True
                    size += 1
                    conflicts += conflict_count
                    
            elif move_type == 1:
                # Remove a random node
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                
                nbrs = neighbors[v]
                conflict_count = np.count_nonzero(in_set[nbrs])
                
                delta_size = -1
                delta_conflicts = -conflict_count
                delta_E = -(delta_size - lambda_pen * delta_conflicts)
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = False
                    size -= 1
                    conflicts -= conflict_count
                    
            else:
                # Swap: Remove v, add w
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                w = rng.integers(N)
                if in_set[w]:
                    continue
                
                # Calculate change in conflicts
                # Remove v: loses conflict_count_v
                # Add w: gains conflict_count_w
                # But if v and w are neighbors, we lose 1 conflict from v-w pair?
                # No, v is removed, w is added.
                # Old conflicts involving v: c_v
                # New conflicts involving w: c_w
                # If v and w are neighbors, the pair (v,w) was a conflict (if w was in set, but w is not).
                # So we just replace v's contribution with w's.
                
                c_v = np.count_nonzero(in_set[neighbors[v]])
                c_w = np.count_nonzero(in_set[neighbors[w]])
                
                # Check if v and w are neighbors
                # If they are, and w is added, does it conflict with v? No, v is removed.
                # Does w conflict with other nodes? Yes, c_w.
                # Does removing v remove conflicts? Yes, c_v.
                
                delta_size = 0
                delta_conflicts = c_w - c_v
                
                delta_E = -(delta_size - lambda_pen * delta_conflicts)
                
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = False
                    in_set[w] = True
                    conflicts += delta_conflicts
                    # Size remains same
            
            if conflicts == 0:
                current_score = size
            else:
                current_score = size - lambda_pen * conflicts
            
            if current_score > best_score:
                best_score = current_score
                best_in_set = in_set.copy()
                
            T *= cooling
            
        # Final Polish: Greedy fill
        counts = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            counts[neighbors[i]] += 1
            
        # If there are conflicts, we must resolve them to get a valid set.
        # The SA might end with conflicts. We need a valid independent set.
        # If conflicts > 0, we remove nodes until valid.
        # This is a "repair" step.
        while calculate_conflicts(best_in_set) > 0 and time.time() < deadline - 2:
             # Find a node with max degree in the conflict subgraph
             counts = np.zeros(N, dtype=np.int32)
             for i in np.where(best_in_set)[0]:
                 counts[neighbors[i]] += 1
             # Only consider nodes in best_in_set
             in_set_indices = np.where(best_in_set)[0]
             if len(in_set_indices) == 0:
                 break
             max_deg = np.max(counts[in_set_indices])
             cands = in_set_indices[counts[in_set_indices] == max_deg]
             v = cands[rng.integers(len(cands))]
             best_in_set[v] = False
             
        # Now best_in_set is valid (independent).
        # Greedy fill to maximize size
        counts = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            counts[neighbors[i]] += 1
            
        greedy_fill(best_in_set, counts)
        
        return best_in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search_sa(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Perturbation
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search_sa(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    result_v4 = all_vectors[best_in_set]
    
    return result_v4

# EVOLVE-BLOCK-END
