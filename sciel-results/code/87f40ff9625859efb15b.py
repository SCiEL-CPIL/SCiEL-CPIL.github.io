import numpy as np
import itertools
import time

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance is at least
    the maximum vector norm.
    
    Strategy: Use weight-4 vectors in {-1, 0, 1}^11.
    Norm squared = 4. Condition: u . v <= 2.
    Conflict if u . v >= 3.
    We solve for the Maximum Independent Set (MIS) of the conflict graph using
    a Greedy Min-Degree initialization followed by Local Search (Add and 2-for-1 Swaps).
    """
    
    # Generate all weight-4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    vectors = np.array(v4_candidates, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products to find conflicts (dot >= 3)
    dots = vectors @ vectors.T
    conflicts = dots >= 3
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency list
    adj = [np.where(conflicts[i])[0] for i in range(N)]
    
    # Greedy Min-Degree Initialization
    in_set = np.zeros(N, dtype=bool)
    remaining = set(range(N))
    # Calculate initial degrees
    deg = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
    
    # We will simulate the greedy process
    # To speed up, we can use a heap, but for N=5280, a simple loop with sorting or min-finding is okay.
    # However, dynamic degree updates are needed.
    
    # Simpler Greedy:
    # 1. Start with all nodes.
    # 2. Pick node with min degree.
    # 3. Add to set, remove it and its neighbors from remaining.
    # 4. Update degrees of remaining nodes.
    
    remaining_set = set(range(N))
    current_deg = deg.copy()
    
    while remaining_set:
        # Find node with min degree in remaining_set
        min_node = min(remaining_set, key=lambda x: current_deg[x])
        in_set[min_node] = True
        # Remove min_node and its neighbors
        neighbors = adj[min_node]
        to_remove = set(neighbors)
        to_remove.add(min_node)
        remaining_set -= to_remove
        # Update degrees
        for n in to_remove:
            if n in remaining_set: # Should be false for n in to_remove, but neighbors of neighbors?
                pass
        # Actually, we need to update degrees of nodes in remaining_set that are neighbors of removed nodes.
        # This is O(N^2) in worst case if done naively.
        # For N=5280, N^2 = 27M. This might be slow in Python.
        
    # The above greedy is too slow in Python if implemented with sets and loops.
    # Let's use a vectorized approach or a simpler heuristic.
    
    # Heuristic: Randomized Greedy with Multiple Restarts
    # This is often faster and finds good solutions.
    
    best_size = 0
    best_mask = np.zeros(N, dtype=bool)
    
    import random
    start_time = time.time()
    deadline = start_time + 110
    
    def run_greedy(order):
        selected = np.zeros(N, dtype=bool)
        excluded = np.zeros(N, dtype=bool)
        for v in order:
            if excluded[v]:
                continue
            selected[v] = True
            # Exclude neighbors
            excluded[adj[v]] = True
            excluded[v] = True
        return selected
    
    # 1. Degree-based order
    degrees = np.array([len(adj[i]) for i in range(N)])
    order = np.argsort(degrees)
    mask = run_greedy(order)
    best_size = mask.sum()
    best_mask = mask
    
    # 2. Random restarts
    rng = np.random.default_rng(42)
    for i in range(200):
        if time.time() > deadline - 30: # Leave time for local search
            break
        random_order = rng.permutation(N)
        mask = run_greedy(random_order)
        size = mask.sum()
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # 3. Local Search on the best mask found
    # We will try to improve best_mask
    in_set = best_mask.copy()
    # Precompute conflict count for each node
    # conflict_count[v] = number of neighbors of v in in_set
    conflict_count = np.zeros(N, dtype=np.int32)
    for v in np.where(in_set)[0]:
        conflict_count[adj[v]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try Add moves
        # Find v not in set with conflict_count[v] == 0
        candidates = np.where((~in_set) & (conflict_count == 0))[0]
        if len(candidates) > 0:
            # Add one (or all? adding one might block others, but if count is 0, they don't conflict with current set.
            # However, adding v might make conflict_count of others > 0.
            # Greedy add: pick one, update.
            v = candidates[0]
            in_set[v] = True
            # Update conflict counts for neighbors
            conflict_count[adj[v]] += 1
            improved = True
            continue # Restart loop to recalculate candidates
            
        # Try 2-for-1 Swap moves
        # Find u in set such that removing it allows adding at least 2 nodes.
        # For each u in set:
        #   Let N_out(u) = neighbors of u not in set.
        #   For each v in N_out(u), if conflict_count[v] == 1 (only u blocks it), it's a candidate.
        #   If we have >= 2 candidates v, w that don't conflict with each other, we can swap.
        
        set_nodes = np.where(in_set)[0]
        
        for u in set_nodes:
            if time.time() > deadline - 5:
                break
            
            # Get neighbors of u not in set
            neighbors = adj[u]
            not_in_set_neighbors = neighbors[~in_set[neighbors]]
            
            # Find candidates among not_in_set_neighbors with conflict_count == 1
            # Note: conflict_count[v] includes u if v is neighbor of u and u is in set.
            # So if conflict_count[v] == 1, it means u is the ONLY blocker.
            candidates = not_in_set_neighbors[conflict_count[not_in_set_neighbors] == 1]
            
            if len(candidates) < 2:
                continue
            
            # Check if any pair in candidates is independent (no conflict between them)
            # We just need to find one pair.
            # candidates is small (degree ~56, so candidates <= 56).
            found_swap = False
            for i in range(len(candidates)):
                v1 = candidates[i]
                # Check if v1 conflicts with any other candidate
                # We can just check against the rest of the candidates list
                # Or simply: if we pick v1, we need another v2 in candidates \ {v1} such that v1, v2 not conflict.
                # Since candidates are neighbors of u, they might conflict with each other.
                # Conflict between v1 and v2 means v2 is in adj[v1].
                # Let's just pick the first two that don't conflict.
                for j in range(i + 1, len(candidates)):
                    v2 = candidates[j]
                    # Check if v1 and v2 conflict
                    # We can check if v2 is in adj[v1]
                    # adj[v1] is a numpy array. Checking membership is O(deg).
                    # To speed up, we can precompute a boolean matrix or use sets.
                    # For now, simple check:
                    if v2 not in adj[v1]:
                        # Perform swap
                        # Remove u
                        in_set[u] = False
                        conflict_count[adj[u]] -= 1
                        
                        # Add v1, v2
                        in_set[v1] = True
                        in_set[v2] = True
                        
                        # Update conflict counts
                        # Adding v1 increases count of its neighbors
                        conflict_count[adj[v1]] += 1
                        # Adding v2 increases count of its neighbors
                        conflict_count[adj[v2]] += 1
                        
                        # Note: v1 and v2 are neighbors of u, so their conflict_count decreased by 1 when u removed.
                        # Then increased by 1 when v1/v2 added?
                        # Let's trace:
                        # Before: in_set[u]=T. conflict_count[v1] included u.
                        # Remove u: conflict_count[v1] -= 1.
                        # Add v1: conflict_count[v1] is not changed by adding itself.
                        # But conflict_count[neighbors of v1] += 1.
                        # So we must update neighbors of v1 and v2.
                        
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break # Restart outer loop
        
        # If no swap found, try a more complex move or break
        if not improved:
            # Try a random perturbation? Or break.
            # Let's try one more pass with random restarts if time is left
            if time.time() < deadline - 10:
                 # Perturb current solution
                 # Remove 5 random nodes from set
                 set_nodes = np.where(in_set)[0]
                 if len(set_nodes) > 5:
                     to_remove = rng.choice(set_nodes, size=5, replace=False)
                     for u in to_remove:
                         in_set[u] = False
                         conflict_count[adj[u]] -= 1
                     improved = True # Force another iteration
            else:
                 break

    result = vectors[in_set]
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
