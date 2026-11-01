import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np) # 5280
    
    # Build adjacency list for vectors
    # Conflict if dot > 2
    # We can build this by iterating pairs of supports that share 3 elements.
    # Or just compute dots for all pairs? 5280^2 = 27M.
    # 27M dot products of length 11.
    # Vectorized: all_vectors_np @ all_vectors_np.T
    # This is 5280 x 5280 matrix.
    # 27M floats. ~200MB. Acceptable.
    
    # Better: iterate by support.
    # Supports sharing 3 elements.
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Map support index to vector indices
    # Vectors for support s are at s*16 to s*16+15
    
    adj = [[] for _ in range(n_total)]
    
    # For each 3-subset, we have a list of supports containing it.
    # For each pair of supports (s1, s2) in this list, they share at least 3 elements.
    # Since they are distinct 4-subsets, they share exactly 3.
    # We need to check conflicts between vectors in s1 and s2.
    
    for tri, tets in tri_to_tet.items():
        # tets is list of support indices
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
                base1 = s1 * 16
                base2 = s2 * 16
                
                # Get vectors
                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]
                
                # Compute dots between v1 and v2
                # 16x16 matrix
                d = v1 @ v2.T
                
                # Find conflicts: d > 2
                # For each i in 16, j in 16 where d[i,j] > 2:
                # adj[base1+i].append(base2+j)
                # adj[base2+j].append(base1+i)
                
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    # Initial degrees
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    # Precompute neighbor sets for fast membership testing
    adj_sets = [set(neigh) for neigh in adj]
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    def simulated_annealing_mis(adj, adj_sets, initial_deg, n_total, rng, start_time, deadline):
        """
        Simulated annealing for Maximum Independent Set.
        State: set of selected nodes.
        Moves: add a node, remove a node, or swap.
        We use a simpler approach: start with greedy, then do local search with swaps.
        """
        # Start with a greedy minimum degree solution
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected_set = set()
        
        while avail.any():
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
            
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            min_indices_rel = np.where(degs == min_deg)[0]
            pick_rel = min_indices_rel[rng.integers(len(min_indices_rel))]
            u = avail_indices[pick_rel]
            
            selected_set.add(u)
            avail[u] = False
            
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
        
        current_k = len(selected_set)
        
        # Simulated annealing phase
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.995
        T = T_start
        
        max_sa_iterations = 50000
        
        for sa_iter in range(max_sa_iterations):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            if T < T_end:
                break
            
            # Random move: try to swap one node out and one in
            # Or just try to add a node
            # Or try to remove a node and add a different one
            
            # Strategy: pick a random selected node u, remove it, then try to add a random non-selected node v
            # that is compatible with the new set.
            
            if len(selected_set) == 0:
                break
            
            # With probability 0.5, try to add a node
            # With probability 0.5, try to swap
            
            if rng.random() < 0.5 and len(selected_set) < n_total:
                # Try to add a random node
                # Pick a random node not in selected_set
                # But picking uniformly from non-selected is hard without a list.
                # Let's just pick a random node from [0, n_total)
                # If it's already in selected_set, skip.
                # If it's not, check if it conflicts with any selected node.
                
                # More efficient: pick a random node that is NOT in selected_set
                # We can maintain a list of available nodes, but it's complex to update.
                # For now, let's just pick a random node and check.
                
                # Actually, a better move for MIS is to swap:
                # Remove u from selected, add v to selected if v is compatible with selected \ {u}.
                
                pass # Implement swap below
            
            # Swap move
            u = list(selected_set)[rng.integers(len(selected_set))]
            
            # Find a candidate v to add
            # v must be compatible with all nodes in selected_set \ {u}
            # And v != u
            
            # We can pick a random neighbor of u? No, that's a conflict.
            # We need a node that is NOT a neighbor of any node in selected_set \ {u}.
            
            # Let's try a few random nodes
            found_v = None
            for _ in range(10):
                v = rng.integers(n_total)
                if v == u or v in selected_set:
                    continue
                
                # Check if v conflicts with any node in selected_set \ {u}
                conflict = False
                for w in selected_set:
                    if w == u:
                        continue
                    if v in adj_sets[w]:
                        conflict = True
                        break
                
                if not conflict:
                    found_v = v
                    break
            
            if found_v is None:
                # Try to just remove u? No, that reduces size.
                # Maybe accept a removal if temperature is high?
                # Or just skip.
                T *= cooling_rate
                continue
            
            # Check if adding v and removing u is beneficial
            # New size is same. So we need a different objective or just accept with probability.
            # For MIS, size is the objective. A swap doesn't change size.
            # We need a move that changes size.
            
            # Let's change strategy:
            # 1. Try to add a node (increases size by 1). Accept if valid.
            # 2. If can't add, try to swap (same size). Accept with prob exp(-delta/T) where delta=0?
            #    This doesn't help escape local optima if we only care about size.
            #    We need to allow temporary size decrease to escape.
            
            # Let's implement:
            # Move 1: Add a random compatible node. (Size +1)
            # Move 2: Remove a random selected node. (Size -1)
            # Move 3: Swap (Size 0)
            
            # Let's try Move 1 first.
            added = False
            # Try to add a random node
            for _ in range(5):
                v = rng.integers(n_total)
                if v in selected_set:
                    continue
                # Check compatibility
                conflict = False
                for w in selected_set:
                    if v in adj_sets[w]:
                        conflict = True
                        break
                if not conflict:
                    selected_set.add(v)
                    current_k += 1
                    added = True
                    break
            
            if not added:
                # Try Move 2: Remove a random node (to escape local optimum)
                # Only accept if temperature is high enough
                u = list(selected_set)[rng.integers(len(selected_set))]
                # Probability of accepting a size decrease
                # delta = -1 (size decreases by 1)
                # Accept with prob exp(delta/T) = exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    selected_set.remove(u)
                    current_k -= 1
                # Else skip
            
            T *= cooling_rate
        
        return selected_set
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        selected_set = simulated_annealing_mis(adj, adj_sets, initial_deg, n_total, rng, start_time, deadline)
        
        k = len(selected_set)
        
        if k > best_k:
            best_k = k
            best_selected = np.zeros(n_total, dtype=bool)
            for idx in selected_set:
                best_selected[idx] = True
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result_w4 = all_vectors_np[best_selected].astype(np.int64)
    
    # Add weight-1 vectors (norm 2, entries ±2)
    weight1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            weight1_vectors.append(v)
    
    weight1_np = np.array(weight1_vectors, dtype=np.int64)
    
    # Combine
    final_vectors = np.vstack([result_w4, weight1_np])
    
    return final_vectors
