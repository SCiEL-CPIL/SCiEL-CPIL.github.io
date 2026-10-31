import numpy as np
import time
import itertools
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Build support adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Build conflict adjacency
    adj = [[] for _ in range(n_total)]
    
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                
                base1 = s1 * 16
                base2 = s2 * 16
                
                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]
                
                d = v1 @ v2.T
                
                indices = np.argwhere(d > 2)
                for i, j in indices:
                    idx1 = base1 + i
                    idx2 = base2 + j
                    adj[idx1].append(idx2)
                    adj[idx2].append(idx1)
    
    # Convert adj to numpy arrays for faster iteration if needed, or keep lists
    # Lists are fine for sparse graphs.
    
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    # Pre-sort nodes by degree for greedy heuristic
    # We will use this order for greedy selection
    sorted_nodes = np.argsort(initial_deg)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    def run_greedy(order):
        """
        Greedy MIS using a specific order.
        """
        avail = np.ones(n_total, dtype=bool)
        selected = np.zeros(n_total, dtype=bool)
        
        for u in order:
            if not avail[u]:
                continue
            # Select u
            selected[u] = True
            avail[u] = False
            # Remove neighbors
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    # No need to update degrees for simple greedy
            
        return selected

    def local_search(selected):
        """
        Improve the solution by adding nodes with 0 conflicts and swapping.
        """
        # Initialize conflict counts
        conflict_count = np.zeros(n_total, dtype=np.int32)
        sel_indices = np.where(selected)[0]
        for s in sel_indices:
            for v in adj[s]:
                if not selected[v]:
                    conflict_count[v] += 1
        
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
                break
            
            # Find nodes with conflict_count == 0 and not selected
            # We can maintain a list of candidates, but scanning is O(N) which is ~5000, fast.
            addable = np.where((conflict_count == 0) & (~selected))[0]
            if len(addable) == 0:
                break
            
            for idx in addable:
                selected[idx] = True
                conflict_count[idx] = 0
                for v in adj[idx]:
                    if not selected[v]:
                        conflict_count[v] += 1
                improved = True
        
        # Swap phase
        # Try to remove a node and add more
        # This is expensive, so we limit it
        for swap_iter in range(20):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                break
            
            # Try removing a random selected node
            rem = sel_indices[rng.integers(len(sel_indices))]
            selected[rem] = False
            
            # Update conflict counts
            for v in adj[rem]:
                if not selected[v]:
                    conflict_count[v] -= 1
            
            # Try to add compatible nodes
            improved = True
            while improved:
                improved = False
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                
                addable = np.where((conflict_count == 0) & (~selected))[0]
                if len(addable) == 0:
                    break
                
                for idx in addable:
                    selected[idx] = True
                    conflict_count[idx] = 0
                    for v in adj[idx]:
                        if not selected[v]:
                            conflict_count[v] += 1
                    improved = True
            
            k_new = np.sum(selected)
            if k_new > np.sum(best_selected) if best_selected is not None else 0:
                # Update best if this local search result is better
                pass # We return the local searched one, caller checks
            else:
                # Revert if not improved? 
                # Actually, the function returns the modified 'selected'.
                # We should probably keep the best state.
                pass
        
        return selected

    max_restarts = 2000 # Increased because greedy is faster
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Use sorted order or shuffled sorted order
        # Shuffling the sorted order can help escape local optima
        if restart % 5 == 0:
            order = sorted_nodes
        else:
            order = sorted_nodes.copy()
            rng.shuffle(order)
            
        selected = run_greedy(order)
        
        # Local search
        selected = local_search(selected)
        
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
            
        # Early stop if we hit a known good bound? 
        # Max possible is 5280. 
        # If we reach e.g. 500, we might stop? No, keep going.

    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
