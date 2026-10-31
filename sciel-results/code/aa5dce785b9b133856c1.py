import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin for 150s limit
    
    D = 11
    
    vectors = []
    weights = []
    
    def add_vectors(weight):
        supports = list(itertools.combinations(range(D), weight))
        for s in supports:
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                vectors.append(np.array(v, dtype=np.int8))
                weights.append(weight)
                
    add_vectors(3)
    add_vectors(4)
    add_vectors(5)
    
    n_total = len(vectors)
    all_vectors_np = np.array(vectors)
    
    map_2 = defaultdict(list)
    
    for i, v in enumerate(vectors):
        support = tuple(np.nonzero(v)[0])
        for sub in itertools.combinations(support, 2):
            map_2[sub].append(i)
            
    adj = [[] for _ in range(n_total)]
    
    for sub, indices in map_2.items():
        if len(indices) < 2:
            continue
        ws = [weights[i] for i in indices]
        
        for u_idx in range(len(indices)):
            i = indices[u_idx]
            w_i = ws[u_idx]
            for v_idx in range(u_idx + 1, len(indices)):
                j = indices[v_idx]
                w_j = ws[v_idx]
                
                if w_i == 3 and w_j == 3:
                    thresh = 1
                elif (w_i == 3 and w_j == 4) or (w_i == 4 and w_j == 3):
                    thresh = 1
                elif (w_i == 3 and w_j == 5) or (w_i == 5 and w_j == 3):
                    thresh = 1
                elif w_i == 4 and w_j == 4:
                    thresh = 2
                elif (w_i == 4 and w_j == 5) or (w_i == 5 and w_j == 4):
                    thresh = 2
                elif w_i == 5 and w_j == 5:
                    thresh = 2
                else:
                    thresh = 100
                
                dot = np.dot(vectors[i], vectors[j])
                if dot > thresh:
                    adj[i].append(j)
                    adj[j].append(i)
                    
    for i in range(n_total):
        if adj[i]:
            adj[i] = list(set(adj[i]))
            
    initial_deg = np.array([len(adj[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected = None
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
            
        avail = np.ones(n_total, dtype=bool)
        deg = initial_deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        while avail.any():
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
                
            avail_indices = np.where(avail)[0]
            if len(avail_indices) == 0:
                break
                
            degs = deg[avail_indices]
            min_deg = np.min(degs)
            min_rel = np.where(degs == min_deg)[0]
            pick_rel = min_rel[rng.integers(len(min_rel))]
            u = avail_indices[pick_rel]
            
            selected[u] = True
            avail[u] = False
            
            for v in adj[u]:
                if avail[v]:
                    avail[v] = False
                    for w in adj[v]:
                        if avail[w]:
                            deg[w] -= 1
                            
        k = np.sum(selected)
        
        # Local search: try to add compatible nodes
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
                
            unselected = ~selected
            unselected_indices = np.where(unselected)[0]
            if len(unselected_indices) == 0:
                break
                
            sel_indices = np.where(selected)[0]
            if len(sel_indices) == 0:
                for idx in unselected_indices:
                    selected[idx] = True
                    improved = True
                break
                
            # Check conflicts with selected
            # We can do this by checking adj lists
            # For each unselected node, check if any neighbor is selected
            # This is O(E_unsel)
            
            # To speed up, we can check in batches?
            # Or just iterate.
            
            for idx in unselected_indices:
                # Check if any neighbor of idx is selected
                conflict = False
                for nb in adj[idx]:
                    if selected[nb]:
                        conflict = True
                        break
                if not conflict:
                    selected[idx] = True
                    improved = True
                    
        k = np.sum(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
            
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result = all_vectors_np[best_selected]
    return result.astype(np.int64)
