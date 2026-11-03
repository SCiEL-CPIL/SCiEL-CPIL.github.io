import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    D = 11
    
    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build conflict graph: two supports conflict if they share >= 3 coordinates
    # Precompute for each pair of supports whether they conflict
    # Two 4-subsets share >= 3 elements iff their intersection size is 3 or 4
    
    # For efficiency, use triple-based approach
    # Generate all triples
    triples = list(itertools.combinations(range(D), 3))
    n_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # For each support, list the triple indices it contains
    sup_triple_indices = []
    for s in supports:
        ts = [tuple(t) for t in itertools.combinations(s, 3)]
        idxs = [triple_to_idx[t] for t in ts]
        sup_triple_indices.append(idxs)
    
    # Build conflict adjacency: conflict[i] is a set of support indices that conflict with i
    # Two supports conflict if they share a triple
    # Use bitsets for efficiency
    conflict = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for t_idx in sup_triple_indices[i]:
            for j in range(n_sup):
                if i == j:
                    continue
                if t_idx in sup_triple_indices[j]:
                    conflict[i].add(j)
    
    # Better: build conflict using triple-to-supports mapping
    triple_to_supports = [[] for _ in range(n_triples)]
    for i in range(n_sup):
        for t_idx in sup_triple_indices[i]:
            triple_to_supports[t_idx].append(i)
    
    conflict = [set() for _ in range(n_sup)]
    for t_idx in range(n_triples):
        sups = triple_to_supports[t_idx]
        for a in range(len(sups)):
            for b in range(a+1, len(sups)):
                i, j = sups[a], sups[b]
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Find maximum independent set using greedy + local search
    # Greedy: pick node with minimum degree in remaining graph
    def greedy_is(deadline_time):
        remaining = set(range(n_sup))
        selected = set()
        deg = [len(conflict[i]) for i in range(n_sup)]
        
        while remaining and time.time() < deadline_time:
            # Pick node with minimum degree in remaining graph
            min_deg = float('inf')
            best_node = -1
            for node in remaining:
                d = sum(1 for n in remaining if n != node and n in conflict[node])
                if d < min_deg:
                    min_deg = d
                    best_node = node
            if best_node == -1:
                break
            selected.add(best_node)
            remaining.discard(best_node)
            # Remove neighbors of best_node from remaining
            for n in list(remaining):
                if n in conflict[best_node]:
                    remaining.discard(n)
        
        return selected
    
    # Local search: try to add nodes back or swap
    def local_search(is_set, deadline_time):
        is_set = set(is_set)
        # Try to add any node that doesn't conflict with selected nodes
        for i in range(n_sup):
            if i not in is_set:
                if not any(c in is_set for c in conflict[i]):
                    is_set.add(i)
        
        # Try 1-swap: remove one selected, add two non-selected
        changed = True
        while changed and time.time() < deadline_time:
            changed = False
            selected_list = list(is_set)
            for i in selected_list:
                if time.time() > deadline_time:
                    break
                # Try removing i and adding two others
                trial = set(is_set)
                trial.discard(i)
                added = 0
                for j in range(n_sup):
                    if j not in trial and j != i:
                        if not any(c in trial for c in conflict[j]):
                            trial.add(j)
                            added += 1
                            if added >= 2:
                                break
                if len(trial) > len(is_set):
                    is_set = trial
                    changed = True
                    break
        return is_set
    
    # Run greedy multiple times with different orderings
    best_is = set()
    best_size = 0
    
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Shuffle the order of nodes for greedy
        nodes = list(range(n_sup))
        random.seed(r * 12345 + 678)
        random.shuffle(nodes)
        
        # Greedy in shuffled order
        remaining = set(nodes)
        selected = set()
        while remaining and time.time() < deadline - 10.0:
            # Pick first available node in shuffled order
            best_node = -1
            for node in nodes:
                if node in remaining:
                    best_node = node
                    break
            if best_node == -1:
                break
            selected.add(best_node)
            remaining.discard(best_node)
            for n in list(remaining):
                if n in conflict[best_node]:
                    remaining.discard(n)
        
        # Local search
        selected = local_search(selected, deadline - 10.0)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_is = set(selected)
    
    # Also try min-degree greedy without shuffle
    if time.time() < deadline - 10.0:
        min_deg_is = greedy_is(deadline - 10.0)
        min_deg_is = local_search(min_deg_is, deadline - 10.0)
        if len(min_deg_is) > best_size:
            best_size = len(min_deg_is)
            best_is = min_deg_is
    
    # Construct vectors from the independent set
    all_vectors = []
    for s_idx in best_is:
        sup = supports[s_idx]
        # All 16 sign patterns
        for bits in range(16):
            signs = [1 if (bits >> b) & 1 else -1 for b in range(4)]
            v = np.zeros(D, dtype=np.int8)
            for b, idx in enumerate(sup):
                v[idx] = signs[b]
            all_vectors.append(v)
    
    if not all_vectors:
        # Fallback: weight-2 vectors
        w2_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = si
                        v[j] = sj
                        w2_vectors.append(v)
        return np.array(w2_vectors, dtype=np.int8)
    
    vec_arr = np.array(all_vectors, dtype=np.int8)
    
    # Verify the configuration
    n = len(vec_arr)
    norms_sq = np.sum(vec_arr.astype(np.int64)**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    ok = True
    for i in range(n):
        if time.time() > deadline - 2.0:
            ok = False
            break
        for j in range(i+1, n):
            if time.time() > deadline - 2.0:
                ok = False
                break
            diff = vec_arr[i].astype(np.int64) - vec_arr[j].astype(np.int64)
            d2 = int(np.sum(diff**2))
            if d2 < max_norm_sq:
                ok = False
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to weight-2
        w2_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = si
                        v[j] = sj
                        w2_vectors.append(v)
        return np.array(w2_vectors, dtype=np.int8)
    
    return vec_arr
