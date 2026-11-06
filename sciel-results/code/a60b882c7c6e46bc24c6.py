import numpy as np
import time

def _jc_distance_matrix_fast(seq_array):
    n, L = seq_array.shape
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_array[i] != seq_array[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _nj_tree_fast(dist_matrix):
    n = dist_matrix.shape[0]
    if n <= 2:
        return []
    
    max_nodes = 2 * n
    d = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    d[:n, :n] = dist_matrix
    
    active = list(range(n))
    masks = np.zeros(max_nodes, dtype=np.int64)
    for i in range(n):
        masks[i] = 1 << i
        
    splits = []
    all_mask = (1 << n) - 1
    
    while len(active) > 3:
        m = len(active)
        idx = np.array(active)
        d_sub = d[np.ix_(idx, idx)]
        
        r = d_sub.sum(axis=1) - np.diag(d_sub)
        
        Q = (m - 2) * d_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        a = active[i]
        b = active[j]
        
        new_id = n + len(splits)
        
        masks[new_id] = masks[a] | masks[b]
        
        s = masks[new_id]
        comp = all_mask ^ s
        split_mask = s if s < comp else comp
        splits.append(split_mask)
        
        for k in active:
            if k != a and k != b:
                d[new_id, k] = 0.5 * (d[a, k] + d[b, k] - d[a, b])
                d[k, new_id] = d[new_id, k]
                
        active.remove(a)
        active.remove(b)
        active.append(new_id)
        
    return splits

def _splits_to_newick(splits, n):
    all_mask = (1 << n) - 1
    
    def recursive_build(taxa_mask):
        found_split = None
        for s in splits:
            side1 = s & taxa_mask
            side2 = (all_mask ^ s) & taxa_mask
            if side1 != 0 and side2 != 0:
                found_split = s
                break
                
        if not found_split:
            taxon_idx = (taxa_mask.bit_length() - 1)
            return f"t{taxon_idx}"
            
        left_mask = found_split & taxa_mask
        right_mask = (all_mask ^ found_split) & taxa_mask
        
        left_str = recursive_build(left_mask)
        right_str = recursive_build(right_mask)
        
        return f"({left_str},{right_str})"
        
    return recursive_build(all_mask)

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    seq_array = np.array([[base_map[c] for c in seq] for seq in seq_list], dtype=np.int8)
    
    orig_dist = _jc_distance_matrix_fast(seq_array)
    orig_splits = _nj_tree_fast(orig_dist)
    
    num_bootstraps = 20
    
    split_freq = {}
    
    for i in range(num_bootstraps):
        if time.time() > deadline - 5.0:
            break
            
        boot_idx = np.random.choice(L, L, replace=True)
        boot_seq_array = seq_array[:, boot_idx]
        
        boot_dist = _jc_distance_matrix_fast(boot_seq_array)
        boot_splits = _nj_tree_fast(boot_dist)
        
        for s in boot_splits:
            split_freq[s] = split_freq.get(s, 0) + 1
            
    all_splits = set(split_freq.keys())
    all_splits.update(orig_splits)
    
    scored_splits = [(split_freq.get(s, 0), s) for s in all_splits]
    scored_splits.sort(key=lambda x: x[0], reverse=True)
    
    consensus_splits = []
    all_mask = (1 << n) - 1
    
    for score, s in scored_splits:
        if len(consensus_splits) == n - 3:
            break
            
        compatible = True
        for cs in consensus_splits:
            inter = s & cs
            if inter != 0 and inter != s and inter != cs and (s | cs) != all_mask:
                compatible = False
                break
                
        if compatible:
            consensus_splits.append(s)
            
    if len(consensus_splits) < n - 3:
        for s in orig_splits:
            if len(consensus_splits) == n - 3:
                break
            if s in consensus_splits:
                continue
            compatible = True
            for cs in consensus_splits:
                inter = s & cs
                if inter != 0 and inter != s and inter != cs and (s | cs) != all_mask:
                    compatible = False
                    break
            if compatible:
                consensus_splits.append(s)
                
    if len(consensus_splits) < n - 3:
        consensus_splits = orig_splits
        
    newick = _splits_to_newick(consensus_splits, n)
    
    return newick
