import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    # Using a 16x16 matrix for fast lookup
    edge_idx_matrix = np.zeros((n, n), dtype=np.int32)
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_matrix[u, v] = idx
            edge_idx_matrix[v, u] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1

    # Q3 abstract edges
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8
    # local_edge_index[u, v] for u < v
    local_edge_idx_matrix = np.zeros((8, 8), dtype=np.int32)
    local_inv_edge_map = [None] * 28
    l_idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_idx_matrix[u, v] = l_idx
            local_edge_idx_matrix[v, u] = l_idx
            local_inv_edge_map[l_idx] = (u, v)
            l_idx += 1

    # Precompute all local Q3 edge sets (as arrays of local edge indices)
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_idx_matrix[u_val, v_val]
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(tuple(sorted(edges_set)))

    # Convert to NumPy array of shape (840, 12)
    local_q3_array = np.array(list(local_q3_edge_sets), dtype=np.int32)
    num_local_q3s = local_q3_array.shape[0]

    # Precompute global Q3s
    # We will store them in a list of arrays, then concatenate
    all_global_q3s = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge (u,v) indices for vectorization
    # We need to map local edge indices to global edge indices for a given subset
    # local_u, local_v are indices 0..7
    local_u, local_v = np.triu_indices(8, k=1)
    local_edge_indices = local_edge_idx_matrix[local_u, local_v] # shape (28,)

    for subset in all_subsets:
        s = np.array(subset, dtype=np.int32)
        # Map local edges to global edges
        # s[local_u] gives global u for each local edge
        glob_u = s[local_u]
        glob_v = s[local_v]
        # Get global edge indices
        glob_edge_indices = edge_idx_matrix[glob_u, glob_v] # shape (28,)
        
        # For each local Q3, get the corresponding global edge indices
        # local_q3_array is (840, 12)
        # glob_edge_indices is (28,)
        # We want to index glob_edge_indices with local_q3_array
        global_q3s_for_subset = glob_edge_indices[local_q3_array] # shape (840, 12)
        all_global_q3s.append(global_q3s_for_subset)

    if time.time() > deadline:
        # Fallback
        return [[i, i+1] for i in range(0, 16, 2)]

    # Concatenate all global Q3s
    Q3_EDGES = np.vstack(all_global_q3s) # shape (N_TOTAL, 12)
    N_TOTAL = Q3_EDGES.shape[0]
    
    # Build Q3_OF_EDGE: list of arrays
    # Initialize list of lists
    q3_of_edge = [[] for _ in range(num_edges_total)]
    
    # Iterate over all Q3s and add their indices to the lists for their edges
    # This is O(N_TOTAL * 12)
    # N_TOTAL is ~10.8M. 10.8M * 12 = 130M operations.
    # This might be slow in Python loop.
    # Let's try to vectorize or use a more efficient method.
    
    # Alternative: Use np.unique with return_inverse? No, we need lists.
    # We can flatten Q3_EDGES and use np.searchsorted or similar?
    # Let's just do the loop, but maybe it's fast enough?
    # 130M simple operations in Python ~ 10-20s.
    # This is risky.
    
    # Optimization: Only build Q3_OF_EDGE for edges that are likely to be removed?
    # No, we need it for the greedy step.
    
    # Let's try to speed up the loop.
    # Instead of appending to lists, we can count sizes first, then allocate arrays.
    # But we don't know the sizes without iterating.
    
    # Let's try a different approach for the greedy step that doesn't require Q3_OF_EDGE.
    # We can maintain a count of "active" Q3s for each edge.
    # Initially, count[e] = number of Q3s containing e.
    # We can compute this by summing over Q3_EDGES.
    # counts = np.bincount(Q3_EDGES.flatten(), minlength=num_edges_total)
    
    # When we remove an edge `e`, we need to decrement counts for all edges in Q3s containing e.
    # This requires knowing which Q3s contain e.
    # So we still need Q3_OF_EDGE.
    
    # Let's optimize the construction of Q3_OF_EDGE.
    # We can use a single pass over Q3_EDGES.
    # For i in range(N_TOTAL):
    #    for e in Q3_EDGES[i]:
    #        q3_of_edge[e].append(i)
    
    # This is 10.8M * 12 = 130M appends.
    # Python list append is fast, but 130M is a lot.
    
    # Can we avoid storing Q3_OF_EDGE?
    # If we store Q3_EDGES, we can scan it to find Q3s containing e?
    # That's O(N_TOTAL) per edge, which is too slow for the greedy loop.
    
    # Let's assume the construction takes ~10s.
    # We have 115s budget.
    
    q3_of_edge = [[] for _ in range(num_edges_total)]
    # Use a local variable for speed
    qoe = q3_of_edge
    qe = Q3_EDGES
    for i in range(N_TOTAL):
        for e in qe[i]:
            qoe[e].append(i)
            
    # Convert lists to arrays for faster indexing later?
    # Arrays are contiguous, lists are pointers.
    # For `np.add.at`, we need an array of indices.
    q3_of_edge_arrays = [np.array(lst, dtype=np.int32) for lst in q3_of_edge]

    if time.time() > deadline:
        return [[i, i+1] for i in range(0, 16, 2)]

    # Initialize counts and hits
    # counts[e] = number of active Q3s containing e
    # Initially, all Q3s are active.
    counts = np.bincount(Q3_EDGES.flatten(), minlength=num_edges_total).astype(np.int32)
    
    # hits_per_q3: number of removed edges in this Q3
    hits_per_q3 = np.zeros(N_TOTAL, dtype=np.int32)
    
    removed = np.zeros(num_edges_total, dtype=bool)
    
    # Greedy Loop
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find edge with max count among non-removed
        # Set counts of removed edges to -1
        counts_temp = counts.copy()
        counts_temp[removed] = -1
        best_edge = np.argmax(counts_temp)
        
        if counts_temp[best_edge] <= 0:
            break
            
        # Remove best_edge
        removed[best_edge] = True
        
        # Get Q3s containing best_edge
        q3_indices = q3_of_edge_arrays[best_edge]
        
        # Update hits_per_q3
        np.add.at(hits_per_q3, q3_indices, 1)
        
        # Check if all Q3s are hit
        if np.all(hits_per_q3 > 0):
            break
            
        # Update counts for edges in these Q3s
        # affected_edges = Q3_EDGES[q3_indices]
        # np.add.at(counts, affected_edges.flatten(), -1)
        
        # This is the expensive part.
        # affected_edges has shape (len(q3_indices), 12)
        # We can flatten it.
        affected_edges = Q3_EDGES[q3_indices].flatten()
        np.add.at(counts, affected_edges, -1)

    # Polish Phase
    # Try to add back edges
    # For each removed edge a, check if it can be added back.
    # It can be added back if no Q3 is "unhit" by R \ {a}.
    # i.e., for all Q in Q3s containing a, Q contains some other edge in R.
    
    removed_list = np.where(removed)[0]
    
    # We will iterate through removed edges and try to remove them from the removed set.
    # To be safe, we can do this in a loop until no more improvements.
    
    polish_deadline = deadline - 2
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for a in removed_list:
            if not removed[a]:
                continue
                
            # Get Q3s containing a
            q3_indices = q3_of_edge_arrays[a]
            
            # Get all edges in these Q3s
            edges_in_q3s = Q3_EDGES[q3_indices] # shape (M, 12)
            
            # Check if any Q3 has only 'a' from R.
            # Create a mask of removed edges
            # We can check: for each row in edges_in_q3s, does it contain any edge from R \ {a}?
            
            # Efficient check:
            # Count how many edges from R are in each Q3.
            # We can use np.isin or boolean indexing.
            
            # Since R is a boolean array, we can do:
            # in_R = removed[edges_in_q3s] # shape (M, 12)
            # But we want to exclude 'a'.
            # in_R_without_a = in_R & (edges_in_q3s != a)
            
            # This creates a large temporary array (M * 12).
            # M is ~90,000. 90,000 * 12 = 1,080,000 booleans. ~1MB.
            # This is fast.
            
            in_R = removed[edges_in_q3s]
            # Exclude a
            in_R_without_a = in_R & (edges_in_q3s != a)
            
            # If any row has sum 0, then that Q3 is only hit by a.
            # If all rows have sum > 0, then a is not the only hit for any Q3.
            # Wait, if a Q3 contains a and NO other edge from R, then sum is 0.
            # So if np.any(np.sum(in_R_without_a, axis=1) == 0), then we CANNOT add a back.
            # If np.all(np.sum(in_R_without_a, axis=1) > 0), then we CAN add a back.
            
            sums = np.sum(in_R_without_a, axis=1)
            if np.all(sums > 0):
                # We can add a back
                removed[a] = False
                improved = True
                # Update counts and hits?
                # If we add a back, we are effectively un-removing it.
                # This changes the state.
                # For simplicity, we might just restart the greedy or just accept the improvement.
                # Since we are in a polish phase, we can just mark it as not removed.
                # The final result is determined by `removed`.
                # We don't strictly need to update `counts` or `hits_per_q3` unless we do more greedy steps.
                # But we are just doing polish.
                
    # The remaining edges are the solution
    result_edges = []
    for e in range(num_edges_total):
        if not removed[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
