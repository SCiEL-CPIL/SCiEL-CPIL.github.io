import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Precompute edge indices
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_index(u, v)
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)

    # Local edge index for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

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

    # Precompute local Q3 edge sets
    # There are 840 distinct Q3 edge sets in K8
    local_q3_edge_sets = set()
    # We need to iterate permutations to find all distinct edge sets
    # 8! = 40320 permutations.
    # To speed up, we can just generate them and deduplicate.
    for p in __import__('itertools').permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(tuple(sorted(edges_set)))
    
    local_q3_edges = np.array(list(local_q3_edge_sets), dtype=np.int32)
    num_local_q3s = local_q3_edges.shape[0]

    # Precompute global Q3 edges
    all_subsets = list(combinations(range(n), 8))
    
    # Store all global Q3 edge lists in a list of arrays, then vstack
    chunks = []
    for subset in all_subsets:
        # Map local edge (u,v) to global edge index
        l2g = np.zeros(28, dtype=np.int32)
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
        
        # Apply mapping to all local Q3s
        # local_q3_edges is (840, 12)
        # l2g is (28,)
        # Result is (840, 12)
        global_edges_for_subset = l2g[local_q3_edges]
        chunks.append(global_edges_for_subset)
        
        if time.time() > deadline:
            # Fallback
            return [[i, i+1] for i in range(0, 16, 2)]

    all_q3_edges = np.vstack(chunks)
    num_q3s = all_q3_edges.shape[0]

    if time.time() > deadline:
        return [[i, i+1] for i in range(0, 16, 2)]

    # Build CSR-like structure for edge -> q3s
    flat_edges = all_q3_edges.ravel()
    flat_q3s = np.repeat(np.arange(num_q3s), 12)
    
    order = np.argsort(flat_edges, kind='mergesort')
    sorted_edges = flat_edges[order]
    sorted_q3s = flat_q3s[order]
    
    # Find boundaries for each edge
    edge_boundaries = np.searchsorted(sorted_edges, np.arange(num_edges_total + 1))
    
    # Initialize counts
    counts = np.zeros(num_edges_total, dtype=np.int32)
    for e in range(num_edges_total):
        start = edge_boundaries[e]
        end = edge_boundaries[e+1]
        counts[e] = end - start

    removed_edges = np.zeros(num_edges_total, dtype=bool)
    active_q3s = np.ones(num_q3s, dtype=bool)

    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        # Only consider non-removed edges
        current_counts = counts.copy()
        current_counts[removed_edges] = -1
        best_edge = np.argmax(current_counts)
        max_count = current_counts[best_edge]
        
        if max_count <= 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Get Q3s containing best_edge
        start = edge_boundaries[best_edge]
        end = edge_boundaries[best_edge + 1]
        q3_indices = sorted_q3s[start:end]
        
        # Update active Q3s and counts
        for q3_idx in q3_indices:
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                # Get edges of this Q3
                q3_edges = all_q3_edges[q3_idx]
                for e_idx in q3_edges:
                    if e_idx != best_edge:
                        counts[e_idx] -= 1

    # Local Search Polish
    # current_mask: bool array of present edges
    current_mask = ~removed_edges
    
    # List of removed edges to try adding back
    removed_list = np.where(removed_edges)[0]
    
    # Sort by "safety" (fewer Q3s containing them first)
    # We can approximate safety by the initial count
    removed_list = removed_list[np.argsort(counts[removed_list])]
    
    improved = True
    while improved:
        improved = False
        for e in removed_list:
            if time.time() > deadline:
                break
            if not removed_edges[e]:
                continue
                
            # Check if adding e is safe
            safe = True
            start = edge_boundaries[e]
            end = edge_boundaries[e + 1]
            q3_indices = sorted_q3s[start:end]
            
            for q3_idx in q3_indices:
                if not active_q3s[q3_idx]:
                    # This Q3 is already hit by other removed edges?
                    # Wait, active_q3s tracks if the Q3 is "unhit" in the current graph.
                    # If active_q3s[q3_idx] is False, it means the Q3 is already broken.
                    # So adding e to a broken Q3 doesn't create a new Q3.
                    continue
                
                # Q3 is currently active (unbroken).
                # If we add e, does it become complete?
                # It becomes complete if all OTHER edges are present.
                q3_edges = all_q3_edges[q3_idx]
                other_edges = q3_edges[q3_edges != e]
                
                # Check if all other_edges are in current_mask
                if np.all(current_mask[other_edges]):
                    safe = False
                    break
            
            if safe:
                # Add edge e
                removed_edges[e] = False
                current_mask[e] = True
                improved = True
                # Update active_q3s?
                # Adding e might "unbreak" some Q3s? No, adding edges only completes Q3s.
                # It doesn't break them. So active_q3s remains valid?
                # Wait. If a Q3 was broken because edge X was removed, and we add edge Y,
                # the Q3 is still broken if X is still removed.
                # So active_q3s (meaning "this Q3 is present in the graph") is False for broken Q3s.
                # If we add an edge, a Q3 that was broken might become complete?
                # No, a Q3 is complete if ALL its edges are present.
                # If it was broken (some edge missing), adding another edge doesn't make it complete
                # unless the missing edge was the one we just added?
                # But we are adding e. If the Q3 was broken because e was missing,
                # and all other edges were present, then it becomes complete.
                # But we checked `safe`. If it becomes complete, it's NOT safe.
                # So if it's safe, it means no Q3 becomes complete.
                # So active_q3s doesn't change?
                # Actually, active_q3s was used to skip Q3s that were already broken.
                # If a Q3 is broken, it stays broken.
                # So active_q3s is static during the "add back" phase?
                # No, if we add an edge, we don't break any Q3s. We only risk completing them.
                # So the set of "broken" Q3s only grows or stays same.
                # So active_q3s (unbroken) only shrinks or stays same.
                # So we don't need to update active_q3s.

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
