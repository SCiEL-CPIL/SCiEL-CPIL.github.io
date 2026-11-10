import random
import itertools
import time
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
edges = []
edge_map = {}
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))
        edge_map[(u, v)] = idx
        idx += 1

def generate_q3_edges_numpy():
    q3_edges_list = []
    
    # Precompute the 24 permutation matrices (complements) for 4x4
    # Actually, just generate the edges for a canonical 4x4 Q3
    # Bipartition A={0,1,2,3}, B={4,5,6,7}
    # We need all 3-regular bipartite graphs.
    # These correspond to all 4x4 0-1 matrices with row/col sum 3.
    # There are 4! = 24 such matrices.
    
    # Let's generate the 24 edge sets for local vertices 0-7
    local_q3s = set()
    for perm in itertools.permutations(range(4)):
        # perm maps A_i to B_{perm[i]}?
        # No, the complement of the identity matrix is one.
        # The matrices are complements of permutation matrices.
        # Let P be a permutation matrix. M = J - P.
        # Edges are (i, j) where M[i][j] = 1.
        
        # Construct M
        # Actually, we can just generate the edges directly.
        # For a permutation p, the missing edges are (i, 4+p[i]).
        # So the present edges are all (i, 4+j) except (i, 4+p[i]).
        
        edge_set = set()
        for i in range(4):
            for j in range(4):
                if j != perm[i]:
                    edge_set.add((i, 4+j))
        local_q3s.add(tuple(sorted(edge_set)))
        
    # local_q3s should have 24 elements.
    
    for verts in itertools.combinations(range(N), 8):
        # Generate all 35 bipartitions
        # Choose 4 for A, rest for B
        for a_subset in itertools.combinations(range(8), 4):
            a_set = set(a_subset)
            b_set = set(range(8)) - a_set
            
            # Map local 0-3 to a_set, 4-7 to b_set?
            # We need to map the local Q3 structure to these vertices.
            # The local Q3s are defined on 0-7 with specific bipartitions.
            # We need to apply a bijection from {0..7} to verts.
            
            # This is getting complicated to map correctly.
            # Let's stick to the permutation method but optimize it.
            pass
            
    # Fallback to simple perm method if this is too complex
    # Actually, the simple perm method (40k iter) * 12k sets = 500M iter is too slow.
    # The bipartition method (35*24=840 iter) * 12k sets = 10M iter is fast.
    
    # Let's implement the bipartition mapping carefully.
    q3_edges_list = []
    seen_global = set() # To dedupe across different bipartitions?
    
    for verts in itertools.combinations(range(N), 8):
        # We will generate all 840 Q3s for this vertex set.
        # We can iterate over all 8! perms? No.
        # We can iterate over all 35 splits and 24 graphs.
        
        for a_indices in itertools.combinations(range(8), 4):
            a_verts = [verts[i] for i in a_indices]
            b_verts = [verts[i] for i in range(8) if i not in a_indices]
            
            # For each of the 24 local Q3s (defined on 0-3 and 4-7)
            # We need to map local 0-3 to a_verts and 4-7 to b_verts.
            # But the local Q3s are defined with specific bipartitions.
            # The 24 graphs cover all 3-regular bipartite graphs on 4+4.
            # So for this specific bipartition (a_verts, b_verts),
            # there are exactly 24 Q3s.
            
            for perm in itertools.permutations(range(4)):
                # Generate the 12 edges
                edge_set = set()
                for i in range(4):
                    for j in range(4):
                        if j != perm[i]:
                            # Edge between a_verts[i] and b_verts[j]
                            u = a_verts[i]
                            v = b_verts[j]
                            if u > v: u, v = v, u
                            edge_set.add((u, v))
                
                edge_tuple = tuple(sorted(edge_set))
                # Add to list
                # We don't need to dedupe globally if we generate each unique Q3 once per vertex set?
                # Different (a_indices, perm) pairs might generate the same edge set?
                # Yes, if the Q3 has symmetry.
                # But we just want to add it to the list of all Q3s.
                # Duplicates in the list are fine for the Numpy check (just redundant work).
                # But they increase memory.
                # Let's dedupe per vertex set.
                
                if edge_tuple not in seen_global: # This is wrong, seen_global should be per vertex set
                    pass
            # This logic is flawed.
            pass
            
    # Let's just use the permutation method and hope it's fast enough or optimize the inner loop in C?
    # No, let's use the bipartition method correctly.
    
    q3_edges_list = []
    
    for verts in itertools.combinations(range(N), 8):
        local_seen = set()
        for a_indices in itertools.combinations(range(8), 4):
            a_verts = [verts[i] for i in a_indices]
            b_verts = [verts[i] for i in range(8) if i not in a_indices]
            
            for perm in itertools.permutations(range(4)):
                edge_set = set()
                for i in range(4):
                    for j in range(4):
                        if j != perm[i]:
                            u = a_verts[i]
                            v = b_verts[j]
                            if u > v: u, v = v, u
                            edge_set.add((u, v))
                
                edge_tuple = tuple(sorted(edge_set))
                if edge_tuple in local_seen:
                    continue
                local_seen.add(edge_tuple)
                
                q3_edges_list.append([edge_map[e] for e in edge_tuple])
                
    return np.array(q3_edges_list, dtype=np.int32)

def construct():
    start_time = time.time()
    timeout = 115
    
    q3_edges_np = generate_q3_edges_numpy()
    
    graph_edges = np.ones(NUM_EDGES, dtype=bool)
    
    def is_valid():
        present = graph_edges[q3_edges_np]
        counts = present.sum(axis=1)
        return not np.any(counts == 12)
        
    while not is_valid():
        e_idx = random.randint(0, NUM_EDGES - 1)
        graph_edges[e_idx] = False
        
    best_edges = graph_edges.copy()
    best_count = np.sum(graph_edges)
    current_edges = graph_edges.copy()
    
    T = 1.0
    cooling = 0.999
    
    while T > 0.01 and time.time() - start_time < timeout:
        e_idx = random.randint(0, NUM_EDGES - 1)
        if current_edges[e_idx]:
            current_edges[e_idx] = False
            if not is_valid():
                current_edges[e_idx] = True
        else:
            current_edges[e_idx] = True
            if is_valid():
                new_count = np.sum(current_edges)
                if new_count > best_count:
                    best_count = new_count
                    best_edges = current_edges.copy()
            else:
                current_edges[e_idx] = False
        T *= cooling
        
    result = []
    for i in range(NUM_EDGES):
        if best_edges[i]:
            result.append(list(edges[i]))
    return result
