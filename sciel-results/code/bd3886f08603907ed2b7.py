import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    n = 16
    # Edge indexing
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    
    total_edges = len(idx_to_edge)
    
    # Precomputation of Q3 constraints and dominators
    # This might take time. We check deadline.
    if time.time() - start_time > DEADLINE - 10:
        # Fallback to simple greedy without precomp
        return _fallback_construct(n, start_time, DEADLINE)

    # constraints[edge_idx] = list of bitmasks (int) representing the other 11 edges of Q3s containing edge_idx
    # dominators[edge_idx] = set of edge_idx such that every Q3 containing edge_idx also contains dominator
    constraints = [[] for _ in range(total_edges)]
    dominators = [set(range(total_edges)) for _ in range(total_edges)]
    
    # Generate all Q3s
    # Iterate over all 8-subsets of vertices
    vertices = list(range(n))
    for V8 in itertools.combinations(vertices, 8):
        # Generate all 840 Q3s on V8
        # Bipartitions of V8 into A, B of size 4
        V8_list = list(V8)
        for A in itertools.combinations(V8_list, 4):
            B = tuple(v for v in V8_list if v not in A)
            # To avoid duplicates, enforce A < B in some order? 
            # Combinations generates unique sets. 
            # But (A,B) and (B,A) are the same bipartition.
            # We can just iterate all A and check if A is the "first" half?
            # Or just iterate all A, and for each, generate the 24 Q3s.
            # This will generate each Q3 twice (once for A, once for B).
            # We can divide by 2 later or just accept the duplication in precomputation (slower but simpler).
            # Actually, we can just enforce min(A) < min(B) or similar.
            if min(A) > min(B):
                continue
            
            # Generate 24 Q3s for this bipartition
            # A Q3 is defined by a permutation of B relative to A?
            # No, it's a 4x4 matrix with 0s in a permutation pattern.
            # The 0s form a permutation matrix.
            # There are 24 permutations.
            for perm in itertools.permutations(B):
                # Build edge set
                edges = set()
                for i in range(4):
                    a = A[i]
                    b = perm[i]
                    # a is NOT connected to b
                    # a is connected to all other B
                    for j in range(4):
                        if i != j:
                            u, v = a, B[j]
                            if u > v: u, v = v, u
                            edges.add((u, v))
                
                # Convert to edge indices
                edge_indices = [edge_to_idx[e] for e in edges]
                
                # Update constraints and dominators
                for e_idx in edge_indices:
                    # Create bitmask of other edges
                    mask = 0
                    for o_idx in edge_indices:
                        if o_idx != e_idx:
                            mask |= (1 << o_idx)
                    constraints[e_idx].append(mask)
                    
                    # Update dominators
                    # dominators[e_idx] = intersection of (edge_indices - {e_idx}) for all Q3s
                    # We update incrementally
                    current_dom = dominators[e_idx]
                    # new_dom = current_dom & (set(edge_indices) - {e_idx})
                    # But we can just intersect with the set of other indices
                    other_set = set(edge_indices)
                    other_set.discard(e_idx)
                    dominators[e_idx] = current_dom.intersection(other_set)
        
        if time.time() - start_time > DEADLINE - 5:
            break

    # Search Phase
    best_edges = 0
    best_graph = None
    best_adj = None
    
    # Seed 1: Current program's structure
    # Seed 2: Random
    # Seed 3: Turan T(16,3) minus some edges?
    
    seeds = []
    
    # Seed 1: Replicate current logic
    adj1 = np.zeros((n, n), dtype=bool)
    for a in range(3):
        for b in range(3, n):
            adj1[a, b] = True
            adj1[b, a] = True
    for i in range(3, 15):
        adj1[i, i+1] = True
        adj1[i+1, i] = True
    adj1[15, 3] = True
    adj1[3, 15] = True
    seeds.append(adj1)
    
    # Seed 2: Random dense
    adj2 = np.zeros((n, n), dtype=bool)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.6:
                adj2[u, v] = True
                adj2[v, u] = True
    seeds.append(adj2)
    
    # Seed 3: Turan T(16,3)
    adj3 = np.zeros((n, n), dtype=bool)
    parts = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14,15]]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj3[u, v] = True
                    adj3[v, u] = True
    seeds.append(adj3)

    random.seed(42)
    
    for trial, seed_adj in enumerate(seeds):
        if time.time() - start_time > DEADLINE:
            break
            
        adj = seed_adj.copy()
        
        # Greedy Add
        changed = True
        while changed:
            changed = False
            # Find all addable edges
            addable = []
            for e_idx in range(total_edges):
                u, v = idx_to_edge[e_idx]
                if not adj[u, v]:
                    # Check if adding creates Q3
                    # A Q3 is created if there is a mask in constraints[e_idx] that is a subset of current edges
                    # current_edges_mask is expensive to build every time.
                    # Instead, we can check:
                    # For each mask in constraints[e_idx], check if (mask & ~current_mask) == 0
                    # We need current_mask.
                    pass
            
            # Optimization: Maintain a bitset of current edges
            # But Python sets of edges are slower than bitsets.
            # Let's use a list of edge indices in the graph.
            
            # Re-evaluating: The check "is there a Q3 containing e_idx in G?"
            # is equivalent to:
            # exists mask in constraints[e_idx] such that mask is subset of E(G).
            # We can precompute for each e_idx the list of masks.
            # To check subset, we need E(G) as a bitset.
            
            # Let's maintain G as a list of edge indices.
            # And a set for fast lookup?
            
            # For speed, let's just iterate.
            
            # Actually, we can just use the adj matrix and check the 11 edges directly?
            # No, 11 edges check is fast.
            # For each mask in constraints[e_idx]:
            #   unpack mask to 11 edges? No, mask is an int.
            #   We can check if all bits in mask are set in G_mask.
            
            # Let's maintain G_mask (int of 120 bits).
            
        # This approach is getting complicated. Let's simplify.
        
        # Simple Greedy:
        # Iterate all missing edges.
        # For each, check if it creates a Q3.
        # Check: 
        #   For each Q3 containing this edge (represented by the 11 other edges):
        #      If all 11 edges are in G, then adding creates a Q3.
        
        # We can precompute the 11 edges for each constraint as a tuple of indices.
        # Then checking is: all(e in G_set for e in tuple)
        
        # Let's convert constraints to tuples of indices.
        # This takes time but makes the check faster?
        
        pass

    # Fallback
    return _fallback_construct(n, start_time, DEADLINE)

def _fallback_construct(n, start_time, DEADLINE):
    # Simple greedy from the seed program
    adj = np.zeros((n, n), dtype=bool)
    for a in range(3):
        for b in range(3, n):
            adj[a, b] = True
            adj[b, a] = True
    for i in range(3, 15):
        adj[i, i+1] = True
        adj[i+1, i] = True
    adj[15, 3] = True
    adj[3, 15] = True
    
    R = list(range(3, 16))
    adj_R = np.zeros((13, 13), dtype=bool)
    for i in range(13):
        for j in range(13):
            adj_R[i, j] = adj[R[i], R[j]]
    
    def check_q3_after_add(adj_R, i, j):
        others = [k for k in range(13) if k != i and k != j]
        for four in itertools.combinations(others, 4):
            six_set = [i, j] + list(four)
            for part1 in itertools.combinations(six_set, 3):
                part2 = tuple(v for v in six_set if v not in part1)
                ok = True
                for a in part1:
                    deg = sum(1 for b in part2 if adj_R[a, b])
                    if deg != 2:
                        ok = False
                        break
                if not ok: continue
                for b in part2:
                    deg = sum(1 for a in part1 if adj_R[a, b])
                    if deg != 2:
                        ok = False
                        break
                if not ok: continue
                return True
        for six in itertools.combinations(others, 6):
            eight_set = [i, j] + list(six)
            for part1 in itertools.combinations(eight_set, 4):
                part2 = tuple(v for v in eight_set if v not in part1)
                ok = True
                for a in part1:
                    deg = sum(1 for b in part2 if adj_R[a, b])
                    if deg != 3:
                        ok = False
                        break
                if not ok: continue
                for b in part2:
                    deg = sum(1 for a in part1 if adj_R[a, b])
                    if deg != 3:
                        ok = False
                        break
                if not ok: continue
                return True
        return False

    candidates = [(i, j) for i in range(13) for j in range(i+1, 13) if not adj_R[i, j]]
    random.seed(42)
    random.shuffle(candidates)
    
    for (i, j) in candidates:
        if time.time() > DEADLINE:
            break
        if adj_R[i, j]:
            continue
        adj_R[i, j] = True
        adj_R[j, i] = True
        if check_q3_after_add(adj_R, i, j):
            adj_R[i, j] = False
            adj_R[j, i] = False
            
    edges = []
    for a in range(3):
        for b in range(3, n):
            edges.append([a, b])
    for i in range(13):
        for j in range(i+1, 13):
            if adj_R[i, j]:
                edges.append([R[i], R[j]])
    return edges
