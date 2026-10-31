import numpy as np
import itertools
import time

D = 11

def _greedy_mis(adj, n):
    """Greedy Maximum Independent Set: repeatedly pick lowest-degree node."""
    used = [False] * n
    # Work with a copy of degrees
    deg = [len(adj[i]) for i in range(n)]
    # Order by degree
    order = list(range(n))
    result = []
    for u in order:
        if not used[u]:
            result.append(u)
            used[u] = True
            for v in adj[u]:
                if not used[v]:
                    used[v] = True
                    deg[v] = max(0, deg[v] - 1)
    return result

def _local_improve(adj, n, initial_set):
    """Improve independent set by adding vertices that don't conflict with current set."""
    in_set = [False] * n
    for v in initial_set:
        in_set[v] = True
    
    # Try to add vertices not in set that have no neighbors in set
    added = []
    for v in range(n):
        if in_set[v]:
            continue
        # Check if v conflicts with any vertex in current set
        has_conflict = False
        for u in adj[v]:
            if in_set[u]:
                has_conflict = True
                break
        if not has_conflict:
            in_set[v] = True
            added.append(v)
    
    result = [v for v in range(n) if in_set[v]]
    return result

def _branch_and_bound_mis(adj, n, time_limit, deadline):
    """Branch and bound Maximum Independent Set using Tomita's algorithm."""
    # Precompute: for each vertex, its neighbors
    # We'll work with a candidate set and expand recursively
    
    best_set = []
    best_size = 0
    
    # Color ordering for upper bound (greedy coloring of the candidate subgraph)
    def greedy_color(candidates, adj_local):
        """Greedy coloring to get an upper bound on MIS size."""
        color = {}
        colors_used = 0
        # Process vertices in order of degree (highest first)
        degs = [(len(adj_local[v]), v) for v in candidates]
        degs.sort(reverse=True)
        
        for _, v in degs:
            if v not in color:
                # Find smallest available color
                used_colors = set()
                for u in adj_local[v]:
                    if u in color:
                        used_colors.add(color[u])
                c = 0
                while c in used_colors:
                    c += 1
                color[v] = c
                colors_used = max(colors_used, c + 1)
        return colors_used
    
    def expand(candidates, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Upper bound: |current_set| + greedy_color_bound
        bound = len(current_set) + len(candidates)
        if bound <= best_size:
            return
        
        # Get a better bound using greedy coloring
        # Build local adjacency for candidates
        cand_set = set(candidates)
        adj_local = {v: [u for u in adj[v] if u in cand_set] for v in candidates}
        color_bound = greedy_color(candidates, adj_local)
        if len(current_set) + color_bound <= best_size:
            return
        
        # Choose pivot: vertex with maximum degree in candidate subgraph
        pivot = max(candidates, key=lambda v: len(adj_local[v]))
        
        # Branch: include pivot or exclude pivot
        # Include pivot
        new_current = current_set + [pivot]
        # Remove pivot and its neighbors from candidates
        new_candidates = [v for v in candidates if v != pivot and v not in adj_local[pivot]]
        expand(new_candidates, new_current)
        
        # Exclude pivot
        new_candidates2 = [v for v in candidates if v != pivot]
        expand(new_candidates2, current_set)
    
    # Initial candidate set: all vertices
    candidates = list(range(n))
    current_set = []
    expand(candidates, current_set)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    
    vectors = []
    vector_supports = []
    
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, sig in zip(s, signs):
                v[i] = sig
            vectors.append(v)
            vector_supports.append(s)
            
    vectors_np = np.array(vectors, dtype=np.int64)
    n = len(vectors_np)
    
    # Build conflict graph
    adj = [set() for _ in range(n)]
    
    support_to_vecs = {}
    for i, s in enumerate(vector_supports):
        if s not in support_to_vecs:
            support_to_vecs[s] = []
        support_to_vecs[s].append(i)
        
    support_list = list(support_to_vecs.keys())
    
    for i in range(len(support_list)):
        s1 = support_list[i]
        vecs1 = support_to_vecs[s1]
        for j in range(i+1, len(support_list)):
            s2 = support_list[j]
            inter = set(s1).intersection(s2)
            if len(inter) != 3:
                continue
            
            vecs2 = support_to_vecs[s2]
            
            for idx1 in vecs1:
                v1 = vectors_np[idx1]
                for idx2 in vecs2:
                    v2 = vectors_np[idx2]
                    conflict = True
                    for k in inter:
                        if v1[k] != v2[k]:
                            conflict = False
                            break
                    if conflict:
                        adj[idx1].add(idx2)
                        adj[idx2].add(idx1)
    
    # Convert adj sets to lists for faster access
    adj_lists = [list(s) for s in adj]
    
    # Phase 1: Greedy MIS
    greedy_result = _greedy_mis(adj_lists, n)
    
    # Phase 2: Local improvement
    improved = _local_improve(adj_lists, n, greedy_result)
    
    best_set = improved
    best_size = len(improved)
    
    # Phase 3: Branch and bound with time limit
    if time.time() < deadline - 10.0:  # Reserve 10s for verification
        bb_result = _branch_and_bound_mis(adj_lists, n, deadline, deadline)
        if len(bb_result) > best_size:
            best_set = bb_result
            best_size = len(bb_result)
    
    # Final local improvement
    final_set = _local_improve(adj_lists, n, best_set)
    
    result = vectors_np[final_set]
    
    # Verify constraints
    if len(result) > 0:
        # Check min pairwise distance >= max norm
        norms = np.linalg.norm(result, axis=1)
        max_norm = np.max(norms)
        
        # Check pairwise distances
        min_dist = float('inf')
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                dist = np.linalg.norm(result[i] - result[j])
                if dist < min_dist:
                    min_dist = dist
        
        # If constraint violated, fall back to greedy
        if min_dist < max_norm - 1e-9:
            result = vectors_np[greedy_result]
    
    return result
