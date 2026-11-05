"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300s wall-clock limit; use 290s safety margin
    safety_margin = 10
    
    def time_left():
        return deadline - time.time() - safety_margin
    
    def check_timeout():
        return time.time() > deadline - safety_margin
    
    # Build Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    n = q
    # adj[i] is a set of neighbors of vertex i
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def has_clique_k(adj, k, vertex_set=None):
        """Check if graph has a clique of size k. If vertex_set is given, only check within that set."""
        if vertex_set is None:
            vertex_set = set(range(len(adj)))
        vs = sorted(vertex_set)
        if len(vs) < k:
            return False
        if k == 1:
            return len(vs) >= 1
        if k == 2:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        return True
            return False
        if k == 3:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common:
                                return True
            return False
        if k == 4:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common_ij = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common_ij:
                                common_ijk = common_ij & adj[m]
                                for p in vs[vs.index(m)+1:]:
                                    if p in common_ijk:
                                        return True
            return False
        if k == 5:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] in adj[vs[i]]:
                        common_ij = adj[vs[i]] & adj[vs[j]] & vertex_set
                        for m in vs[j+1:]:
                            if m in common_ij:
                                common_ijk = common_ij & adj[m]
                                for p in vs[vs.index(m)+1:]:
                                    if p in common_ijk:
                                        common_ijkl = common_ijk & adj[p]
                                        for r in vs[vs.index(p)+1:]:
                                            if r in common_ijkl:
                                                return True
            return False
        return False
    
    def has_independent_k(adj, k, vertex_set=None):
        """Check if graph has an independent set of size k. If vertex_set is given, only check within that set."""
        if vertex_set is None:
            vertex_set = set(range(len(adj)))
        # Independent set in original = clique in complement
        # Build complement adjacency within vertex_set
        vs = sorted(vertex_set)
        if len(vs) < k:
            return False
        if k == 1:
            return len(vs) >= 1
        if k == 2:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        return True
            return False
        if k == 3:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        if common_non:
                            return True
            return False
        if k == 4:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non_ij = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        for m_idx, m in enumerate(common_non_ij):
                            common_non_ijk = [x for x in common_non_ij[m_idx+1:] if x not in adj[m]]
                            for p in common_non_ijk:
                                return True
            return False
        if k == 5:
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        common_non_ij = [x for x in vs[j+1:] if x not in adj[vs[i]] and x not in adj[vs[j]]]
                        for m_idx, m in enumerate(common_non_ij):
                            common_non_ijk = [x for x in common_non_ij[m_idx+1:] if x not in adj[m]]
                            for p_idx, p in enumerate(common_non_ijk):
                                common_non_ijkl = [x for x in common_non_ijk[p_idx+1:] if x not in adj[p]]
                                for r in common_non_ijkl:
                                    return True
            return False
        return False
    
    def is_valid_addition(adj, n, new_neighbor_set):
        """Check if adding a new vertex (index n) with the given neighbors is valid."""
        # new_neighbor_set is a set of existing vertices adjacent to new vertex
        # 1. new_neighbor_set must not contain a K4 (clique of size 4) - would form K5 with new vertex
        if has_clique_k(adj, 4, new_neighbor_set):
            return False
        # 2. Non-neighbors (existing vertices not in new_neighbor_set) must not contain an independent 4-set - would form independent 5-set with new vertex
        non_neighbors = set(range(n)) - new_neighbor_set
        if has_independent_k(adj, 4, non_neighbors):
            return False
        return True
    
    def try_add_vertex(adj, n, max_attempts=50000):
        """Try to find a valid connection pattern for a new vertex n."""
        all_vertices = list(range(n))
        
        # First try: connect to all (check if valid)
        if is_valid_addition(adj, n, set(all_vertices)):
            return set(all_vertices)
        
        # Try: connect to none
        if is_valid_addition(adj, n, set()):
            return set()
        
        # Randomized search with different biases
        attempts = 0
        while attempts < max_attempts:
            if check_timeout():
                break
            # Try different biases
            bias = random.uniform(0.2, 0.8)
            S = set(v for v in all_vertices if random.random() < bias)
            if is_valid_addition(adj, n, S):
                return S
            attempts += 1
        
        # More systematic: try biased random with more attempts
        attempts = 0
        while attempts < max_attempts:
            if check_timeout():
                break
            # Try different biases
            bias = random.uniform(0.3, 0.7)
            S = set(v for v in all_vertices if random.random() < bias)
            if is_valid_addition(adj, n, S):
                return S
            attempts += 1
        
        return None
    
    # Try to extend the graph
    max_target = 45  # We know upper bound is 46, so try up to 45
    current_n = n
    
    for target_n in range(current_n + 1, max_target + 1):
        if time_left() < 5:
            break
        added = try_add_vertex(adj, current_n)
        if added is not None:
            # Add the new vertex
            new_idx = current_n
            adj.append(set(added))
            for v in added:
                adj[v].add(new_idx)
            current_n += 1
        else:
            break
    
    # Final validation: ensure no K5 and no independent 5-set
    if has_clique_k(adj, 5) or has_independent_k(adj, 5):
        # Fallback to original Paley graph if extension created invalid graph
        adj = [set() for _ in range(q)]
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        current_n = q
    
    # Build edge list
    edges = []
    for i in range(current_n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return current_n, edges
# EVOLVE-BLOCK-END
