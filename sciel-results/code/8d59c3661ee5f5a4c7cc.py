"""C6-free graph on 40 vertices - bipartite decomposition approach."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    n = 40
    # Bipartition: A = vertices 0-19, B = vertices 20-39
    nA = 20
    nB = 20
    
    def build_adj_bipartite(edge_set):
        """Build adjacency bitmasks for bipartite graph.
        adj_A[i] = bitmask of B-neighbors of A-vertex i
        adj_B[j] = bitmask of A-neighbors of B-vertex j
        """
        adj_A = [0] * nA
        adj_B = [0] * nB
        for u, v in edge_set:
            if u < nA and v >= nA:
                a, b = u, v - nA
                adj_A[a] |= (1 << b)
                adj_B[b] |= (1 << a)
            elif v < nA and u >= nA:
                a, b = v, u - nA
                adj_A[a] |= (1 << b)
                adj_B[b] |= (1 << a)
        return adj_A, adj_B
    
    def count_common_neighbors_bipartite(adj_A, adj_B):
        """Count common neighbors for all pairs in A and B."""
        # For pairs in A: common neighbors in B
        common_A = np.zeros((nA, nA), dtype=np.int32)
        for i in range(nA):
            for j in range(i + 1, nA):
                common_A[i, j] = (adj_A[i] & adj_A[j]).bit_count()
        
        # For pairs in B: common neighbors in A
        common_B = np.zeros((nB, nB), dtype=np.int32)
        for i in range(nB):
            for j in range(i + 1, nB):
                common_B[i, j] = (adj_B[i] & adj_B[j]).bit_count()
        
        return common_A, common_B
    
    def is_c6_free_bipartite(adj_A, adj_B):
        """Check if bipartite graph is C6-free.
        C6-free iff no two A-vertices share 3+ common B-neighbors
        and no two B-vertices share 3+ common A-neighbors.
        """
        for i in range(nA):
            for j in range(i + 1, nA):
                if (adj_A[i] & adj_A[j]).bit_count() >= 3:
                    return False
        for i in range(nB):
            for j in range(i + 1, nB):
                if (adj_B[i] & adj_B[j]).bit_count() >= 3:
                    return False
        return True
    
    def can_add_edge_bipartite(adj_A, adj_B, a, b):
        """Check if adding edge (A:a, B:b) creates a C6.
        This creates a C6 iff there exist a'≠a in A, b'≠b in B such that
        a' is adjacent to b and b' is adjacent to a (forming a-b-a'-b'-a... wait)
        
        Actually, adding (a,b) creates a C6 iff there's a path of length 5 from a to b
        not using edge (a,b). In bipartite graph, a C6 is a-b-c-d-e-f-a.
        If we add (a,b), a C6 exists iff there are a'∈A, b'∈B, a''∈A, b''∈B, a'''∈A
        with edges (a,b), (b,b''), (b'',a''), (a'',b'''), (b''',a'), (a',a) ... 
        
        Simpler: adding (a,b) creates a C6 iff there exist a'≠a in A and b'≠b in B
        such that a'~b and a~b' and a'~b' (forming a-b-a'-b'-a? No that's C4).
        
        Let me think again. A C6 in bipartite graph: a1-b1-a2-b2-a3-b3-a1
        If we add edge (a,b) = (a1,b1), then we need a2-b2-a3-b3-a1 path of length 5.
        This means: a2~b1, a2~b2, a3~b2, a3~b3, a1~b3.
        So: there exist a2≠a1, b2, a3≠a1, b3≠b1 such that:
        - a2 is adjacent to b1 (b1)
        - a2 is adjacent to b2
        - a3 is adjacent to b2  
        - a3 is adjacent to b3
        - a1 is adjacent to b3
        
        Equivalently: there exist a2≠a in A, b3≠b in B, with a~b3 and a2~b,
        and there exists b2 such that a2~b2 and a3~b2 for some a3 with a3~b3.
        
        This is getting complex. Let me use the common neighbor criterion directly.
        
        After adding (a,b), check if any pair in A has 3+ common B-neighbors
        or any pair in B has 3+ common A-neighbors.
        """
        # Tentatively add the edge
        adj_A[a] |= (1 << b)
        adj_B[b] |= (1 << a)
        
        # Check if this creates a C6
        # Only need to check pairs involving a or b
        for j in range(nA):
            if j != a:
                if (adj_A[a] & adj_A[j]).bit_count() >= 3:
                    adj_A[a] &= ~(1 << b)
                    adj_B[b] &= ~(1 << a)
                    return False
        for j in range(nB):
            if j != b:
                if (adj_B[b] & adj_B[j]).bit_count() >= 3:
                    adj_A[a] &= ~(1 << b)
                    adj_B[b] &= ~(1 << a)
                    return False
        
        # Keep the edge
        return True
    
    def remove_edge_bipartite(adj_A, adj_B, a, b):
        adj_A[a] &= ~(1 << b)
        adj_B[b] &= ~(1 << a)
    
    def greedy_fill_bipartite(adj_A, adj_B, seed_val):
        """Greedily add all possible edges."""
        random.seed(seed_val)
        edges_added = 0
        changed = True
        while changed:
            changed = False
            for a in range(nA):
                for b in range(nB):
                    if not (adj_A[a] & (1 << b)):
                        if can_add_edge_bipartite(adj_A, adj_B, a, b):
                            edges_added += 1
                            changed = True
        return edges_added
    
    def dense_bipartite_init(seed_val, density=0.5):
        """Create dense random bipartite graph and prune C6s."""
        random.seed(seed_val)
        adj_A = [0] * nA
        adj_B = [0] * nB
        edge_set = set()
        
        for a in range(nA):
            for b in range(nB):
                if random.random() < density:
                    adj_A[a] |= (1 << b)
                    adj_B[b] |= (1 << a)
                    edge_set.add((a, nA + b))
        
        # Prune: iteratively remove edges that create C6
        changed = True
        while changed:
            changed = False
            to_remove = []
            for a in range(nA):
                for b in range(nB):
                    if adj_A[a] & (1 << b):
                        # Temporarily remove
                        adj_A[a] &= ~(1 << b)
                        adj_B[b] &= ~(1 << a)
                        # Check if adding it back creates C6
                        if can_add_edge_bipartite(adj_A, adj_B, a, b):
                            pass  # Keep it
                        else:
                            to_remove.append((a, b))
                            changed = True
                        # Note: can_add_edge_bipartite already restored if it kept it
                        # If it removed it, we need to keep it removed
            
            # Rebuild properly
            # Actually, let me redo this more carefully
            break
        
        # Better approach: start with empty, greedily add
        adj_A = [0] * nA
        adj_B = [0] * nB
        edge_set = set()
        
        # Random order
        all_edges = [(a, b) for a in range(nA) for b in range(nB)]
        random.shuffle(all_edges)
        
        for a, b in all_edges:
            if can_add_edge_bipartite(adj_A, adj_B, a, b):
                edge_set.add((a, nA + b))
        
        return edge_set, adj_A, adj_B
    
    def local_search_bipartite(edge_set, adj_A, adj_B, seed_val, max_iter=200):
        """Local search: try to add edges, occasionally remove to enable others."""
        random.seed(seed_val)
        current_edges = set(edge_set)
        current_adj_A = list(adj_A)
        current_adj_B = list(adj_B)
        current_count = len(current_edges)
        
        # Build non-edge list
        non_edges = []
        for a in range(nA):
            for b in range(nB):
                if not (current_adj_A[a] & (1 << b)):
                    non_edges.append((a, b))
        
        T = 1.0
        T_min = 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.5 and non_edges:
                # Try to add a random non-edge
                idx = random.randint(0, len(non_edges) - 1)
                a, b = non_edges[idx]
                
                if can_add_edge_bipartite(current_adj_A, current_adj_B, a, b):
                    current_edges.add((a, nA + b))
                    current_count += 1
                    non_edges.pop(idx)
            
            elif move_type < 0.7 and current_edges:
                # Remove a random edge
                idx = random.randint(0, len(current_edges) - 1)
                e = list(current_edges)[idx]
                a, b = e[0], e[1] - nA
                
                current_adj_A[a] &= ~(1 << b)
                current_adj_B[b] &= ~(1 << a)
                current_edges.remove(e)
                current_count -= 1
                non_edges.append((a, b))
                
                # Accept with probability
                if random.random() < np.exp(-1.0 / T):
                    pass  # Keep removed
                else:
                    # Revert
                    current_adj_A[a] |= (1 << b)
                    current_adj_B[b] |= (1 << a)
                    current_edges.add(e)
                    current_count += 1
                    non_edges.pop()
            
            else:
                # Swap: remove one, try to add another
                if current_edges and non_edges:
                    idx_rem = random.randint(0, len(current_edges) - 1)
                    e_rem = list(current_edges)[idx_rem]
                    a_rem, b_rem = e_rem[0], e_rem[1] - nA
                    
                    idx_add = random.randint(0, len(non_edges) - 1)
                    a_add, b_add = non_edges[idx_add]
                    
                    # Remove e_rem
                    current_adj_A[a_rem] &= ~(1 << b_rem)
                    current_adj_B[b_rem] &= ~(1 << a_rem)
                    current_edges.remove(e_rem)
                    
                    # Try to add e_add
                    if can_add_edge_bipartite(current_adj_A, current_adj_B, a_add, b_add):
                        current_edges.add((a_add, nA + b_add))
                        non_edges.pop(idx_add)
                    else:
                        # Revert removal
                        current_adj_A[a_rem] |= (1 << b_rem)
                        current_adj_B[b_rem] |= (1 << a_rem)
                        current_edges.add(e_rem)
            
            T = max(T_min, T * 0.97)
            
            # Periodic greedy fill
            if it % 30 == 0:
                added = False
                for i in range(len(non_edges)):
                    a, b = non_edges[i]
                    if can_add_edge_bipartite(current_adj_A, current_adj_B, a, b):
                        current_edges.add((a, nA + b))
                        current_count += 1
                        non_edges.pop(i)
                        added = True
                        break
                if not added:
                    # Try a few more
                    pass
        
        return current_edges, current_adj_A, current_adj_B
    
    def perturb_and_refine(edge_set, adj_A, adj_B, seed_val, num_remove=5):
        """Remove random edges, then greedy fill + local search."""
        random.seed(seed_val)
        edges = set(edge_set)
        a_adj = list(adj_A)
        b_adj = list(adj_B)
        
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            a, b = e[0], e[1] - nA
            a_adj[a] &= ~(1 << b)
            b_adj[b] &= ~(1 << a)
            edges.remove(e)
        
        # Greedy fill
        greedy_fill_bipartite(a_adj, b_adj, seed_val)
        
        # Rebuild edge set from adjacency
        new_edges = set()
        for a in range(nA):
            for b in range(nB):
                if a_adj[a] & (1 << b):
                    new_edges.add((a, nA + b))
        
        # Local search
        new_edges, a_adj, b_adj = local_search_bipartite(new_edges, a_adj, b_adj, seed_val)
        
        return new_edges, a_adj, b_adj
    
    # Main search
    best_edges = set()
    best_count = 0
    best_adj_A = [0] * nA
    best_adj_B = [0] * nB
    
    # Try multiple initializations
    if time.time() < deadline - 50:
        for seed_val in [42, 123, 777, 999, 555, 111, 222, 333]:
            if time.time() > deadline - 45:
                break
            edges, adj_A, adj_B = dense_bipartite_init(seed_val)
            if len(edges) > best_count:
                best_edges = edges
                best_count = len(edges)
                best_adj_A = adj_A
                best_adj_B = adj_B
    
    # Greedy fill on best
    greedy_fill_bipartite(best_adj_A, best_adj_B, 42)
    for a in range(nA):
        for b in range(nB):
            if best_adj_A[a] & (1 << b):
                best_edges.add((a, nA + b))
    best_count = len(best_edges)
    
    # Local search
    best_edges, best_adj_A, best_adj_B = local_search_bipartite(
        best_edges, best_adj_A, best_adj_B, 42, max_iter=300)
    best_count = len(best_edges)
    
    # Multiple perturb-and-refine restarts
    restart_seed = 100
    for i in range(80):
        if time.time() > deadline - 10:
            break
        
        num_remove = random.randint(3, 10)
        perturbed_edges, p_adj_A, p_adj_B = perturb_and_refine(
            best_edges, best_adj_A, best_adj_B, restart_seed + i, num_remove)
        
        if len(perturbed_edges) > best_count:
            best_edges = perturbed_edges
            best_count = len(perturbed_edges)
            best_adj_A = p_adj_A
            best_adj_B = p_adj_B
        
        restart_seed += 1
    
    # Final polish
    greedy_fill_bipartite(best_adj_A, best_adj_B, 999)
    final_edges = set()
    for a in range(nA):
        for b in range(nB):
            if best_adj_A[a] & (1 << b):
                final_edges.add((a, nA + b))
    
    return [list(e) for e in final_edges]
