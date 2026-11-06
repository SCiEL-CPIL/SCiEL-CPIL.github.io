import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = 20
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        K3,3 exists iff there exist two disjoint sets of 3 vertices with all 9 cross-edges.
        Equivalently, for any two vertices u, v, the number of common neighbors must be at most 2?
        No, that's for K2,3.
        For K3,3: A graph contains K3,3 iff there are two disjoint triples A={a1,a2,a3} and B={b1,b2,b3}
        such that every ai is connected to every bj.
        
        When adding edge (u, v), we might create a K3,3 where u is in A and v is in B (or vice versa, or both in same set? No, sets are disjoint).
        Actually, adding (u,v) creates a K3,3 iff there exist:
        1. u in A, v in B: Then we need two other vertices a2, a3 in A \ {u} and b2, b3 in B \ {v} such that all cross edges exist.
           Specifically:
           - a2, a3 must be common neighbors of v, b2, b3? No.
           - A = {u, a2, a3}, B = {v, b2, b3}.
           - Edges required:
             u-v (added)
             u-b2, u-b3
             a2-v, a3-v
             a2-b2, a2-b3
             a3-b2, a3-b3
           
           So we need to find disjoint pairs {a2, a3} and {b2, b3} from V \ {u, v} such that:
           - a2, a3 are neighbors of v.
           - b2, b3 are neighbors of u.
           - a2, a3 are neighbors of b2, b3.
           
        2. v in A, u in B: Symmetric.
        
        3. u, v both in A? Then B must be disjoint from {u, v}.
           A = {u, v, a3}, B = {b1, b2, b3}.
           Edges:
           u-b1, u-b2, u-b3
           v-b1, v-b2, v-b3
           a3-b1, a3-b2, a3-b3
           
           So we need a3 and {b1, b2, b3} disjoint from {u, v} such that:
           - a3 is common neighbor of u, v, b1, b2, b3? No, a3 must be connected to b1, b2, b3.
           - u, v must be common neighbors of b1, b2, b3.
           
        4. u, v both in B? Symmetric to 3.
        
        This check is complex. A simpler necessary condition:
        If we add (u, v), let C = N(u) ∩ N(v).
        If |C| >= 3, say {c1, c2, c3}, then {u, v, c1} and {c2, c3, ...} ? No.
        Consider A={u, v, c1} and B={c2, c3, x}. We need x connected to u, v, c1.
        If x is in N(u) ∩ N(v) ∩ N(c1), then we have a K3,3.
        So if there exist c1, c2, c3 in N(u) ∩ N(v) and x in N(c1) ∩ N(c2) ∩ N(c3) ∩ N(u) ∩ N(v) ?
        Actually, if we have 3 common neighbors c1, c2, c3, do we automatically have a K3,3?
        A = {u, v, c1}, B = {c2, c3, x}.
        We need x connected to u, v, c1.
        Also need c2, c3 connected to u, v (yes, they are common neighbors).
        And c2, c3 connected to c1? No, B vertices don't need to be connected to each other.
        But A vertices don't need to be connected to each other.
        So we need x such that x ~ u, x ~ v, x ~ c1.
        If such an x exists (x != u, v, c1, c2, c3), then {u, v, c1} and {c2, c3, x} form a K3,3.
        Note: x must be distinct from u, v, c1, c2, c3.
        
        So a sufficient condition for invalidity:
        Let C = N(u) ∩ N(v).
        If |C| >= 3:
            Pick any 3 from C: c1, c2, c3.
            If there exists x in (N(c1) ∩ N(c2) ∩ N(c3)) \ {u, v, c1, c2, c3} such that x ~ u and x ~ v?
            Wait, if x is in N(c1) ∩ N(c2) ∩ N(c3), is it necessarily in N(u) ∩ N(v)?
            Not necessarily. But if we choose x from N(u) ∩ N(v) ∩ N(c1) ∩ N(c2) ∩ N(c3), then yes.
            Actually, if x is in N(c1) ∩ N(c2) ∩ N(c3), and we set A={u, v, c1}, B={c2, c3, x}.
            We need x ~ u and x ~ v.
            So we need x in N(u) ∩ N(v) ∩ N(c1) ∩ N(c2) ∩ N(c3).
            
        Also consider the case where u, v are in the same part.
        If u, v in A, then we need 3 common neighbors B = {b1, b2, b3} of u and v, and a third vertex a3 in A such that a3 ~ b1, b2, b3.
        So if |N(u) ∩ N(v)| >= 3, let B be any 3 of them.
        If there exists a3 in N(b1) ∩ N(b2) ∩ N(b3) \ {u, v, b1, b2, b3}, then {u, v, a3} and B form a K3,3.
        
        So the check is:
        1. Let C = N(u) ∩ N(v).
        2. If |C| >= 3:
           a. For any 3 distinct c1, c2, c3 in C:
              If there exists x in N(c1) ∩ N(c2) ∩ N(c3) ∩ N(u) ∩ N(v) \ {u, v, c1, c2, c3}, return False.
           b. For any 3 distinct c1, c2, c3 in C:
              If there exists a3 in N(c1) ∩ N(c2) ∩ N(c3) \ {u, v, c1, c2, c3}, return False.
        
        This is O(|C|^3 * n) which is manageable for n=20.
        """
        if adj[u][v]:
            return False
            
        # Get neighbors as sets for intersection
        nu = set(adj[u].nonzero()[0])
        nv = set(adj[v].nonzero()[0])
        common = nu & nv
        
        if len(common) >= 3:
            common_list = list(common)
            # Check condition 1: x in N(c1) ∩ N(c2) ∩ N(c3) ∩ N(u) ∩ N(v)
            # Since x must be in N(u) ∩ N(v), x is in common.
            # So we check if there is x in common \ {c1, c2, c3} such that x ~ c1, x ~ c2, x ~ c3.
            # This means c1, c2, c3 have a common neighbor x in common.
            
            # Check condition 2: a3 in N(c1) ∩ N(c2) ∩ N(c3)
            # a3 can be any vertex not in {u, v, c1, c2, c3}.
            
            # Iterate over triples in common
            for i in range(len(common_list)):
                for j in range(i+1, len(common_list)):
                    for k in range(j+1, len(common_list)):
                        c1, c2, c3 = common_list[i], common_list[j], common_list[k]
                        
                        # Intersection of N(c1), N(c2), N(c3)
                        nc1 = set(adj[c1].nonzero()[0])
                        nc2 = set(adj[c2].nonzero()[0])
                        nc3 = set(adj[c3].nonzero()[0])
                        inter = nc1 & nc2 & nc3
                        
                        # Remove u, v, c1, c2, c3
                        inter.discard(u)
                        inter.discard(v)
                        inter.discard(c1)
                        inter.discard(c2)
                        inter.discard(c3)
                        
                        if inter:
                            # Condition 2 is satisfied (a3 exists)
                            # We need to check if a3 is valid.
                            # Wait, if a3 exists, does it always form a K3,3?
                            # A = {u, v, a3}, B = {c1, c2, c3}.
                            # u ~ c1, c2, c3 (yes, c's in N(u))
                            # v ~ c1, c2, c3 (yes, c's in N(v))
                            # a3 ~ c1, c2, c3 (yes, a3 in inter)
                            # Disjoint? a3 not in {u, v, c1, c2, c3}. Yes.
                            return False
        
        return True

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = np.zeros((n, n), dtype=np.int8)
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                
        return adj

    def local_search(adj):
        current_adj = adj.copy()
        current_edges = count_edges(current_adj)
        
        # Try to add edges in random order
        missing_pairs = [(i, j) for i, j in all_pairs if current_adj[i][j] == 0]
        random.shuffle(missing_pairs)
        
        improved = True
        while improved:
            improved = False
            for u, v in missing_pairs:
                if time.time() > deadline:
                    return current_adj, current_edges
                if current_adj[u][v] == 0:
                    if can_add(current_adj, u, v):
                        current_adj[u][v] = 1
                        current_adj[v][u] = 1
                        current_edges += 1
                        improved = True
                        missing_pairs.remove((u, v))
            
        return current_adj, current_edges

    # Baseline
    best_adj = None
    best_edges = 0
    
    # Generate initial candidates
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        adj = greedy_construct(i)
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj.copy()
            
    # If we have time, try to improve the best solution further with more aggressive local search
    if best_adj is not None and time.time() < deadline - 10:
        # Try removing random edges to escape local optima?
        # Simple: run local search again on best_adj with different shuffles
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            adj, edges = local_search(best_adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj.copy()
                
    # Convert adjacency matrix to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
