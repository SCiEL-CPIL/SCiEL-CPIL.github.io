import numpy as np
import random
import time

N = 31
E = 91
DEADLINE_SECONDS = 135.0

def count_c4(adj):
    """
    Count C4s in graph represented by adjacency matrix.
    C4 count = 1/2 * sum_{i<j} C(A2[i,j], 2)
    where A2 = adj @ adj gives number of length-2 paths.
    """
    A2 = adj @ adj
    i, j = np.triu_indices(N, k=1)
    common_counts = A2[i, j]
    terms = common_counts * (common_counts - 1) // 2
    total = np.sum(terms)
    return total // 2

def _delta_c4(adj, u, v, adding):
    """
    Compute the change in C4 count when adding or removing edge (u, v).
    When adding edge (u,v), new C4s are those containing u-v and two common neighbors
    of u and v that are not directly connected... actually, let's think about it.
    
    A C4 containing edge (u,v) looks like u-a-b-v or u-b-a-v... 
    Actually, a C4 is a set of 4 vertices {w,x,y,z} forming a cycle.
    Each C4 has 4 edges. 
    
    For counting C4s via common neighbors:
    #C4 = 1/2 * sum_{i<j} C(cn(i,j), 2)
    where cn(i,j) is number of common neighbors.
    
    When we add edge (u,v), how does this affect the count?
    - It changes cn for pairs that are common neighbors of u and v... no.
    - Adding edge (u,v) means u and v become adjacent. 
    - For any pair (a,b) where a is neighbor of u and b is neighbor of v (or vice versa)...
    Actually, the cleaner way:
    cn(i,j) counts common neighbors. Adding edge (u,v) increases cn(u,v) by ... 
    No, cn(u,v) is the number of common neighbors of u and v, i.e., vertices w such that (w,u) and (w,v) are edges.
    Adding edge (u,v) does NOT change the set of common neighbors of u and v.
    It does, however, change the cn values for pairs (w,u) and (w,v) for all w:
    - For any w that is a neighbor of v (w != u), w becomes a common neighbor of u and... 
      Let's reconsider: cn(w, x) = |N(w) ∩ N(x)|.
    When we add edge (u,v):
    - N(u) gains v, N(v) gains u.
    - For any pair (w, x), cn(w,x) increases if and only if 
      {u, v} ⊆ {w, x}... no.
      cn(w,x) = |N(w) ∩ N(x)|.
      The intersection N(w) ∩ N(x) gains a new element y if y is the newly added neighbor 
      that is common to both.
      
      New neighbors: u's new neighbor is v. v's new neighbor is u.
      - cn(w, u) increases by 1 if w is a neighbor of v (w != u), because v is now in N(u), 
        and if w ∈ N(v), then v ∈ N(w) ∩ N(u). Wait, v ∈ N(u) now. 
        Is v ∈ N(w)? That requires edge (w,v). If (w,v) exists, then v ∈ N(w) ∩ N(u) now.
        So cn(w, u) increases by 1 for each neighbor w of v (w != u).
      - Similarly, cn(w, v) increases by 1 for each neighbor w of u (w != v).
      - cn(u, v) does not change because common neighbors of u and v are unchanged 
        (the edge (u,v) itself is not a "neighbor").
      - What about cn(w, x) for w, x neither u nor v? 
        No change, because u and v only become neighbors of each other.
      
    So the change in cn is:
    - For each neighbor w of v (w != u): cn(w, u) += 1
    - For each neighbor w of u (w != v): cn(w, v) += 1
    
    Let S_u = N(u) and S_v = N(v). (Before adding)
    For w in S_v (w != u): cn(w, u) increases by 1.
    For w in S_u (w != v): cn(w, v) increases by 1.
    
    The C4 count is 1/2 * sum C(cn(i,j), 2).
    When cn(i,j) increases by 1, C(cn, 2) increases by cn.
    
    So delta_C4 = 1/2 * [sum_{w in S_v, w!=u} cn(w,u) + sum_{w in S_u, w!=v} cn(w,v)]
    
    But wait, if u and v share common neighbors, say c, then c is in S_u ∩ S_v.
    For w = c: 
    - c is in S_v, so cn(c, u) increases by 1.
    - c is in S_u, so cn(c, v) increases by 1.
    
    This seems correct. Let's verify:
    The pairs whose cn changes are (w,u) for w in S_v and (w,v) for w in S_u.
    
    delta_C4 = 1/2 * (sum_{w in S_v \ {u}} cn(w,u) + sum_{w in S_u \ {v}} cn(w,v))
    
    For REMOVING edge (u,v), the delta is the negative of this, but computed AFTER removal 
    or we can just negate the add-delta if we compute it before removal.
    Actually, it's safer to compute directly.
    When removing, the cn's decrease by 1, so the change is negative.
    delta_remove = -1/2 * (sum_{w in S_v \ {u}} cn(w,u) + sum_{w in S_u \ {v}} cn(w,v))
    where S_u, S_v are the neighborhoods AFTER removal (i.e., not containing v/u respectively).
    Hmm, it's easier to just use the add formula with the state BEFORE removal,
    because removing edge (u,v) is like "undoing" the addition.
    The cn values before removal are the "post-add" values.
    So delta_remove = -1/2 * (sum_{w in S_v_after} cn_before(w,u) + sum_{w in S_u_after} cn_before(w,v))
    where S_v_after = N(v) without u, S_u_after = N(u) without v.
    But cn_before(w,u) is the cn value in the current graph (before removal).
    
    Let's simplify: I'll just recompute count_c4 if I can't trust the delta.
    Given N=31, count_c4 is quite fast. Let's benchmark it.
    """
    pass

def construct():
    start_time = time.time()
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_edges_set = set(all_edges)
    
    # Strategy: Greedy construction followed by local search
    # Or just start with random and do local search with multiple restarts.
    
    best_edges = None
    best_score = float('inf')
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Random restart
        edges = random.sample(all_edges, E)
        edge_set = set(edges)
        adj = np.zeros((N, N), dtype=int)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_c4 = count_c4(adj)
        
        if current_c4 < best_score:
            best_score = current_c4
            best_edges = edges[:]
        
        if best_score == 1:
            break
        
        # Local Search
        iterations = 0
        no_improve_count = 0
        while time.time() - start_time < DEADLINE_SECONDS - 3:
            # Pick a random edge to remove
            u1, v1 = random.choice(edges)
            
            # Pick a random non-edge to add
            # We need an efficient way to pick a non-edge.
            # Precompute list of non-edges? It changes.
            # Just pick random and check.
            max_attempts = 100
            u2 = v2 = None
            for _ in range(max_attempts):
                u2, v2 = random.choice(all_edges)
                if (u2, v2) not in edge_set:
                    break
            if u2 is None:
                continue
                
            # Perform swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            edge_set.remove((u1, v1))
            
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            edge_set.add((u2, v2))
            
            # Update edges list
            idx = edges.index((u1, v1))
            edges[idx] = (u2, v2)
            
            new_c4 = count_c4(adj)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                no_improve_count = 0
                if current_c4 < best_score:
                    best_score = current_c4
                    best_edges = edges[:]
                if current_c4 == 1:
                    break
            else:
                # Revert
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                edge_set.remove((u2, v2))
                
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                edge_set.add((u1, v1))
                
                edges[idx] = (u1, v1)
                no_improve_count += 1
                
                # If stuck for too long, random jump?
                if no_improve_count > 1000:
                    break
            
            iterations += 1
            
    if best_edges is None:
        best_edges = random.sample(all_edges, E)
        
    return best_edges

if __name__ == "__main__":
    print(construct())
