"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing the number of edges.
    
    Strategy:
    1. Start with a known K3,3-free structure (e.g., complete bipartite K2,18 has 18 edges,
       but we can do much better).
    2. Use a greedy/local search approach to add edges while maintaining the K3,3-free property.
    
    A graph is K3,3-free if and only if for every pair of vertices (u, v),
    the number of common neighbors is at most 2.
    (Because if there were 3 common neighbors, say a, b, c, then {u, v, w} on one side
     and {a, b, c} on the other side would form a K3,3, where w is any other vertex... 
     wait, no. The definition of K3,3 subgraph means there exist two disjoint sets of 3 vertices 
     such that every vertex in one set is connected to every vertex in the other.
     
     Let's re-verify the condition.
     A graph contains K3,3 if there exist two disjoint sets A={a1,a2,a3} and B={b1,b2,b3} 
     such that all edges between A and B are present.
     
     This is equivalent to: there is no pair of vertices u, v such that they have 3 or more 
     common neighbors? 
     No, that's for C4-free or something? 
     If u and v have 3 common neighbors a, b, c, then the subgraph induced by {u, v, a, b, c} 
     contains K2,3 (u and v connected to a,b,c). It does NOT necessarily contain K3,3.
     To get K3,3, we need 3 vertices on the left and 3 on the right.
     
     Let's look at the condition "no three vertices have three common neighbours".
     The prompt says: "no three vertices have three common neighbours (no K3,3)".
     Let A = {a1, a2, a3}. If they have 3 common neighbors b1, b2, b3, 
     then every ai is connected to every bj. This forms a K3,3.
     So the condition is: For any set of 3 vertices A, |N(A)| <= 2 is false? 
     No, |Intersection of N(ai) for i in A| <= 2.
     
     So the constraint is: For any distinct u, v, w, the number of common neighbors of u, v, and w is at most 2.
     
     This is a strong constraint.
     
    Let's use a heuristic search.
    1. Start with a random graph or a specific structure.
    2. Try to add edges.
    3. If adding an edge (u,v) violates the constraint, skip it.
    4. Use local search to swap edges or move edges.
    
    Given the time limit (150s), we need an efficient check.
    Checking if adding edge (u,v) creates a K3,3:
    This is tricky. Adding an edge (u,v) might complete a K3,3.
    
    Let's stick to a simpler but effective heuristic:
    Start with a bipartite graph? No, K3,3 is bipartite, so bipartite graphs can contain K3,3.
    
    Let's try a greedy construction with backtracking or local search.
    
    Actually, let's look at the score 2.25 program mentioned in the observations.
    It used `popcount`, `is_valid_addition`, `add_edge`.
    This suggests a bitmask representation for neighbors.
    
    Let's implement a local search:
    1. Represent graph as adjacency bitmasks.
    2. Start with a random graph or a known good seed.
    3. Iterate: pick a random pair (u,v) not connected.
    4. Check if adding (u,v) preserves K3,3-free.
       How to check efficiently?
       If we add (u,v), we need to ensure no new K3,3 is formed.
       A new K3,3 must involve the new edge (u,v).
       So the K3,3 must have u and v in it.
       Case 1: u and v are on the same side of the K3,3.
       Then the other side has 3 vertices, and both u and v are connected to all 3.
       So we need to check if there are 3 vertices common to N(u) and N(v).
       If |N(u) & N(v)| >= 3, then adding (u,v) creates a K3,3?
       Wait. If u and v are on the same side, say side A.
       Then A = {u, v, x}. B = {b1, b2, b3}.
       We need u-vi, v-vj, x-vk all present.
       The new edge is (u,v).
       So we need to find x such that x is connected to b1,b2,b3, and u,v are connected to b1,b2,b3.
       This means b1,b2,b3 are in N(u) & N(v) & N(x).
       So if there exists x such that |N(u) & N(v) & N(x)| >= 3, then adding (u,v) creates a K3,3 with u,v,x on one side.
       
       Case 2: u and v are on opposite sides.
       Then u is in A, v is in B.
       A = {u, a2, a3}, B = {v, b2, b3}.
       Edges: u-vi (i=1..3), uj-v (j=1..3).
       Specifically, u is connected to v, b2, b3.
       v is connected to u, a2, a3.
       And a2,a3 are connected to v, b2, b3.
       This is more complex.
       
    Let's simplify. The condition "no 3 vertices have 3 common neighbors" is the key.
    If we add edge (u,v), does it create a triple with 3 common neighbors?
    
    Let's just use a simple local search with a validity check that is a bit conservative or exact.
    
    For now, let's try to improve the seed K2,18 (18 edges) significantly.
    A known bound for ex(n, K3,3) is roughly n^2 / 4 * something? No, it's less.
    For n=20, maybe we can get 40-50 edges?
    
    Let's try a randomized local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def common_neighbors_count(u, v):
        return bin(adj[u] & adj[v]).count('1')
    
    def has_k33_with_new_edge(u, v):
        """
        Check if adding edge (u,v) creates a K3,3.
        This is an approximation/conservative check.
        If |N(u) & N(v)| >= 3, then u and v have 3 common neighbors.
        Let C = N(u) & N(v).
        If |C| >= 3, pick any 3 from C, say c1,c2,c3.
        Then u and v are both connected to c1,c2,c3.
        This forms a K2,3 between {u,v} and {c1,c2,c3}.
        Does it form a K3,3?
        We need a third vertex x on the side of u,v such that x is also connected to c1,c2,c3.
        If such x exists, then {u,v,x} and {c1,c2,c3} form a K3,3.
        So if there exists x (x != u, x != v) such that (adj[x] & (1<<c1) & (1<<c2) & (1<<c3)) != 0 for some triple in C,
        then it's invalid.
        
        Actually, simpler:
        If |N(u) & N(v)| >= 3, let C = N(u) & N(v).
        We need to check if there is any x such that x is connected to at least 3 vertices in C.
        If so, we have a K3,3.
        
        Also, what if u and v are on opposite sides?
        If we add (u,v), and there are a2,a3 such that u is connected to a2,a3 and v is connected to a2,a3... 
        This is getting complicated.
        
        Let's use a simpler sufficient condition for validity:
        A graph is K3,3-free if for every pair (u,v), |N(u) & N(v)| <= 2 is NOT sufficient.
        Counterexample: Two disjoint K2,3's sharing no vertices? No.
        
        Let's just check the triple condition directly if we can, or use a simpler heuristic.
        """
        # Conservative check:
        # If u and v have >= 3 common neighbors, it's risky.
        # But not always invalid.
        pass

    # Let's try a different approach: Simulated Annealing or Hill Climbing on edge set.
    # Objective: Maximize edges subject to no K3,3.
    # Penalty: -1000 * (number of K3,3 subgraphs)
    
    # Since counting K3,3 is hard, let's use the "3 common neighbors" constraint as a proxy?
    # No, the task says "no three vertices have three common neighbours".
    # This is EXACTLY the condition for K3,3-free?
    # Let's verify.
    # If G contains K3,3, let A={a1,a2,a3}, B={b1,b2,b3}.
    # Then a1,a2,a3 have b1,b2,b3 as common neighbors.
    # So yes, if G has K3,3, then there exist 3 vertices with 3 common neighbors.
    # Conversely, if there exist 3 vertices u,v,w with 3 common neighbors x,y,z,
    # then u,v,w are all connected to x,y,z.
    # This forms a K3,3 with parts {u,v,w} and {x,y,z}.
    # So the condition "no 3 vertices have 3 common neighbors" is EXACTLY equivalent to K3,3-free.
    
    # Great! So we just need to ensure that for every triple (u,v,w), 
    # |N(u) & N(v) & N(w)| <= 2.
    
    def is_valid_addition(u, v):
        """
        Check if adding edge (u,v) keeps the graph K3,3-free.
        New violations must involve u or v.
        Any triple containing u or v must have at most 2 common neighbors.
        """
        # Triples containing u and v: (u, v, x) for all x != u, v
        # Triples containing u but not v: (u, x, y) for x,y != u,v
        # Triples containing v but not u: (v, x, y) for x,y != u,v
        
        # When we add (u,v):
        # 1. For any x, the common neighbors of (u, v, x) gain u and v? No.
        #    N(u) gains v. N(v) gains u. N(x) unchanged.
        #    Common neighbors of (u, v, x) = (N(u) \cup {v}) & (N(v) \cup {u}) & N(x)
        #    = ((N(u) & N(v) & N(x)) | (N(u) & {u} & N(x)) ... wait.
        #    Let's just compute the intersection.
        
        # It's easier to just check the constraint for all triples involving u or v.
        # There are O(n^2) such triples. n=20, so 400 checks. Each check is O(1) with bitmasks.
        # So O(n^2) per edge addition. 20*20 = 400. Very fast.
        
        new_adj_u = adj[u] | (1 << v)
        new_adj_v = adj[v] | (1 << u)
        
        # Check triples (u, v, x)
        for x in range(n):
            if x == u or x == v: continue
            common = new_adj_u & new_adj_v & adj[x]
            if bin(common).count('1') > 2:
                return False
        
        # Check triples (u, x, y) where x,y != u,v
        # N(u) changes, N(x), N(y) don't.
        # Common = new_adj_u & adj[x] & adj[y]
        for x in range(n):
            if x == u or x == v: continue
            for y in range(n):
                if y == u or y == v or y == x: continue
                common = new_adj_u & adj[x] & adj[y]
                if bin(common).count('1') > 2:
                    return False
        
        # Check triples (v, x, y) where x,y != u,v
        for x in range(n):
            if x == u or x == v: continue
            for y in range(n):
                if y == u or y == v or y == x: continue
                common = new_adj_v & adj[x] & adj[y]
                if bin(common).count('1') > 2:
                    return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Initialize with a random graph or the seed
    # Seed: K2,18
    # Vertices 0,1 connected to 2..19
    for a in range(2):
        for b in range(2, 20):
            add_edge(a, b)
    
    current_edges = count_edges()
    best_edges = current_edges
    best_adj = adj[:]
    
    # Local Search: Try to add random edges
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random non-edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if adj[u] & (1 << v): continue
        
        if is_valid_addition(u, v):
            add_edge(u, v)
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
        else:
            # Optional: try to remove an edge from u or v and add (u,v)?
            # This is a 1-opt move.
            # Skip for now to keep it simple and fast.
            pass
            
        # Occasionally, try a random restart or perturbation?
        # Let's just keep adding edges.
        
    # Return the best graph found
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END
