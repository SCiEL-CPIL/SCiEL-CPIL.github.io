import time
import random
import itertools
import sys

# Q3 edges in canonical form (0..7)
_Q3_CANONICAL_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# Precompute the structure for the check
# For each edge (i, j) in Q3, we need:
# - The 2 other neighbors of i in Q3: Ni
# - The 2 other neighbors of j in Q3: Nj
# - The mapping of the remaining vertices relative to these
# Actually, the check logic is:
# If u->i, v->j:
#   a, b are images of Ni (neighbors of u in G, not v)
#   c, d are images of Nj (neighbors of v in G, not u)
#   We need to find e, f such that:
#   e is common neighbor of a, b
#   f is common neighbor of c, d
#   (e, f) is an edge
#   All 8 vertices distinct.
# Note: The specific identities of e and f depend on the Q3 structure.
# In Q3, the vertices are Z2^3.
# Let's just hardcode the check for the specific structure relative to the edge (i,j).
# Due to symmetry, all edges in Q3 are equivalent under automorphism.
# So we can just fix one edge, say (0,1), and check for that pattern.
# If (u,v) is added, we check if there exists an embedding where (u,v) maps to (0,1).
# Since any edge in Q3 is isomorphic to (0,1), checking this one case covers all possibilities
# provided we allow u to map to 0 and v to 1, OR u to 1 and v to 0?
# No, the graph is undirected. The embedding is a bijection from V_Q3 to V_G.
# If we check "does there exist an embedding such that phi(0)=u and phi(1)=v", we cover all Q3s containing edge (u,v).
# Why? Because for any Q3 subgraph containing (u,v), there is an automorphism of Q3 mapping the edge containing u,v to (0,1).
# So we only need to check the case where u corresponds to 0 and v corresponds to 1.

# Structure for edge (0,1) in Q3:
# Neighbors of 0 (excluding 1): 2, 4
# Neighbors of 1 (excluding 0): 3, 5
# Let's trace the rest:
# Vertices: 0, 1, 2, 3, 4, 5, 6, 7
# Edges:
# 0-1, 0-2, 0-4
# 1-3, 1-5
# 2-3, 2-6
# 3-7
# 4-5, 4-6
# 5-7
# 6-7

# If phi(0)=u, phi(1)=v:
# phi(2)=a, phi(4)=b (a,b in N_G(u)\{v})
# phi(3)=c, phi(5)=d (c,d in N_G(v)\{u})
# phi(6)=e, phi(7)=f
# Edges required:
# (a,c) from 2-3
# (a,e) from 2-6
# (b,d) from 4-5
# (b,e) from 4-6
# (c,f) from 3-7
# (d,f) from 5-7
# (e,f) from 6-7

# So:
# e must be in N_G(a) \cap N_G(b)
# f must be in N_G(c) \cap N_G(d)
# And (e,f) must be an edge.
# And all {u,v,a,b,c,d,e,f} must be distinct.

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    best_edges_set = set()
    best_count = 0
    
    # We will run multiple random trials
    # To improve quality, we can also try a "dense start" heuristic?
    # For now, standard random greedy.
    
    # Precompute adjacency sets for fast lookup
    # We will maintain adj as a list of sets
    
    def creates_q3(adj, u, v):
        # Check if adding (u,v) creates a Q3
        # u, v are the new edge endpoints
        # We assume (u,v) is NOT in adj yet
        
        # Get neighbors
        Nu = adj[u]
        Nv = adj[v]
        
        # Candidates for a, b (neighbors of u, not v)
        # Candidates for c, d (neighbors of v, not u)
        
        # To speed up, we can iterate over pairs in Nu and Nv
        # If degrees are high, this is O(d^4).
        # We can optimize by precomputing common neighbors?
        
        # Let's iterate over pairs (a,b) in Nu
        # and pairs (c,d) in Nv
        
        # Optimization: 
        # The condition is:
        # exists a,b in Nu\{v}, c,d in Nv\{u}, e,f in V
        # such that:
        # 1. a != b, c != d
        # 2. {a,b,c,d} disjoint from {u,v} and from each other? 
        #    a,b from Nu, c,d from Nv. 
        #    a can be in Nv? Yes. But if a is in Nv, then a is neighbor of v.
        #    In Q3, 2 (a) is not neighbor of 1 (v). So a should NOT be in Nv?
        #    Wait, the graph G might have extra edges. 
        #    The subgraph Q3 just requires the 12 edges to be present.
        #    It does NOT require non-edges to be absent.
        #    So a CAN be in Nv in G, as long as the Q3 edges exist.
        #    However, in the Q3 structure, a (vertex 2) is NOT connected to v (vertex 1).
        #    So if a is in Nv in G, it's fine, it's just an extra edge in G not part of Q3.
        #    BUT, we must ensure the vertices are distinct.
        
        # 3. e in N(a) \cap N(b)
        # 4. f in N(c) \cap N(d)
        # 5. (e,f) in E_G
        # 6. {u,v,a,b,c,d,e,f} are all distinct.
        
        # Implementation:
        # Iterate a in Nu, b in Nu, a < b (to avoid duplicates, though order matters for mapping? 
        # No, a and b map to 2 and 4. Swapping them is an automorphism? 
        # 2 and 4 are symmetric with respect to 0? 
        # 0-2, 0-4. 2-3, 2-6. 4-5, 4-6. 
        # Yes, swapping 2 and 4 is an automorphism of Q3 fixing 0 and 1? 
        # Let's check: 0->0, 1->1. 
        # 2<->4. 
        # 3 (neighbor of 2) -> must map to neighbor of 4? 
        # Neighbors of 2: 0,3,6. Neighbors of 4: 0,5,6.
        # If 2->4, then 3 (neighbor of 2, not 0) must map to neighbor of 4 (not 0, not 6?).
        # This is getting complicated. 
        # It is safer to treat (a,b) as an ordered pair or just check all combinations.
        # Since d is small, O(d^4) is fine.
        
        # Let's just iterate all pairs.
        
        # To make it faster, precompute common neighbors for all pairs of vertices?
        # common[u][v] = set of common neighbors.
        # This takes 16*16*16 operations to build/update.
        
        pass # Placeholder, will implement in main loop for simplicity of state management
        
    # We will maintain:
    # adj: list of sets
    # common: 16x16 matrix of sets (common neighbors)
    
    # Initial state: empty graph
    adj = [set() for _ in range(n)]
    common = [[set() for _ in range(n)] for _ in range(n)]
    
    def update_common(u, v):
        # When adding edge (u,v), update common neighbors for all pairs
        # common[x][y] includes u if (x,u) and (y,u) are edges
        # This is O(n^2) per edge addition.
        # n=16, so 256 ops. Very fast.
        for x in range(n):
            if x == u or x == v: continue
            if x in adj[u]:
                # u is a common neighbor of x and any neighbor of u?
                # No, u is a neighbor of x.
                # For any y in adj[u] (y != x, y != v):
                # u is a common neighbor of x and y.
                for y in adj[u]:
                    if y == x or y == v: continue
                    common[x][y].add(u)
                    common[y][x].add(u)
        
        # Also, v is a common neighbor of x and any neighbor of v
        for x in range(n):
            if x == u or x == v: continue
            if x in adj[v]:
                for y in adj[v]:
                    if y == x or y == u: continue
                    common[x][y].add(v)
                    common[y][x].add(v)

    def creates_q3_fast(adj, common, u, v):
        # Check if adding (u,v) creates a Q3
        # u, v are the new edge endpoints
        
        Nu = list(adj[u])
        Nv = list(adj[v])
        
        # We need a,b in Nu, c,d in Nv
        # To reduce checks, we can filter:
        # e must be in common[a][b]
        # f must be in common[c][d]
        
        # Iterate over pairs in Nu
        for i in range(len(Nu)):
            a = Nu[i]
            for j in range(i+1, len(Nu)):
                b = Nu[j]
                if a == v or b == v: continue # Should not happen as v not in adj[u] yet
                
                # Candidates for e
                Es = common[a][b]
                if not Es: continue
                
                # Iterate over pairs in Nv
                for k in range(len(Nv)):
                    c = Nv[k]
                    for l in range(k+1, len(Nv)):
                        d = Nv[l]
                        if c == u or d == u: continue
                        
                        # Check distinctness of a,b,c,d,u,v
                        if a in Nv or b in Nv:
                            # This doesn't invalidate the Q3, but let's check distinctness explicitly
                            pass
                            
                        distinct_vertices = {u, v, a, b, c, d}
                        if len(distinct_vertices) < 6:
                            continue
                            
                        Fs = common[c][d]
                        if not Fs: continue
                        
                        # Check if there is an edge (e,f) with e in Es, f in Fs
                        # And e,f distinct from the 6 vertices
                        for e in Es:
                            if e in distinct_vertices: continue
                            for f in Fs:
                                if f in distinct_vertices: continue
                                if e == f: continue
                                # Check if (e,f) is an edge
                                if f in adj[e]:
                                    return True
        return False

    # Randomized Greedy
    # We run several trials to find a good graph
    # To ensure we finish in time, we limit the number of trials or time per trial
    
    num_trials = 10
    time_per_trial = (deadline - start_time) / num_trials
    
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        # Reset
        adj = [set() for _ in range(n)]
        common = [[set() for _ in range(n)] for _ in range(n)]
        edges_in_graph = set()
        
        # Shuffle edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for (u, v) in shuffled_edges:
            if time.time() > start_time + time_per_trial:
                break
                
            # Check if adding (u,v) creates Q3
            if creates_q3_fast(adj, common, u, v):
                continue
            else:
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges_in_graph.add((u, v))
                update_common(u, v)
                
        # Update best
        if len(edges_in_graph) > best_count:
            best_count = len(edges_in_graph)
            best_edges_set = set(edges_in_graph)
            
    # Return best graph
    result = []
    for u, v in best_edges_set:
        result.append([u, v])
        
    # If best_count is 0 (shouldn't happen), fallback to seed
    if best_count == 0:
         return [[a, b] for a in range(3) for b in range(3, 16)]
         
    return result

# To ensure the function is defined even if there are issues, we wrap it
# But the above logic is sound.

# One optimization: The inner loop of creates_q3_fast can be optimized by checking 
# if the sets Es and Fs have any edge between them.
# We can precompute a matrix of edge existence for faster lookup.
# adj is already a set, so `f in adj[e]` is O(1).

# Let's refine the distinctness check.
# a,b are neighbors of u. c,d are neighbors of v.
# In Q3, a (2) is not connected to v (1). So if a is in Nv, it's an extra edge.
# The vertices must be distinct.
# {u, v, a, b, c, d} must be distinct.
# e must be distinct from these 6.
# f must be distinct from these 6 and e.

# The current check `if e in distinct_vertices` covers e.
# `if f in distinct_vertices` covers f.
# `if e == f` covers e != f.

# This looks correct.

# Final check on time.
# 120 edges * 10 trials = 1200 edge additions.
# Each addition: O(d^4) check + O(n^2) update.
# d ~ 10. 10^4 = 10000.
# 1200 * 10000 = 12M operations.
# Python can do ~10M-100M simple ops per second.
# Should fit in 115s easily.

if __name__ == "__main__":
    # No main block needed, just the function
    pass
