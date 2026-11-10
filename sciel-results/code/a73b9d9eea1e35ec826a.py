import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 40
    
    def creates_c6(u, v, adj_mask):
        # Check if adding edge (u, v) creates a C6.
        # We need a path of length 4 between u and v in the current graph.
        # Path: u-w-x-y-z-v
        # w in N(u), z in N(v), x in N(w) \cap N(y), y in N(x) \cap N(z)
        # More efficiently:
        # For each w in N(u):
        #   For each z in N(v):
        #     If w != z and w not in N(v) and z not in N(u):
        #       Check if there is a path of length 2 between w and z in G - {u, v}
        #       Path w-x-y-z means x in N(w) \cap N(z) (excluding u,v,w,z)
        
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # Candidates for w
        w_mask = mask_u
        # Candidates for z
        z_mask = mask_v
        
        # We iterate w and z.
        # Optimization: If N(u) and N(v) are small, this is fast.
        # In dense graphs, it might be slower, but we are maximizing edges so graph becomes dense.
        # However, C6-free graphs are quite restricted.
        
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # z must be in N(v)
            # z cannot be w
            # z cannot be u (u is not in N(v) usually, but if it is, w!=u is guaranteed since w in N(u) and simple graph)
            # Actually u and v are not adjacent in the check context (we are adding them).
            # So u is not in N(v) and v is not in N(u).
            
            z_mask_tmp = z_mask & ~(1 << w)
            
            # For a fixed w, we need to know if there exists z in N(v) such that
            # dist(w, z) = 2 in G - {u, v, w, z}
            # dist(w, z) = 2 means there exists x such that x in N(w) and x in N(z).
            # x cannot be u, v, w, z.
            
            # Let's precompute N(w) excluding u, v, w.
            # Actually, let's just iterate z for each w?
            # Or iterate x?
            
            # Let's try iterating z.
            while z_mask_tmp:
                z_bit = z_mask_tmp & -z_mask_tmp
                z = z_bit.bit_length() - 1
                z_mask_tmp ^= z_bit
                
                # Check if there is x in N(w) \cap N(z) excluding {u, v, w, z}
                # x_mask = N(w) & N(z)
                # x cannot be u, v, w, z
                
                x_mask = adj_mask[w] & adj_mask[z]
                forbidden = (1 << u) | (1 << v) | (1 << w) | (1 << z)
                if x_mask & ~forbidden:
                    return True
                    
        return False

    def add_edge(u, v, adj_mask, edge_set):
        if u == v or (u, v) in edge_set or (v, u) in edge_set:
            return False
        if creates_c6(u, v, adj_mask):
            return False
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))
        return True

    def remove_edge(u, v, adj_mask, edge_set):
        if (u, v) not in edge_set and (v, u) not in edge_set:
            return
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        if (u, v) in edge_set:
            edge_set.remove((u, v))
        else:
            edge_set.remove((v, u))

    def random_partition(n, k):
        # Returns list of lists of vertices
        verts = list(range(n))
        random.shuffle(verts)
        parts = [[] for _ in range(k)]
        for i, v in enumerate(verts):
            parts[i % k].append(v)
        return parts

    def build_graph_from_partition(parts, target_edge_count_per_cluster=None):
        # Build a graph that is a union of cliques (or near cliques)
        adj_mask = [0] * n
        edge_set = set()
        
        for part in parts:
            size = len(part)
            # Make it a clique
            for i in range(size):
                for j in range(i + 1, size):
                    u = part[i]
                    v = part[j]
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edge_set.add((u, v))
                    
        return adj_mask, edge_set

    # Initial Population: Different partitions
    # 1. 8xK5
    # 2. 4xK10 (might be too dense? K10 has 45 edges. 4*45=180. Max is ~105. So K10 is too big.)
    # Wait, ex(40; C6) is around 105.
    # 8 * 10 = 80 edges.
    # If we use smaller cliques, we have room for inter-cluster edges.
    # K5 has 10 edges.
    # K4 has 6 edges. 10 * K4 = 60 edges. Room for 45 more.
    # K3 has 3 edges. 13 * K3 + 1 = 39+1. 39 edges.
    
    # Let's try a mix.
    # Strategy: Start with 8xK5 (104 edges). This is very close to the known max.
    # The "improvement" comes from finding a better arrangement of inter-cluster edges
    # or a different partition that allows more edges.
    
    # However, 8xK5 is a local optimum for "disjoint cliques".
    # Maybe a "blow-up" of a C6-free graph?
    # E.g. blow up C5? C5 is C6-free.
    # Blow up C5 with 8 vertices per node? 5 * 8 = 40.
    # Edges: 5 * (8*8) = 320? No, between clusters it's complete bipartite?
    # If we blow up C5, edges are between adjacent clusters.
    # Each pair of clusters has 8*8=64 edges. 5 pairs. 320 edges.
    # But does blow-up of C5 contain C6?
    # C5 is C6-free. Blow-up of C6-free graph is C6-free?
    # Yes, if H is C6-free, then any blow-up is C6-free.
    # Wait, ex(40; C6) is ~105. 320 is way higher.
    # Why? Because "blow-up" usually means replacing vertices with independent sets and edges with complete bipartite graphs.
    # If we replace vertices of C5 with independent sets of size 8, and edges with K_{8,8},
    # then we have 5 * 64 = 320 edges.
    # Is this graph C6-free?
    # A cycle in the blow-up projects to a closed walk in the base graph.
    # If the base graph is C5, any cycle projects to a closed walk of length multiple of 5?
    # No, a cycle of length 6 in the blow-up could project to a walk of length 2 in C5?
    # Let's check.
    # u1 in V1, u2 in V2, u3 in V1, u4 in V2, u5 in V1, u6 in V2.
    # Edges: u1-u2 (V1-V2), u2-u3 (V2-V1), u3-u4 (V1-V2), u4-u5 (V2-V1), u5-u6 (V1-V2), u6-u1 (V2-V1).
    # This is a C6!
    # So blow-up of C5 is NOT C6-free.
    # In fact, blow-up of any graph with an odd cycle will have even cycles if the bipartite connections allow it.
    # Specifically, if base graph has a path of length 2 (u-v-w), and we have independent sets, we can form u1-v1-w1-v2-u2...
    # Actually, if base graph is not bipartite, its blow-up contains C6?
    # If base graph has a triangle, blow-up has K_{3,3,3} structure?
    # It is known that ex(n, C6) is related to generalized Ramsey numbers.
    # The 8xK5 construction is a "clique partition".
    
    # Let's stick to the 8xK5 base but use GA to optimize the *inter-cluster* edges.
    # The current program does this with SA.
    # New Strategy: 
    # 1. Generate several diverse initial graphs (e.g. 8xK5, 10xK4, 13xK3+1, Random Sparse).
    # 2. Run a fast local search (Hill Climbing) on each to improve.
    # 3. Crossover: Take the edge set of two good graphs.
    #    How to crossover?
    #    "Bitwise" crossover on the edge bitmask?
    #    This might create invalid graphs (with C6s).
    #    Repair: Remove edges that create C6s.
    
    def repair_graph(adj_mask, edge_set):
        # Remove edges that create C6.
        # Iterate through edges. If removing an edge doesn't help (i.e. edge is part of C6), remove it?
        # Better: Iterate through edges. Check if edge is "safe".
        # If edge (u,v) is in graph, and removing it doesn't create a C6? No.
        # We want to keep as many edges as possible.
        # So, try to remove edges that are "critical" for C6?
        # Actually, just check if the graph is C6-free. If not, remove a random edge involved in a C6.
        
        # Find a C6.
        # This is expensive.
        # Alternative: Start with a valid graph.
        # Crossover: 
        # Take Graph A and Graph B.
        # New Graph C starts empty.
        # For each edge in A, add to C if it doesn't create C6.
        # For each edge in B, add to C if it doesn't create C6.
        # This is just a union with repair.
        
        pass

    # Let's implement the "Union with Repair" crossover.
    # It's simple and robust.
    
    def union_graph(adj1, set1, adj2, set2):
        adj_new = [0] * n
        set_new = set()
        
        # Try adding all edges from 1
        for u, v in set1:
            if not creates_c6(u, v, adj_new):
                adj_new[u] |= (1 << v)
                adj_new[v] |= (1 << u)
                set_new.add((u, v))
                
        # Try adding all edges from 2
        for u, v in set2:
            if (u, v) in set_new or (v, u) in set_new:
                continue
            if not creates_c6(u, v, adj_new):
                adj_new[u] |= (1 << v)
                adj_new[v] |= (1 << u)
                set_new.add((u, v))
                
        return adj_new, set_new

    def hill_climb(adj_mask, edge_set, time_limit):
        # Random restarts of hill climbing
        best_score = len(edge_set)
        best_adj = adj_mask[:]
        best_set = set(edge_set)
        
        while time.time() < time_limit:
            # Try to add a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or u > v:
                if u == v: continue
                u, v = v, u
            if (u, v) in edge_set:
                continue
                
            if not creates_c6(u, v, adj_mask):
                add_edge(u, v, adj_mask, edge_set)
                if len(edge_set) > best_score:
                    best_score = len(edge_set)
                    best_adj = adj_mask[:]
                    best_set = set(edge_set)
                    
            else:
                # Try to remove a random edge to unblock?
                # Or just skip.
                # Let's try a swap: Remove one edge, add another?
                # For simplicity, just skip.
                pass
                
        return best_adj, best_set

    # Initial Individuals
    individuals = []
    
    # 1. 8xK5
    parts = [[i*5+j for j in range(5)] for i in range(8)]
    adj1, set1 = build_graph_from_partition(parts)
    individuals.append((adj1, set1))
    
    # 2. 10xK4
    parts = [[i*4+j for j in range(4)] for i in range(10)]
    adj2, set2 = build_graph_from_partition(parts)
    individuals.append((adj2, set2))
    
    # 3. Random Sparse (start with 50 random edges, repair)
    adj3 = [0] * n
    set3 = set()
    for _ in range(100):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u < v and (u, v) not in set3:
            if not creates_c6(u, v, adj3):
                add_edge(u, v, adj3, set3)
    individuals.append((adj3, set3))
    
    # 4. Another 8xK5 with shuffled vertices
    verts = list(range(n))
    random.shuffle(verts)
    parts = [[verts[i*5+j] for j in range(5)] for i in range(8)]
    adj4, set4 = build_graph_from_partition(parts)
    individuals.append((adj4, set4))
    
    # Optimize each individually for a short time
    time_per_indiv = 10
    for i in range(len(individuals)):
        adj, eset = individuals[i]
        best_adj, best_eset = hill_climb(adj, eset, time.time() + time_per_indiv)
        individuals[i] = (best_adj, best_eset)
        
    # Main GA Loop
    while time.time() < deadline:
        # Select two parents (tournament selection)
        p1_idx = random.randint(0, len(individuals) - 1)
        p2_idx = random.randint(0, len(individuals) - 1)
        if p1_idx == p2_idx:
            p2_idx = (p2_idx + 1) % len(individuals)
            
        adj1, set1 = individuals[p1_idx]
        adj2, set2 = individuals[p2_idx]
        
        # Crossover
        child_adj, child_set = union_graph(adj1, set1, adj2, set2)
        
        # Mutation: Add some random edges
        for _ in range(10):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and (u, v) not in child_set:
                if not creates_c6(u, v, child_adj):
                    add_edge(u, v, child_adj, child_set)
                    
        # Local Search on Child
        # Short time limit
        child_adj, child_set = hill_climb(child_adj, child_set, time.time() + 2)
        
        # Update Population
        # Replace the worst individual if child is better
        scores = [len(s) for _, s in individuals]
        worst_idx = scores.index(min(scores))
        
        if len(child_set) > scores[worst_idx]:
            individuals[worst_idx] = (child_adj, child_set)
            
    # Find best
    best_score = 0
    best_set = set()
    for adj, eset in individuals:
        if len(eset) > best_score:
            best_score = len(eset)
            best_set = eset
            
    return [[a, b] for (a, b) in sorted(best_set)]

if __name__ == "__main__":
    construct()
