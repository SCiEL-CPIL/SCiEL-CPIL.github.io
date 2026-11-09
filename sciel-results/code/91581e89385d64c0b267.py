"""
Genetic Algorithm for R(4,6) lower bound.
Maximizes n for a K4-free graph with no independent set of size 6.
"""
import time
import random
import numpy as np

_DEADLINE = 145.0  # Safety margin before 150s kill

def _count_k4(adj_matrix):
    """
    Count K4s in a graph given as an adjacency matrix (numpy array of 0/1).
    A K4 is a set of 4 vertices all connected to each other.
    Efficient check: For each vertex v, count triangles in its neighborhood.
    Each K4 is counted 4 times (once for each vertex in the K4).
    """
    n = adj_matrix.shape[0]
    if n < 4:
        return 0
    
    # Use bitmasks for faster intersection
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        mask = 0
        # We only care about neighbors j > i for triangle counting to avoid double counting?
        # No, standard triangle count in neighborhood is fine.
        # Let's just use numpy operations on the matrix directly for simplicity, 
        # or convert to lists of ints for bit ops if n is small enough.
        # For n <= 40, uint64 is perfect.
        
        # Construct mask
        row = adj_matrix[i]
        # Convert row to int
        # This is slow in python loop. 
        # Alternative: Just use matrix multiplication.
        # Number of triangles = 1/6 * trace(A^3)
        # But we need count of K4s.
        # K4 count = 1/24 * sum_{v} (number of triangles in N(v))
        
        pass

    # Vectorized approach:
    # Triangles in G: T = (A^3)_{ii} / 6
    # K4s: For each vertex v, the number of K4s containing v is the number of triangles in G[N(v)].
    # Let A be the adjacency matrix.
    # Triangles in G[N(v)] is 1/6 * trace((A_{N(v)})^3).
    # This is computationally heavy if done naively for each v.
    
    # Simpler heuristic for GA fitness:
    # 1. Count triangles in G. If 0, no K4.
    # 2. If triangles exist, check if any triangle is part of a K4.
    # Actually, for the GA, we just need a penalty. 
    # Penalty = #K4s + #Indep6s.
    
    # Let's implement a fast check using bitmasks for n <= 64.
    masks = []
    for i in range(n):
        m = 0
        # adj_matrix is 0/1
        # We can pack the row into an integer
        # But doing it in a python loop is slow.
        # Let's stick to numpy matrix ops for the count, it's O(n^3) which is fine for n=35 (42k ops).
        pass
        
    # Numpy approach for K4 count:
    # A K4 exists if there are 4 vertices i,j,k,l with all pairs connected.
    # We can iterate over pairs (i,j) and check common neighbors.
    # For each pair (i,j), common neighbors C = N(i) & N(j).
    # Any pair (k,l) in C forms a K4 (i,j,k,l) if k,l are connected.
    # So #K4s = sum_{i<j} ( #edges in G[C_{ij}] )
    
    k4_count = 0
    # Precompute common neighbor bitmasks?
    # For n=35, n^2 = 1225 pairs.
    # For each pair, intersection of two rows (numpy dot or logical and).
    # Then count edges in the subgraph induced by the intersection.
    
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                # Common neighbors
                common = adj_matrix[i] & adj_matrix[j]
                # Count edges in subgraph induced by common
                # Submatrix A_common = A[common, common]
                # Edges = sum(A_common) / 2
                # We can do this by masking A
                # But creating submatrix is slow.
                # Alternative: 
                # Edge count in common = 0.5 * (sum_{k in common} sum_{l in common} A[k,l])
                # This is effectively summing A[k,l] where k,l are in common.
                # We can precompute nothing, just use the fact that common is a boolean vector.
                # A_sub = A[np.ix_(common_indices, common_indices)]
                
                # Faster: 
                # Let's just use the property: 
                # #K4s = 1/8 * sum_{i,j,k,l} A_{ij}A_{ik}A_{il}A_{jk}A_{jl}A_{kl}
                # This is getting complicated.
                
                # Let's go back to: #K4s = 1/24 * sum_{v} T(N(v))
                # Where T(H) is #triangles in H.
                # T(H) = 1/6 * trace(H^3).
                
                # For now, let's use a simple loop over triangles in G.
                # Find all triangles (i,j,k).
                # For each triangle, check if any vertex in N(i) \ {j,k} is connected to j and k.
                # i.e., is there a v in N(i) such that v~j and v~k?
                # This is checking if the triangle is part of a K4.
                
                pass

    # Optimized K4 Count for n <= 40:
    # Iterate over all triangles (i, j, k) with i < j < k.
    # For each triangle, check if there is a vertex v > k such that v is connected to i, j, k.
    # Or v < i? No, just check all v not in {i,j,k}.
    # To avoid double counting, we can just count all K4s directly.
    
    # Let's use the bitset approach properly.
    # Convert adjacency matrix to list of integers (bitmasks).
    masks = []
    for i in range(n):
        m = 0
        # Pack row i into m
        # Since n <= 40, we can use Python ints which are arbitrary precision.
        # Or use numpy uint64 if we handle chunks.
        # Python int is fine for n=40.
        for j in range(n):
            if adj_matrix[i, j]:
                m |= (1 << j)
        masks.append(m)
        
    k4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (masks[i] >> j) & 1:
                continue
            # Common neighbors of i and j
            common = masks[i] & masks[j]
            # We need to count edges between vertices in common.
            # Let C be the set of vertices in common.
            # Edges in C = sum_{u in C, v in C, u < v} A[u,v]
            # We can iterate through bits in common.
            # But common can have up to n-2 bits.
            # Iterating bits in python is slow if done naively.
            # However, n is small.
            
            # Alternative:
            # For each k in common (where k > j to avoid double counting triangles?), 
            # check if k is connected to other vertices in common.
            
            # Let's just iterate through the bits of common.
            # Extract indices
            # This is getting messy.
            
            # Let's use a different strategy for the GA penalty:
            # Penalty = #K4s + #Indep6s.
            # We can approximate or use a fast heuristic if exact is too slow.
            # But n=35 is small.
            
            # Let's try a direct K4 count using numpy matrix multiplication.
            # Triangles in G: T = (A @ A @ A).diagonal() / 6
            # K4s:
            # A K4 is a triangle plus a vertex connected to all 3.
            # Let's find all triangles first.
            # Triangles are triples (i,j,k) such that A[i,j]=A[i,k]=A[j,k]=1.
            # We can find them by checking A^2 diagonal? No.
            
            # Let's stick to the bitmask method but optimize the inner loop.
            # For each pair (i,j) that is an edge:
            #   common = masks[i] & masks[j]
            #   We want to count edges in the subgraph induced by common.
            #   Let's iterate through the set bits in common.
            #   To be fast, we can precompute nothing, just loop.
            
            # Actually, for the GA, we can just use a simpler penalty:
            # 1. If there is any K4, penalty += 1000
            # 2. If there is any Indep6, penalty += 1000
            # 3. Plus number of K4s and Indep6s.
            
            # Let's implement a fast "has K4" and "has Indep6" check first.
            # If valid, return 0.
            # If invalid, return a count.
            
            pass

    # This function is taking too long to design in my head. 
    # Let's use a known fast method for small n.
    # We will use the following:
    # 1. Check for K4:
    #    For each vertex v, look at its neighbors N(v).
    #    Check if N(v) contains a triangle.
    #    If yes, K4 exists.
    #    To count K4s, sum triangles in N(v) for all v, divide by 4.
    
    # 2. Check for Indep6:
    #    Independent set in G = Clique in complement G'.
    #    Check if G' contains a K6.
    #    For each vertex v in G', look at its neighbors N'(v).
    #    Check if N'(v) contains a K5.
    #    If yes, Indep6 exists.
    
    # This reduces the problem to finding cliques of size 3 and 5 in subgraphs.
    # For n=35, this is very fast.
    
    # Implementation:
    masks_g = []
    masks_comp = []
    for i in range(n):
        m_g = 0
        m_c = 0
        for j in range(n):
            if i == j: continue
            if adj_matrix[i, j]:
                m_g |= (1 << j)
            else:
                m_c |= (1 << j)
        masks_g.append(m_g)
        masks_comp.append(m_c)
        
    # Count K4s
    k4_count = 0
    for v in range(n):
        # Find triangles in G[N(v)]
        # Neighbors of v in G
        nb_mask = masks_g[v]
        # Get list of neighbors
        # We need to count triangles in the subgraph induced by nb_mask.
        # Iterate over pairs of neighbors (a, b) such that a < b and both in nb_mask.
        # If a and b are connected, then (v, a, b, c) is a K4 if there is a c in nb_mask connected to a and b.
        # Actually, a triangle in N(v) is a set {a,b,c} subset of N(v) such that a~b, b~c, c~a.
        # Each such triangle forms a K4 with v.
        
        # To count triangles in N(v):
        # Iterate a in N(v)
        # Iterate b in N(v) with b > a
        # If a~b, then common = N(a) & N(b) & N(v)
        # Count c in common with c > b.
        
        # This is O(n^3) per v? No, O(deg(v)^3) roughly.
        # Sum over v is O(n^4) worst case. n=35 -> 1.5M ops. Fine.
        
        # Optimization: Precompute nothing, just use bitmasks.
        # For each v:
        #   For each a in N(v):
        #     For each b in N(v) with b > a:
        #       if a~b:
        #         common = masks_g[a] & masks_g[b] & masks_g[v]
        #         # Count bits in common that are > b
        #         # We can just popcount(common & mask_greater_than_b)
        #         # But we need to avoid double counting triangles.
        #         # If we fix v, and count triangles in N(v), each triangle is counted once?
        #         # No, a triangle {a,b,c} in N(v) is formed by pairs (a,b), (a,c), (b,c).
        #         # If we iterate pairs (a,b), we find c.
        #         # So each triangle is found 3 times (once for each edge in the triangle).
        #         # So #triangles = (sum of counts) / 3.
        
        tri_count_v = 0
        # Get list of neighbors for iteration
        # To iterate bits in python is slow.
        # Let's convert nb_mask to list of indices.
        nbs = [i for i in range(n) if (nb_mask >> i) & 1]
        for idx_a, a in enumerate(nbs):
            for b in nbs[idx_a+1:]:
                if (masks_g[a] >> b) & 1:
                    common = masks_g[a] & masks_g[b] & nb_mask
                    # Count c in common such that c > b
                    # We can just count all and divide by 3 later?
                    # No, the pair (a,b) is specific.
                    # The triangle is {a,b,c}.
                    # We will count this triangle when we see pair (a,b) with c>b,
                    # pair (a,c) with b>c? No, b is fixed.
                    # Let's just count all c in common.
                    # Then each triangle is counted 3 times (for each of its 3 edges).
                    # So tri_count_v += popcount(common)
                    # Then #triangles = tri_count_v / 3.
                    
                    # Popcount in python: bin(x).count('1')
                    tri_count_v += bin(common).count('1')
                    
        k4_count += tri_count_v // 3
        
    # k4_count is the total number of K4s.
    
    # Count Indep6s (K6 in complement)
    indep6_count = 0
    for v in range(n):
        # Find K5 in G'[N'(v)]
        # N'(v) are non-neighbors of v in G.
        nb_mask_c = masks_comp[v]
        nbs_c = [i for i in range(n) if (nb_mask_c >> i) & 1]
        
        # Count K5 in subgraph induced by nbs_c.
        # A K5 is a set of 5 vertices all connected in G'.
        # We can find this by iterating over 4-tuples? No, too slow.
        # Iterate over pairs (a,b) in N'(v).
        # If a~b in G', then common = N'(a) & N'(b) & N'(v).
        # We need to find a K3 in common.
        # If we find a K3 {c,d,e} in common, then {v,a,b,c,d,e} is a K6.
        
        # To count K3s in a subgraph:
        # Similar to triangle counting.
        
        k5_count_v = 0
        for idx_a, a in enumerate(nbs_c):
            for b in nbs_c[idx_a+1:]:
                if (masks_comp[a] >> b) & 1:
                    common = masks_comp[a] & masks_comp[b] & nb_mask_c
                    # Count K3s in subgraph induced by common.
                    # Iterate pairs (c,d) in common.
                    # If c~d in G', then e = common & N'(c) & N'(d)
                    # Count e.
                    
                    # This is O(deg^3) per pair, so O(deg^5) per v.
                    # Too slow for n=35?
                    # deg ~ 17. 17^5 = 1.4M. 35 * 1.4M = 50M. Might be slow in Python.
                    
                    # Optimization:
                    # Just check if there is ANY K3.
                    # For GA penalty, we just need the count.
                    
                    # Let's just count K3s in common.
                    # Get list of bits in common
                    comms = [i for i in range(n) if (common >> i) & 1]
                    for idx_c, c in enumerate(comms):
                        for d in comms[idx_c+1:]:
                            if (masks_comp[c] >> d) & 1:
                                e_mask = masks_comp[c] & masks_comp[d] & common
                                # Count e
                                k5_count_v += bin(e_mask).count('1')
                                
        # Each K5 is counted 5 times? 
        # A K5 has 5 vertices. In the loop for v, we pick a pair (a,b) from the K5.
        # There are C(5,2)=10 pairs.
        # For each pair, we find the remaining 3 vertices forming a K3.
        # The K3 is counted 3 times (for its 3 edges).
        # So each K5 is counted 10 * 3 = 30 times?
        # Let's trace:
        # K5 = {v, a, b, c, d, e}. No, K5 is in N'(v), so vertices are {a,b,c,d,e}.
        # We fix v. We iterate pairs (a,b) in {a,b,c,d,e}.
        # If we pick (a,b), common contains {c,d,e}.
        # We count K3s in {c,d,e}. There is 1 K3.
        # It is counted 3 times in the inner loop (pairs c,d; c,e; d,e).
        # So for pair (a,b), we add 3.
        # There are 10 pairs in the K5.
        # Total added = 10 * 3 = 30.
        # So #K5s = k5_count_v / 30.
        
        indep6_count += k5_count_v // 30
        
    return k4_count + indep6_count

def _count_indep6(adj_matrix):
    # Handled in _count_k4 for combined penalty? 
    # Let's make a combined penalty function.
    pass

def _penalty(adj_matrix):
    return _count_k4(adj_matrix)

def construct():
    start_time = time.time()
    
    # Start with a known valid graph or random
    # Paley 17 is valid.
    # Let's generate Paley 17.
    def _paley17():
        n = 17
        squares = {x * x % n for x in range(1, n)}
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    best_n = 17
    best_adj = _paley17()
    
    # GA Parameters
    pop_size = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Try to increase n
    current_n = 17
    while current_n < 36:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        # Initialize population
        # Seed with best_adj (padded) and random graphs
        pop = []
        
        # Seed 1: best_adj extended with random edges
        seed_adj = np.zeros((current_n, current_n), dtype=np.uint8)
        # Copy best_adj to top-left
        if best_n <= current_n:
            seed_adj[:best_n, :best_n] = best_adj
        # Fill rest randomly
        for i in range(best_n, current_n):
            for j in range(i+1, current_n):
                if random.random() < 0.5:
                    seed_adj[i, j] = 1
                    seed_adj[j, i] = 1
        pop.append(seed_adj)
        
        # Random seeds
        for _ in range(pop_size - 1):
            adj = np.zeros((current_n, current_n), dtype=np.uint8)
            for i in range(current_n):
                for j in range(i+1, current_n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            pop.append(adj)
        
        # Evaluate fitness
        fitness = [_penalty(adj) for adj in pop]
        
        best_fitness = min(fitness)
        if best_fitness == 0:
            # Found valid graph
            best_adj = pop[fitness.index(0)]
            best_n = current_n
            current_n += 1
            continue
            
        # GA Loop
        generations = 0
        max_gens = 50
        while generations < max_gens:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            new_pop = []
            # Elitism
            best_idx = fitness.index(min(fitness))
            new_pop.append(pop[best_idx])
            
            for _ in range(pop_size - 1):
                # Tournament selection
                p1_idx = random.randint(0, pop_size - 1)
                p2_idx = random.randint(0, pop_size - 1)
                if fitness[p1_idx] < fitness[p2_idx]:
                    p1 = pop[p1_idx]
                else:
                    p1 = pop[p2_idx]
                    
                p3_idx = random.randint(0, pop_size - 1)
                p4_idx = random.randint(0, pop_size - 1)
                if fitness[p3_idx] < fitness[p4_idx]:
                    p2 = pop[p3_idx]
                else:
                    p2 = pop[p4_idx]
                    
                # Crossover
                if random.random() < crossover_rate:
                    child = np.zeros((current_n, current_n), dtype=np.uint8)
                    for i in range(current_n):
                        for j in range(i+1, current_n):
                            if random.random() < 0.5:
                                child[i, j] = p1[i, j]
                                child[j, i] = p1[j, i]
                            else:
                                child[i, j] = p2[i, j]
                                child[j, i] = p2[j, i]
                else:
                    child = p1.copy()
                    
                # Mutation
                if random.random() < mutation_rate:
                    # Flip random edges
                    for i in range(current_n):
                        for j in range(i+1, current_n):
                            if random.random() < 0.05: # Low prob per edge
                                child[i, j] = 1 - child[i, j]
                                child[j, i] = 1 - child[j, i]
                                
                new_pop.append(child)
                
            pop = new_pop
            fitness = [_penalty(adj) for adj in pop]
            best_fitness = min(fitness)
            
            if best_fitness == 0:
                best_adj = pop[fitness.index(0)]
                best_n = current_n
                current_n += 1
                break
                
            generations += 1
            
    # Return best found
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
