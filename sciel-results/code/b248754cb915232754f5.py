import time
import random
import sys

# EVOLVE-BLOCK-START

def _make_paley(n):
    """Paley graph on n vertices (n must be prime ≡ 1 mod 4)."""
    if n % 4 != 1:
        return None
    squares = {x * x % n for x in range(1, n)}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _edges_to_bitsets(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _count_k4(adj, n):
    count = 0
    for i in range(n):
        # Get neighbors of i
        neighbors_i = []
        mask = adj[i]
        # Iterate bits
        m = mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            neighbors_i.append(j)
            m ^= lsb
        
        # For each pair of neighbors j, k of i, check if j-k is edge
        # And if so, check if they share a common neighbor l > k? 
        # Simpler: Count triangles in G[N(i)]. Each K4 containing i corresponds to a triangle in N(i).
        # But this counts each K4 4 times (once per vertex).
        # So total K4s = Sum(count_triangles(N(i))) / 4.
        
        # Count triangles in subgraph induced by neighbors_i
        # Use bitsets for speed
        n_nb = len(neighbors_i)
        nb_bits = [0] * n_nb
        for idx, v in enumerate(neighbors_i):
            # Map neighbors of v that are also in neighbors_i
            mask_v = adj[v]
            # We only care about bits corresponding to neighbors_i
            # This is tricky with global bit indices.
            # Let's just use the global bitsets.
            pass
        
        # Alternative: Iterate pairs in neighbors_i
        for a_idx in range(n_nb):
            j = neighbors_i[a_idx]
            for b_idx in range(a_idx + 1, n_nb):
                k = neighbors_i[b_idx]
                if (adj[j] >> k) & 1:
                    # j, k, i form a triangle.
                    # This triangle is part of a K4 if there is a 4th vertex.
                    # Actually, a K4 is a set of 4 vertices. 
                    # The number of K4s is not just the number of triangles.
                    # A K4 has 4 triangles.
                    # So if we count all triangles in the graph, sum / 4 gives number of K4s?
                    # No. A K4 has 4 triangles. A graph with just K4 has 4 triangles.
                    # But triangles can be part of multiple K4s? No, a triangle {u,v,w} is in a K4 only if there is a vertex x connected to all three.
                    # So, count for each triangle {u,v,w} the number of common neighbors of u,v,w.
                    # Sum over all triangles. Each K4 is counted 4 times (once for each of its 4 triangles).
                    pass
    
    # Let's use a simpler, slightly slower but correct method for N <= 40
    k4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # Remove i and j from common? They are not in each other's adj usually, but bitsets might have them? No, simple graph.
                # We need to find edges in the subgraph induced by common.
                # Iterate pairs in common
                m = common
                # We only care about vertices > j to avoid double counting?
                # Let's just count all and divide by 4?
                # A K4 {a,b,c,d} is counted for pairs (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
                # For each edge in the K4, the other 2 vertices are in common.
                # So each K4 is counted 6 times?
                # Let's verify.
                # K4 vertices 0,1,2,3.
                # Edge (0,1): common neighbors {2,3}. Edge (2,3) exists. Count 1.
                # Edge (0,2): common neighbors {1,3}. Edge (1,3) exists. Count 1.
                # Edge (0,3): common neighbors {1,2}. Edge (1,2) exists. Count 1.
                # Edge (1,2): common neighbors {0,3}. Edge (0,3) exists. Count 1.
                # Edge (1,3): common neighbors {0,2}. Edge (0,2) exists. Count 1.
                # Edge (2,3): common neighbors {0,1}. Edge (0,1) exists. Count 1.
                # Total 6.
                # So K4 count = Sum(edges in common subgraph) / 6.
                
                # Count edges in subgraph induced by common
                m = common
                while m:
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    m ^= lsb
                    if k <= j: continue # Optimization: only look at k > j? 
                    # No, we are counting edges in common.
                    # If we count all edges in common for every edge (i,j), we count each K4 6 times.
                    # So we can just sum them up and divide by 6.
                    
                    # Check if k is connected to any other node in common?
                    # We need to count the number of edges in the subgraph induced by `common`.
                    # Let's do this:
                    # For each k in common (where k > j to avoid double counting the "base" edge logic? No, the /6 logic handles it).
                    # Actually, if we iterate i < j, and count edges in common, we might double count?
                    # No, the /6 normalization assumes we sum over ALL edges in the graph.
                    # If we only iterate i < j, we sum over half the edges.
                    # Then each K4 is counted 3 times (3 edges with i<j? No, 3 of the 6 edges have i<j).
                    # So if we iterate i < j, sum is 3 * K4_count.
                    
                    # Let's just iterate i < j and count edges in common with k > j?
                    # If we restrict k > j, do we miss any?
                    # K4 {0,1,2,3}.
                    # Edge (0,1): common {2,3}. k>1 -> 2,3. Edge (2,3) exists. Count 1.
                    # Edge (0,2): common {1,3}. k>2 -> 3. Edge (1,3)? 1 is not > 2. So we don't check (1,3).
                    # This restriction breaks the symmetry.
                    
                    # Safest: Iterate all i < j. Count ALL edges in the subgraph induced by common.
                    # Sum them. Divide by 6.
                    
                    # How to count edges in subgraph induced by common quickly?
                    # Iterate k in common. For each k, count neighbors in common that are > k?
                    # This counts each edge in the subgraph once.
                    m2 = common
                    edges_in_common = 0
                    while m2:
                        lsb2 = m2 & -m2
                        k = lsb2.bit_length() - 1
                        m2 ^= lsb2
                        # Count neighbors of k in common that are > k
                        common_gt_k = common & ~((1 << (k + 1)) - 1)
                        edges_in_common += (adj[k] & common_gt_k).bit_count()
                    
                    k4_count += edges_in_common
    
    return k4_count // 6

def _count_ind6(adj, n):
    """Count independent sets of size 6."""
    # Equivalent to counting K6 in complement.
    # Complement adjacency: comp_adj[i] = (~adj[i]) & ((1<<n)-1) ^ (1<<i)
    
    # We can use the same logic as K4, but on the complement.
    # However, building complement bitsets is fast.
    
    # To save time, we can use the property:
    # Ind6 in G = K6 in Comp(G).
    # Count K6 in Comp(G).
    # A K6 has 15 edges.
    # If we iterate edges (i,j) in Comp(G) (i<j), and count edges in common neighbors in Comp(G), 
    # each K6 is counted 15 times?
    # Let's verify.
    # K6 vertices 0..5.
    # Edge (0,1): common neighbors {2,3,4,5}. Edges in this set: (2,3),(2,4),(2,5),(3,4),(3,5),(4,5) -> 6 edges.
    # Each edge in the K6 contributes the number of edges in the common neighbors.
    # For any edge in a K6, the common neighbors are the other 4 vertices, which form a K4.
    # A K4 has 6 edges.
    # So each edge in the K6 contributes 6.
    # There are 15 edges in K6.
    # Total sum = 15 * 6 = 90.
    # So K6 count = Sum / 90.
    
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = (full_mask ^ (1 << i)) ^ adj[i]
        
    sum_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                # Count edges in subgraph induced by common in Comp(G)
                m2 = common
                edges_in_common = 0
                while m2:
                    lsb2 = m2 & -m2
                    k = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    common_gt_k = common & ~((1 << (k + 1)) - 1)
                    edges_in_common += (comp_adj[k] & common_gt_k).bit_count()
                
                sum_edges += edges_in_common
    
    return sum_edges // 90

def construct():
    start_time = time.time()
    deadline = start_time + 150 - 10 # 10s safety margin
    
    # Start with Paley 17
    best_result = _make_paley(17)
    best_n = 17
    best_adj = _edges_to_bitsets(17, best_result['edges'])
    
    # Verify start
    if _count_k4(best_adj, 17) != 0 or _count_ind6(best_adj, 17) != 0:
        # Should not happen for Paley 17
        pass
    
    # Try to grow
    for target_n in range(18, 37):
        if time.time() > deadline:
            break
            
        # Initialize new graph
        new_adj = best_adj[:] + [0]
        new_n = target_n
        
        # Randomize edges for new vertex
        for i in range(best_n):
            if random.random() < 0.5:
                new_adj[i] |= (1 << best_n)
                new_adj[best_n] |= (1 << i)
                
        # Simulated Annealing
        current_adj = new_adj[:]
        current_k4 = _count_k4(current_adj, new_n)
        current_ind6 = _count_ind6(current_adj, new_n)
        current_energy = current_k4 + current_ind6
        
        best_adj_sa = current_adj[:]
        best_energy_sa = current_energy
        
        T = 1.0
        cooling = 0.999
        steps = 0
        step_limit = 50000 # Adjust based on time
        
        sa_start = time.time()
        while time.time() - sa_start < 20: # 20s per growth step
            if current_energy == 0:
                break
                
            # Pick random edge
            u = random.randint(0, new_n - 1)
            v = random.randint(0, new_n - 1)
            if u == v:
                continue
                
            # Calculate Delta
            # This is expensive to do exactly every step.
            # Approximation: Just recompute?
            # For N=35, _count_k4 and _count_ind6 take ~0.1-0.5s?
            # If so, we can only do ~100-200 steps per second.
            # 20s -> 2000-4000 steps. Might be enough.
            
            # Let's try to optimize delta calculation.
            # Or just recompute. It's safer.
            
            # Flip
            if (current_adj[u] >> v) & 1:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
            new_k4 = _count_k4(current_adj, new_n)
            new_ind6 = _count_ind6(current_adj, new_n)
            new_energy = new_k4 + new_ind6
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_energy = new_energy
                if current_energy < best_energy_sa:
                    best_energy_sa = current_energy
                    best_adj_sa = current_adj[:]
            else:
                # Revert
                if (current_adj[u] >> v) & 1:
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
            T *= cooling
            steps += 1
            
        # After SA, check best
        if best_energy_sa == 0:
            # Verify
            if _count_k4(best_adj_sa, new_n) == 0 and _count_ind6(best_adj_sa, new_n) == 0:
                best_adj = best_adj_sa
                best_n = new_n
                # Update best_result edges
                edges = []
                for i in range(best_n):
                    for j in range(i + 1, best_n):
                        if (best_adj[i] >> j) & 1:
                            edges.append([i, j])
                best_result = {'n': best_n, 'edges': edges}
            else:
                break # Should not happen if energy 0 means valid
        else:
            # Failed to grow to this n
            # Try again with different random init?
            # Or just stop and return current best
            # Let's try one more time with a fresh random init
            if time.time() > deadline:
                break
            
            # Retry
            success = False
            for _ in range(3):
                if time.time() > deadline:
                    break
                new_adj = best_adj[:] + [0]
                for i in range(best_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << best_n)
                        new_adj[best_n] |= (1 << i)
                        
                current_adj = new_adj[:]
                current_k4 = _count_k4(current_adj, new_n)
                current_ind6 = _count_ind6(current_adj, new_n)
                current_energy = current_k4 + current_ind6
                best_energy_sa = current_energy
                best_adj_sa = current_adj[:]
                
                T = 1.0
                sa_start = time.time()
                while time.time() - sa_start < 10:
                    if current_energy == 0:
                        break
                    u = random.randint(0, new_n - 1)
                    v = random.randint(0, new_n - 1)
                    if u == v: continue
                    
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        
                    new_k4 = _count_k4(current_adj, new_n)
                    new_ind6 = _count_ind6(current_adj, new_n)
                    new_energy = new_k4 + new_ind6
                    delta = new_energy - current_energy
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy_sa:
                            best_energy_sa = current_energy
                            best_adj_sa = current_adj[:]
                    else:
                        if (current_adj[u] >> v) & 1:
                            current_adj[u] ^= (1 << v)
                            current_adj[v] ^= (1 << u)
                        else:
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                            
                    T *= cooling
                    
                if best_energy_sa == 0:
                    if _count_k4(best_adj_sa, new_n) == 0 and _count_ind6(best_adj_sa, new_n) == 0:
                        best_adj = best_adj_sa
                        best_n = new_n
                        edges = []
                        for i in range(best_n):
                            for j in range(i + 1, best_n):
                                if (best_adj[i] >> j) & 1:
                                    edges.append([i, j])
                        best_result = {'n': best_n, 'edges': edges}
                        success = True
                        break
                        
            if not success:
                break # Cannot grow further, return current best

    return best_result
# EVOLVE-BLOCK-END
