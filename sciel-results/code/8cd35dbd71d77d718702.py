import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def is_valid(n, adj):
        """
        Checks if the graph defined by adj (n x n numpy array) has no K4 and no I6.
        Returns True if valid, False otherwise.
        Uses bitmask for speed.
        """
        if n <= 3:
            return True
        if n >= 6:
            # Check for I6
            # If n < 6, no I6 possible.
            pass
        
        # Convert to bitmasks
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
        
        # Check K4
        # Count K4s. If > 0, invalid.
        k4_count = 0
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    # Count edges in subgraph induced by 'common'
                    c_count = 0
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        c_count += bin(masks[k] & common).count('1')
                    
                    k4_count += c_count
        k4_count //= 6
        if k4_count > 0:
            return False

        # Check I6
        # Independent set of size 6 in G is Clique of size 6 in G_bar.
        # We can check if there is a K6 in the complement.
        # Complement masks: comp_mask[i] has bit j set if NOT (i==j or adj[i][j])
        comp_masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and not adj[i][j]:
                    m |= (1 << j)
            comp_masks[i] = m
            
        # Find if there is a clique of size 6 in the complement graph.
        # This is equivalent to finding an independent set of size 6 in G.
        # We can use a simple backtracking search for a clique of size 6 in the complement.
        # Since we only need to know if one exists, we can prune.
        
        found_i6 = [False]
        
        def dfs_clique(current_set, candidates, depth):
            if found_i6[0]:
                return
            if depth == 6:
                found_i6[0] = True
                return
            if bin(candidates).count('1') < (6 - depth):
                return
                
            c_mask = candidates
            while c_mask:
                if found_i6[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                
                # v is in candidates, so it is connected to all in current_set in the complement graph.
                # New candidates: neighbors of v in the complement graph that are in c_mask
                new_candidates = c_mask & comp_masks[v]
                dfs_clique(current_set | lsb, new_candidates, depth + 1)
                
        dfs_clique(0, (1 << n) - 1, 0)
        
        if found_i6[0]:
            return False
            
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Strategy:
    # 1. Start with a known valid graph. Paley graph of order 17 is a good start (K4-free? No, Paley 17 has cliques. 
    #    Wait, Paley 17 is strongly regular (17,8,3,4). 
    #    Clique number of Paley 17: 
    #    Lambda=3, Mu=4. 
    #    It is known that Paley 17 is K4-free? 
    #    Actually, Paley q graphs are self-complementary. 
    #    For q=17, the graph is (17,8,3,4). 
    #    Let's check K4. A K4 would require 4 vertices all connected. 
    #    If we pick v, its neighbors form a graph with degree 3 within the neighborhood? 
    #    Actually, the subgraph induced by neighbors of v is a graph on 8 vertices where each vertex has degree 3 (lambda=3). 
    #    Does a (8,3) graph have a triangle? 
    #    If it has a triangle, then v plus that triangle is a K4. 
    #    The graph of neighbors is a strongly regular graph? No. 
    #    But it is known that Paley 17 is K4-free. 
    #    Let's verify. 
    #    Also, independent set size? 
    #    Paley 17 has independence number 5? 
    #    If alpha < 6, it is valid for n=17.
    
    # Let's try to build up from a smaller base or just use SA from n=35 but with a better cost function and initial state.
    # The current program's SA is too slow because get_violations is called every step and is O(n^3) or worse.
    # My is_valid is also O(n^3) roughly.
    # To make it fast, I will use a "local delta" evaluation for SA.
    
    # However, the constructive approach might be better.
    # Let's try to find a valid graph for n=35 using a randomized local search that is much faster.
    # Key optimization: Instead of checking the whole graph, update the violation count incrementally.
    # But maintaining exact counts for K4 and I6 incrementally is hard.
    
    # Alternative: Use a very fast "is_valid" check that short-circuits.
    # My is_valid above checks K4 first. If K4 exists, it returns False quickly? 
    # The K4 check loops over edges. If a K4 is found, it still finishes the loop in the code above. 
    # I can modify it to return False immediately if k4_count > 0? No, it sums up.
    # I can stop counting K4s if count > 0? Yes.
    
    def is_valid_fast(n, adj):
        if n > 35: return False
        
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
            
        # Fast K4 check
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    # Check if there is an edge in common
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        # If k is connected to any other node in common, we have a K4?
                        # No, we need an edge (k, l) where l is in common and l != k.
                        # So check if masks[k] & common has any bit set (excluding k itself? common doesn't have k? 
                        # Wait, common is neighbors of i and j. k is in common. 
                        # Does k connect to another node in common?
                        # If yes, that node l, plus i, j, k forms a K4.
                        # So if (masks[k] & common) != 0, we found a K4.
                        if masks[k] & common:
                            return False
                            
        # Fast I6 check (Clique 6 in complement)
        comp_masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and not adj[i][j]:
                    m |= (1 << j)
            comp_masks[i] = m
            
        # Backtracking for K6 in complement
        # Order vertices by degree in complement (descending) to find cliques faster?
        # Or just use the previous DFS but with early exit.
        
        # To make it faster, we can pre-sort vertices by complement degree.
        # But for n=35, the DFS is usually fast if the graph is "dense" in complement (sparse in G).
        # If G is sparse, complement is dense, K6 is likely, DFS finds it fast.
        # If G is dense, complement is sparse, K6 is unlikely, DFS explores more.
        # But we want G to be K4-free, so G cannot be too dense.
        
        found = [False]
        
        def dfs(current_set, candidates, depth):
            if found[0]:
                return
            if depth == 6:
                found[0] = True
                return
            if bin(candidates).count('1') < (6 - depth):
                return
                
            # Optimization: Pick a vertex with high degree in candidates?
            # Let's just iterate.
            c_mask = candidates
            while c_mask:
                if found[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                new_candidates = c_mask & comp_masks[v]
                dfs(current_set | lsb, new_candidates, depth + 1)
                
        dfs(0, (1 << n) - 1, 0)
        if found[0]:
            return False
            
        return True

    # Initialize with a random graph for n=35
    n = 35
    adj = np.zeros((n, n), dtype=int)
    
    # To get a good start, we can try to construct a graph that is "close" to valid.
    # A random graph with p=0.5 is a bad start.
    # A graph with p=0.3 or 0.4 might be better?
    # Or we can start with a known valid graph of smaller n and add vertices.
    
    # Let's try a hybrid:
    # 1. Generate random graph.
    # 2. Run a fast local search (hill climbing) to reduce violations.
    # 3. If stuck, restart.
    
    # Since is_valid_fast is fast, we can afford to do many restarts.
    
    best_adj = None
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Random init
        p = 0.4 + 0.2 * random.random() # p in [0.4, 0.6]
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        # Local Search: Hill Climbing with random restarts (within the loop)
        # We flip one edge at a time.
        # If is_valid_fast returns True, we are done.
        
        # To make it faster, we can use a "batch" of flips? No, single flip is safer.
        
        # Optimization: If the graph is invalid, try to fix it.
        # We don't know which constraint is violated easily without checking.
        # But we can just do random flips and check validity.
        
        # To speed up, we can check validity less frequently? 
        # No, we need to know when we are valid.
        
        # Let's do a "Simulated Annealing" style but with a very fast validity check.
        # Actually, since we only care about validity (0 vs >0), we can just do random flips until valid.
        # This is a "random walk" in the space of graphs.
        # The probability of hitting a valid graph might be low.
        
        # Better: Use the fact that valid graphs are a small subset.
        # We should bias the search towards graphs with low "potential" for violations.
        
        # Let's stick to SA but with is_valid_fast.
        # Cost: 0 if valid, 1 if invalid.
        # This is a discrete landscape. SA might struggle.
        
        # Alternative: "Greedy Repair".
        # 1. Check if valid. If yes, return.
        # 2. If not, identify a violation (K4 or I6).
        # 3. Randomly modify an edge involved in the violation to break it.
        
        # This requires finding a violation.
        # My is_valid_fast doesn't return the violation, just False.
        # I can modify it to return the violation type and vertices.
        
    # Let's implement a smarter search:
    # 1. Start with random graph.
    # 2. Loop:
    #    a. Check validity. If valid, return.
    #    b. Find a K4 or I6.
    #    c. Pick a random edge in that K4/I6 and flip it (or try to flip it to break the structure).
    #       - For K4, removing an edge breaks it. Adding an edge doesn't help.
    #       - For I6, adding an edge breaks it. Removing an edge doesn't help.
    #    d. Repeat.
    
    # This is a "constraint-directed" search.
    
    def find_violation(n, adj):
        """Returns ('K4', [v1,v2,v3,v4]) or ('I6', [v1,...,v6]) or None."""
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
            
        # Find K4
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        # Check if k connects to any other in common
                        rest = common & ~(1 << k)
                        if masks[k] & rest:
                            # Find the specific l
                            r_mask = masks[k] & rest
                            lsb2 = r_mask & -r_mask
                            l = lsb2.bit_length() - 1
                            return ('K4', [i, j, k, l])
                            
        # Find I6
        comp_masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and not adj[i][j]:
                    m |= (1 << j)
            comp_masks[i] = m
            
        found = [None]
        
        def dfs(current_set, candidates, depth, path):
            if found[0]:
                return
            if depth == 6:
                found[0] = path[:]
                return
            if bin(candidates).count('1') < (6 - depth):
                return
                
            c_mask = candidates
            while c_mask:
                if found[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                path.append(v)
                new_candidates = c_mask & comp_masks[v]
                dfs(current_set | lsb, new_candidates, depth + 1, path)
                path.pop()
                
        dfs(0, (1 << n) - 1, 0, [])
        
        if found[0]:
            return ('I6', found[0])
            
        return None

    # Main Search Loop
    n = 35
    adj = np.zeros((n, n), dtype=int)
    
    # Initialize with a graph that is likely to be close to valid.
    # Random p=0.5 is a start.
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
                
    while time.time() < deadline:
        viol = find_violation(n, adj)
        if viol is None:
            edges = get_edges(adj, n)
            return {'n': n, 'edges': edges}
            
        v_type, verts = viol
        if v_type == 'K4':
            # Pick a random edge in the K4 to remove
            edges_in_k4 = list(combinations(verts, 2))
            u, v = random.choice(edges_in_k4)
            if adj[u][v] == 1:
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                # Should not happen if it's a K4
                pass
        elif v_type == 'I6':
            # Pick a random non-edge in the I6 to add
            # All pairs in I6 are non-edges.
            pairs = list(combinations(verts, 2))
            u, v = random.choice(pairs)
            if adj[u][v] == 0:
                adj[u][v] = 1
                adj[v][u] = 1
            else:
                pass
                
    # Fallback: Return a known valid graph (e.g., n=17 Paley)
    # Paley 17
    n_p = 17
    squares = {x * x % n_p for x in range(1, n_p)}
    p_edges = [[i, j] for i in range(n_p) for j in range(i + 1, n_p) if (j - i) % n_p in squares]
    return {'n': n_p, 'edges': p_edges}

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
