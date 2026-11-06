import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    # We know n=17 is possible. We try to find a valid coloring for n=17 quickly.
    # If we find one, we return it. If we have time, we might try n=18 (unlikely to succeed in 150s with simple backtracking, but let's see).
    # Actually, n=18 is very hard. The known bound is 17. 
    # Strategy: 
    # 1. Try to find a valid coloring for n=17 using backtracking.
    # 2. If found, return it.
    # 3. If not found within a small time budget, fallback to the known circulant coloring for n=17 (which is valid).
    # 4. If we have significant time left, we could try n=18, but it's likely to fail. 
    #    However, the task says "Maximize n". The seed returns 17. 
    #    If we can't beat 17, we must ensure we return a valid 17 quickly.
    
    # The seed program's circulant coloring for n=17 is:
    # d = min(j-i, n-(j-i)); color = (d-1)%4
    # Let's verify if this is always valid. The seed assumes it is.
    # We will first try to generate a valid coloring via search to potentially find a "better" structure or just confirm validity faster,
    # but actually, the search might be slower than just checking the known one.
    
    # Better strategy:
    # 1. Check the known circulant coloring for n=17. If valid, return it immediately.
    # 2. If invalid (unlikely, but possible if my memory of the seed's validity is wrong or the seed was flawed), use backtracking to find one.
    # 3. If we have time, try to search for n=18.
    
    # Let's start with the known good candidate: n=17 circulant.
    n_target = 17
    colors = _generate_circulant(n_target)
    
    # Verify quickly
    if _is_valid(colors, n_target):
        return {'n': n_target, 'colors': colors}
    
    # If the circulant one failed (shouldn't happen), try backtracking for n=17
    # We need a fast verifier and a smart search.
    
    # Let's try to search for n=17 from scratch with backtracking if the above failed.
    # But wait, if the above failed, it means my circulant formula is wrong for C6.
    # Let's implement a robust backtracker for n=17 just in case, or to find a solution if the static one is bad.
    
    # Actually, let's try to search for n=18 if we have time.
    # But n=18 is R_4(C6) > 18? No, R_4(C6) is the Ramsey number. 
    # If a coloring exists for K_18, then R_4(C6) >= 19.
    # The task says "best known colouring has 17 vertices". So R_4(C6) > 17 is known.
    # Finding 18 would be new.
    
    # Given the difficulty of n=18, we should focus on ensuring we return a valid n=17.
    # The circulant coloring is the standard construction.
    
    # Let's add a safety net: if the circulant check takes too long or fails, 
    # we run a backtracking solver for n=17.
    
    # Actually, the seed program's _has_c6 is extremely slow (permutations of 6 vertices).
    # My _is_valid uses bitmasks and is much faster.
    
    # Let's try to improve the score by attempting n=18 search?
    # Probability of finding n=18 in 120s with simple backtracking is near zero.
    # Probability of failing to return n=17 due to timeout is the main risk.
    
    # So the best strategy is:
    # 1. Generate n=17 circulant.
    # 2. Verify it with fast bitmask C6 check.
    # 3. If valid, return.
    # 4. If invalid (bug in my formula?), use backtracking to find a valid n=17.
    
    # Let's implement the backtracker for n=17 as a fallback.
    
    # But wait, what if the circulant coloring IS valid? Then we return in <0.1s.
    # This is the safest bet.
    
    # Let's double check the circulant coloring logic.
    # For n=17, d=1..8.
    # d=1 -> 0
    # d=2 -> 1
    # d=3 -> 2
    # d=4 -> 3
    # d=5 -> 0
    # d=6 -> 1
    # d=7 -> 2
    # d=8 -> 3
    # This is a known construction for 4-coloring K_17 without C6?
    # Actually, the known result is R_4(C6) >= 17. The construction is often based on finite fields or similar.
    # The circulant graph C17 with steps {1,2,3,4} colored 0,1,2,3?
    # Let's trust the seed's formula but verify it fast.
    
    # If verification fails, we do a random restart backtracking search for n=17.
    
    if time.time() < deadline - 10:
        # Try backtracking for n=17 if circulant failed
        solution = _backtrack_n(17, deadline)
        if solution:
            return {'n': 17, 'colors': solution}
            
    # If all else fails, return the circulant one anyway (hoping it's valid) or a smaller n
    # But we must return a VALID coloring.
    # If circulant was invalid, and backtracking failed, we are in trouble.
    # Let's try n=16 circulant as fallback.
    
    n_target = 16
    colors = _generate_circulant(n_target)
    if _is_valid(colors, n_target):
        return {'n': n_target, 'colors': colors}
        
    # Fallback to n=9 from seed
    n_target = 9
    colors = [min(j - i, n_target - j + i) - 1 for i in range(n_target) for j in range(i + 1, n_target)]
    return {'n': n_target, 'colors': colors}

def _generate_circulant(n):
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            d = min(j - i, n - (j - i))
            colors.append((d - 1) % 4)
    return colors

def _is_valid(colors, n):
    # Build adjacency bitmasks for each color
    adj = [0] * 4
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c] |= (1 << i) | (1 << j) # Wait, bitmask needs to represent edges? No, vertices.
            # Actually, for C6 detection, we need adjacency matrix or list.
            # Bitmask of vertices is not enough to check edges directly without iterating.
            # Let's use a list of sets or bitmasks per vertex.
            
    # Better: adj[c][u] is a bitmask of neighbors of u in color c.
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    # Check for C6 in each color
    for c in range(4):
        if _has_c6_fast(adj[c], n):
            return False
    return True

def _has_c6_fast(adj, n):
    # adj[u] is bitmask of neighbors of u
    # Check all combinations of 6 vertices? No, too slow.
    # Check all paths of length 5 that close?
    # Or use BFS/DFS from each pair?
    
    # Efficient check:
    # For each pair (u, v) not connected, check if there is a path of length 4 between them?
    # No, C6 is a cycle.
    
    # Let's use the property: A graph contains C6 iff there exists a path of length 5 between two non-adjacent vertices that are distance 1? No.
    
    # Simplest fast check: 
    # For each vertex u, for each neighbor v, for each neighbor w of v (w!=u), ...
    # This is O(N * D^5) which might be okay for N=17.
    
    # Let's try:
    # Iterate over all 6-tuples? No.
    
    # Alternative: 
    # For each pair of vertices (u, v), check if they are in a C6.
    # This is hard.
    
    # Let's stick to a slightly optimized permutation check or BFS.
    # For n=17, combinations(17, 6) = 12376.
    # For each set, check if it induces a C6? No, "not necessarily induced".
    # So we just need to check if the subgraph contains a C6.
    
    # For a set of 6 vertices, we can check if there is a Hamiltonian cycle in the induced subgraph.
    # 6 vertices -> 5! = 120 permutations.
    # 12376 * 120 * 6 checks ~ 9M operations. This is fast in Python if optimized? Maybe 1-2 seconds.
    
    # Let's implement this specific check.
    
    import itertools
    
    for verts in itertools.combinations(range(n), 6):
        # Check if induced subgraph on verts has a Hamiltonian cycle
        # We can use DP or just permutations since 6 is small.
        # To speed up, we can check if the graph is connected and has enough edges.
        # But let's just do permutations with early exit.
        
        v0 = verts[0]
        rest = list(verts[1:])
        
        # Optimization: fix v0. Try permutations of rest.
        # To avoid double counting, we can require v1 < v5? No, that's for undirected cycles.
        # A cycle v0-v1-v2-v3-v4-v5-v0 is the same as v0-v5-v4-v3-v2-v1-v0.
        # So we can require v1 < v5 to break symmetry.
        
        for perm in itertools.permutations(rest):
            v1, v2, v3, v4, v5 = perm
            if v1 > v5:
                continue
            if (adj[v0] & (1 << v1)) and \
               (adj[v1] & (1 << v2)) and \
               (adj[v2] & (1 << v3)) and \
               (adj[v3] & (1 << v4)) and \
               (adj[v4] & (1 << v5)) and \
               (adj[v5] & (1 << v0)):
                return True
    return False

def _backtrack_n(n, deadline):
    # Backtracking search for valid 4-coloring of K_n
    # Variables: edges (i, j) in lexicographic order
    # Values: 0, 1, 2, 3
    
    # This is a very large search space. 4^(n(n-1)/2).
    # For n=17, edges = 136. 4^136 is huge.
    # We need strong pruning.
    
    # Pruning: If adding an edge creates a C6 in any color, prune.
    
    # We will use a simple recursive backtracker.
    # To make it faster, we can use symmetry breaking:
    # Fix the first few edges?
    
    # Given the time limit, this will likely fail for n=17 unless the solution is very close to the start.
    # But it's a fallback.
    
    # We'll use a global time check.
    
    def check_c6_on_add(edge, color, current_adj):
        # current_adj[c] is list of bitmasks
        u, v = edge
        # Tentatively add edge
        current_adj[color][u] |= (1 << v)
        current_adj[color][v] |= (1 << u)
        
        # Check if C6 exists now
        # We only need to check cycles involving the new edge (u, v).
        # A C6 containing (u, v) has the form u-v-a-b-c-d-u or u-v-d-c-b-a-u.
        # So we need a path of length 4 from v to u in the graph WITHOUT the edge (u, v) but WITH the rest?
        # No, the path must use existing edges.
        # So, is there a path of length 4 from v to u in the graph G' where (u, v) is removed?
        # Actually, the path is v-a-b-c-d-u.
        # So we need to check if there is a path of length 4 from v to u in the current graph (before adding edge) 
        # such that the internal vertices are distinct and not u or v.
        
        # Let's find all paths of length 4 from v to u.
        # Start at v.
        # Step 1: neighbors of v (excluding u)
        # Step 2: neighbors of those
        # Step 3: neighbors of those
        # Step 4: neighbors of those -> must be u
        
        # We can do a BFS/DFS limited to depth 4.
        
        # Remove u, v from consideration for internal nodes.
        
        # neighbors of v
        neigh_v = current_adj[color][v] & ~(1 << u)
        
        # We need to find a-b-c-d such that:
        # v-a, a-b, b-c, c-d, d-u are edges.
        
        # Let's iterate over possible 'a' (neighbors of v)
        found = False
        # To speed up, we can use bit operations.
        
        # Path v-a-b-c-d-u
        # a in N(v)
        # b in N(a)
        # c in N(b)
        # d in N(c) and d in N(u)
        
        # Let's precompute N(u)
        neigh_u = current_adj[color][u] & ~(1 << v)
        
        # Iterate a
        a_mask = neigh_v
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask -= a_bit
            
            # b in N(a) \ {v, u}
            b_mask = current_adj[color][a] & ~(1 << v) & ~(1 << u)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask -= b_bit
                
                # c in N(b) \ {a, v, u}
                c_mask = current_adj[color][b] & ~(1 << a) & ~(1 << v) & ~(1 << u)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask -= c_bit
                    
                    # d in N(c) \ {b, v, u} AND d in N(u)
                    d_mask = current_adj[color][c] & ~(1 << b) & ~(1 << v) & ~(1 << u)
                    d_mask &= neigh_u
                    
                    if d_mask:
                        # Found a path
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
        
        # Remove edge
        current_adj[color][u] &= ~(1 << v)
        current_adj[color][v] &= ~(1 << u)
        
        return found

    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    adj = [[0] * n for _ in range(4)]
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if idx == len(edges):
            return [0] * len(edges) # Placeholder, need to reconstruct
            
        u, v = edges[idx]
        
        for c in range(4):
            # Check if adding (u, v) with color c creates a C6
            if check_c6_on_add((u, v), c, adj):
                continue
            
            # Assign
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            # Recurse
            # We need to store the assignment. 
            # Since we return None on timeout, we can't easily reconstruct.
            # Let's store assignments in a list.
            
            # This implementation is getting complex for a fallback.
            # Given the low probability of success, let's simplify:
            # Just return None to indicate failure.
            
            # For now, I will disable the complex backtrack and just rely on the circulant.
            # If circulant fails, we return a smaller n.
            
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            
        return None

    # The backtrack function above is incomplete and likely too slow.
    # I will not use it.
    return None

# EVOLVE-BLOCK-END
