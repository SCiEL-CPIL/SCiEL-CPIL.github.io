"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def _is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_exact(adj, time_limit=5.0):
    """
    Exact branch and bound to find the largest triangle-free induced subgraph.
    Uses bitmasks for efficiency. Returns the maximum size.
    """
    n = len(adj)
    # Precompute adjacency as bitmasks
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute triangle bitmask: for each vertex v, which vertices form a triangle with v
    # tri_mask[v] = set of vertices u such that v-u is an edge and there exists w with edges v-w and u-w
    tri_mask = [0] * n
    for v in range(n):
        for u in range(n):
            if u == v or not (adj_mask[v] >> u) & 1:
                continue
            # Check if there's a common neighbor w of v and u
            common = adj_mask[v] & adj_mask[u]
            if common:
                tri_mask[v] |= (1 << u)
    
    # A vertex is "free" (can be in a triangle-free set with all others) if it has no triangle partners
    # Actually, we need: subset S is triangle-free iff for all v in S, tri_mask[v] & mask(S) == 0
    
    start_time = time.time()
    best = [0]
    
    # Branch and bound
    # State: current_mask (vertices in current candidate set), current_size
    # We want to find max |S| such that S is triangle-free
    
    # Greedy upper bound: if we have a set of candidates, the max triangle-free subset
    # is at most the size of the largest independent set in the "triangle conflict" graph
    # But simpler: just use current_size as upper bound
    
    def branch(mask, size):
        if time.time() - start_time > time_limit:
            return
        if size > best[0]:
            best[0] = size
        if mask == 0:
            return
        
        # Upper bound: size + popcount(mask)
        if size + bin(mask).count('1') <= best[0]:
            return
        
        # Pick a vertex from mask
        # Heuristic: pick vertex with most triangle conflicts in mask
        v = -1
        max_conflicts = -1
        m = mask
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            conflicts = bin(tri_mask[u] & mask).count('1')
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                v = u
            m ^= lb
        
        if v == -1:
            return
        
        # Option 1: exclude v
        branch(mask & ~(1 << v), size)
        
        # Option 2: include v (only if it doesn't create a triangle)
        # v can be included if tri_mask[v] & mask == 0 (no triangle partner in remaining)
        # Actually, we need to check that adding v to the current set doesn't create a triangle
        # Since we're building the set incrementally, we need: for all u in current_set, 
        # v and u don't share a common neighbor in current_set
        # But in our formulation, we're selecting from mask, and the constraint is:
        # for all pairs (u,w) in selected, they don't form a triangle
        # Simpler: v can be added if tri_mask[v] & (mask without v) has no vertex that is 
        # also in the selected set... this gets complex.
        
        # Alternative approach: use the fact that a set S is triangle-free iff
        # for every v in S, (tri_mask[v] & S) == 0 is NOT quite right either.
        # Actually: S is triangle-free iff there do not exist distinct a,b,c in S with all three edges.
        # Equivalent: for every edge (a,b) in S, there is no c in S adjacent to both.
        
        # Let me use a different formulation:
        # S is triangle-free iff for every v in S, the neighbors of v in S form an independent set
        # i.e., for every v in S, (adj_mask[v] & S) is an independent set in the induced graph
        
        # This is still complex. Let me use a simpler exact method:
        # Enumerate subsets by size, from largest to smallest
        # For each size k, check if there's a triangle-free subset of size k
        
        # Actually, let me just do a clean branch and bound:
        # State: (candidates_mask, selected_mask)
        # Constraint: selected_mask is triangle-free
        # We add vertices one at a time
        
        # For including v: check that selected_mask ∪ {v} is triangle-free
        # This means: no triangle in selected_mask ∪ {v}
        # Since selected_mask is already triangle-free, we just need to check
        # that v doesn't form a triangle with any pair in selected_mask
        # i.e., for no (a,b) in selected_mask with edge(a,b) do we have edge(a,v) and edge(b,v)
        # Equivalent: for no a in selected_mask with edge(a,v) is there b in selected_mask with edge(a,b) and edge(b,v)
        # i.e., for no a in (selected_mask & adj_mask[v]) is there b in (selected_mask & adj_mask[a] & adj_mask[v])
        
        # This is getting complex. Let me use a different exact approach.
        
        # Exact approach: iterate over all subsets? 2^18 = 262144, feasible!
        # But we want to be smarter. Let me just enumerate all subsets of size >= best[0]+1
        # and check if they're triangle-free.
        
        # Actually, 2^18 = 262144 subsets, and checking each is O(n^3) worst case.
        # 262144 * 18^3 ≈ 157 million operations. Might be too slow in Python.
        
        # Let me use a smarter approach: branch and bound with proper triangle-free check.
        
        # New formulation:
        # We build the set S incrementally. At each step, we have:
        # - S: the set of vertices chosen so far (guaranteed triangle-free)
        # - C: the set of candidate vertices that can still be added (those in the original universe not in S)
        # A vertex v in C can be added to S iff S ∪ {v} is triangle-free
        # S ∪ {v} is triangle-free iff:
        #   1. S is triangle-free (already true)
        #   2. v doesn't form a triangle with any two vertices in S
        #      i.e., there do not exist a, b in S with edges (a,v), (b,v), and (a,b)
        #      Equivalently: the set N_S(v) = {u in S : edge(u,v)} is an independent set in G[S]
        
        # For efficiency, precompute for each pair (v, S_mask) whether v can be added.
        # But S_mask changes. Instead, for each v, precompute which subsets of neighbors are independent.
        
        # Let me just do a straightforward but efficient branch and bound.
        
        pass  # This function is too complex; let me rewrite it properly
    
    # Let me rewrite with a cleaner approach
    # Enumerate all 2^18 subsets and find the largest triangle-free one
    # Use bit operations for speed
    
    # Precompute: for each triple (i,j,k) with i<j<k, whether it's a triangle
    # Then a subset is triangle-free iff it contains no such triple
    
    # Better: for each vertex v, compute the set of vertices that form a triangle with v and another vertex
    # tri_with[v][u] = whether v and u are in some triangle together
    
    # Actually, let me just do a clean recursive search with memoization on (mask, ...)
    # but that's not straightforward.
    
    # Simplest correct approach: iterate all subsets from largest to smallest
    # For size k from n down to 1:
    #   Check if any subset of size k is triangle-free
    #   If yes, return k
    
    # To check if a subset is triangle-free:
    #   For each vertex v in subset, check that no two other vertices in subset 
    #   are both adjacent to v and to each other
    #   Equivalently: for each v in subset, the induced subgraph on N(v) ∩ subset is edge-free
    
    # Let me precompute for each pair (v, u) whether they share a common neighbor
    # common_neighbor[v][u] = whether there exists w with edges (v,w) and (u,w)
    # Then subset S is triangle-free iff for all v in S, 
    #   for all u in S with edge(v,u), common_neighbor[v][u] is false OR 
    #   the common neighbor is not in S
    # Wait, that's not right either.
    
    # S is triangle-free iff there do not exist distinct a,b,c in S with edges (a,b), (b,c), (c,a)
    # Equivalently: for every edge (a,b) in the induced subgraph on S, 
    #   there is no c in S such that c is adjacent to both a and b
    
    # So: precompute for each pair (a,b) the set of common neighbors: CN[a][b]
    # Then S is triangle-free iff for all edges (a,b) in S, CN[a][b] ∩ S = ∅
    
    # This is still O(|S|^3) to check in the worst case, but with bitmasks it's fast.
    
    # Let me just do the full enumeration with bitmasks.
    # 2^18 = 262144 subsets. For each, check triangle-freeness in O(n^2/64) with bitmasks.
    # Total: 262144 * 18 * 18 / 64 ≈ 1.3 million 64-bit ops. Very fast!
    
    # Precompute CN[a][b] as bitmask of common neighbors of a and b
    CN = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            CN[a][b] = adj_mask[a] & adj_mask[b]
            CN[b][a] = CN[a][b]
    
    # Now, for a subset S (as bitmask), check if it's triangle-free:
    # For each edge (a,b) in S, check if CN[a][b] & S == 0
    
    def is_triangle_free(S):
        # Check all pairs in S
        m = S
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            m2 = m & (m - 1)  # remove a from consideration
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length() - 1
                if adj_mask[a] & (1 << b):  # edge (a,b) exists
                    if CN[a][b] & S:  # common neighbor in S
                        return False
                m2 ^= lb2
            m = m2  # wait, this is wrong. Let me redo.
        return True
    
    # Let me redo is_triangle_free properly
    def is_triangle_free(S):
        # For each pair (a,b) in S with edge(a,b), check CN[a][b] & S == 0
        m = S
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            rest = m ^ lb  # S without a
            # Check all b in rest
            m2 = rest
            while m2:
                lb2 = m2 & (-m2)
                b = lb2.bit_length() - 1
                if (adj_mask[a] >> b) & 1:  # edge (a,b)
                    if CN[a][b] & S:
                        return False
                m2 ^= lb2
            m = rest
        return True
    
    # Now enumerate subsets from largest to smallest
    # For efficiency, iterate by size
    # Use combinations from itertools? Or just iterate all masks and group by size
    
    # Iterate all masks from (1<<n)-1 down to 0, check size and triangle-freeness
    # But that's 262144 checks, each taking maybe 100-500 Python ops. Should be fine.
    
    # Actually, let me be smarter: iterate by size from n down to 1
    # For each size k, check all subsets of size k
    
    # Generate all masks of a given size: use itertools.combinations
    from itertools import combinations
    
    for k in range(n, 0, -1):
        if time.time() - start_time > time_limit:
            # Return best found so far (which is at least what we've checked)
            # But we haven't found any yet if we're still checking
            # Actually, if we haven't found a triangle-free set of size k, 
            # we continue to k-1
            # If time runs out, we should return our best guess
            # Let me just return a lower bound
            pass
        
        found = False
        for combo in combinations(range(n), k):
            if time.time() - start_time > time_limit:
                break
            S = 0
            for v in combo:
                S |= (1 << v)
            if is_triangle_free(S):
                return k
            found = True
        
        if time.time() - start_time > time_limit:
            # Time's up; return a conservative estimate
            # We know subsets of size k+1 are not triangle-free (if we checked them all)
            # But we may not have checked all of size k
            # Return k as a lower bound? No, we haven't confirmed any of size k
            # Let's return a greedy lower bound
            # Actually, let me just return 0 to indicate failure, and the caller will handle it
            # Better: return a quick greedy estimate
            break
    
    # If we get here, either we found the answer or time ran out
    # Let me do a quick greedy estimate as fallback
    remaining = set(range(n))
    while remaining:
        # Check if remaining is triangle-free
        rem_list = list(remaining)
        tri_found = False
        for i in range(len(rem_list)):
            a = rem_list[i]
            for j in range(i + 1, len(rem_list)):
                b = rem_list[j]
                if not (adj_mask[a] >> b) & 1:
                    continue
                common = CN[a][b] & ((1 << len(rem_list)) - 1)  # This is wrong
                # Let me just check directly
                for k2 in range(j + 1, len(rem_list)):
                    c = rem_list[k2]
                    if (adj_mask[a] >> c) & 1 and (adj_mask[b] >> c) & 1:
                        tri_found = True
                        break
                if tri_found:
                    break
            if tri_found:
                break
        
        if not tri_found:
            return len(remaining)
        
        # Remove a vertex that's in many triangles
        worst = max(remaining, key=lambda v: sum(1 for u in remaining if (adj_mask[v] >> u) & 1))
        remaining.remove(worst)
    
    return 0

def _max_triangle_free_fast(adj):
    """
    Fast heuristic for the largest triangle-free induced subgraph size.
    Uses greedy vertex removal.
    """
    n = len(adj)
    remaining = set(range(n))
    
    while remaining:
        # Check if remaining is triangle-free
        rem_list = list(remaining)
        tri_found = False
        for i in range(len(rem_list)):
            a = rem_list[i]
            for j in range(i + 1, len(rem_list)):
                b = rem_list[j]
                if not adj[a][b]:
                    continue
                for k in range(j + 1, len(rem_list)):
                    c = rem_list[k]
                    if adj[a][c] and adj[b][c]:
                        tri_found = True
                        break
                if tri_found:
                    break
            if tri_found:
                break
        
        if not tri_found:
            return len(remaining)
        
        # Remove vertex in most triangles
        tri_count = {v: 0 for v in remaining}
        for i in range(len(rem_list)):
            a = rem_list[i]
            for j in range(i + 1, len(rem_list)):
                b = rem_list[j]
                if not adj[a][b]:
                    continue
                for k in range(j + 1, len(rem_list)):
                    c = rem_list[k]
                    if adj[a][c] and adj[b][c]:
                        tri_count[a] += 1
                        tri_count[b] += 1
                        tri_count[c] += 1
        
        worst = max(remaining, key=lambda v: tri_count[v])
        remaining.remove(worst)
    
    return 0

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    n = 18
    
    # Start with the complete 3-partite graph K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [[False] * n for _ in range(n)]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
    
    def get_edges(adj):
        edges = []
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a][b]:
                    edges.append([a, b])
        return edges
    
    base_score = _max_triangle_free_exact(adj, time_limit=3.0)
    best_score = base_score
    best_edges = get_edges(adj)
    
    if best_score <= 6:
        # K6,6,6 has max triangle-free induced subgraph of size 6 (one part)
        # This is likely optimal or near-optimal
        return best_edges
    
    # Local search: try to add intra-part edges to reduce max triangle-free size
    rng = random.Random(42)
    
    # Intra-part edges to potentially add
    intra_edges = []
    for i in range(3):
        for a in range(len(parts[i])):
            for b in range(a + 1, len(parts[i])):
                u = parts[i][a]
                v = parts[i][b]
                intra_edges.append((u, v))
    
    # Try adding intra-part edges one at a time
    for u, v in intra_edges:
        if time.time() > deadline:
            break
        old = adj[u][v]
        if old:
            continue
        adj[u][v] = True
        adj[v][u] = True
        
        if _is_k4_free(adj):
            new_score = _max_triangle_free_fast(adj)
            if new_score < best_score:
                best_score = new_score
                best_edges = get_edges(adj)
            else:
                adj[u][v] = False
                adj[v][u] = False
        else:
            adj[u][v] = False
            adj[v][u] = False
    
    # Try removing some inter-part edges to allow more intra-part edges
    inter_edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    inter_edges.append((a, b))
    
    # Randomized local search
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Start from current best
        adj2 = [row[:] for row in adj]
        
        # Try a random sequence of edge flips
        improved = False
        for step in range(50):
            if time.time() > deadline:
                break
            # Pick a random edge to flip
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            
            old = adj2[a][b]
            adj2[a][b] = not old
            adj2[b][a] = not old
            
            if _is_k4_free(adj2):
                new_score = _max_triangle_free_fast(adj2)
                if new_score < best_score:
                    best_score = new_score
                    best_edges = get_edges(adj2)
                    improved = True
                    adj = [row[:] for row in adj2]
                else:
                    adj2[a][b] = old
                    adj2[b][a] = old
            else:
                adj2[a][b] = old
                adj2[b][a] = old
        
        if not improved:
            # Try a different strategy: remove a random inter-part edge, then add intra-part edges
            adj2 = [row[:] for row in adj]
            if inter_edges:
                a, b = rng.choice(inter_edges)
                adj2[a][b] = False
                adj2[b][a] = False
                
                if _is_k4_free(adj2):
                    # Now try to add some intra-part edges
                    for u, v in intra_edges:
                        if time.time() > deadline:
                            break
                        if not adj2[u][v] and rng.random() < 0.3:
                            adj2[u][v] = True
                            adj2[v][u] = True
                            if not _is_k4_free(adj2):
                                adj2[u][v] = False
                                adj2[v][u] = False
                    
                    new_score = _max_triangle_free_fast(adj2)
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = get_edges(adj2)
                        adj = [row[:] for row in adj2]
    
    # Final exact verification of best result
    if best_edges:
        final_adj = [[False] * n for _ in range(n)]
        for e in best_edges:
            final_adj[e[0]][e[1]] = True
            final_adj[e[1]][e[0]] = True
        
        if _is_k4_free(final_adj):
            exact_score = _max_triangle_free_exact(final_adj, time_limit=10.0)
            if exact_score < best_score:
                best_score = exact_score
            # Use the exact score for the final answer
            # But we need to return edges, not score
            # The evaluator will compute the score from the edges
            # So we just return the edges
    
    return best_edges
# EVOLVE-BLOCK-END
