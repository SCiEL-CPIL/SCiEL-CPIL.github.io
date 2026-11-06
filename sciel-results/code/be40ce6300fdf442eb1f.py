"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # We want to find a C4-free graph on n vertices with alpha < 11.
    # Best known is 39. We aim for 40+.
    
    # Strategy: Random greedy construction with backtracking-like pruning.
    # 1. Start with a random ordering of vertices.
    # 2. Add edges greedily to maximize density while avoiding C4.
    # 3. Check independence number using a branch-and-bound or exact solver.
    # 4. If alpha >= 11, try to fix by adding edges or removing vertices (if n is fixed, we must add edges).
    #    Actually, we want to MAXIMIZE n. So we try n=43, 42, ... down to 39.
    #    For a given n, we try to construct a C4-free graph with alpha < 11.
    
    def is_c4_free(adj):
        # adj is a list of sets
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    if len(common) > 1:
                        return False
        return True

    def has_independent_set(adj, k):
        # Returns True if there is an independent set of size k.
        # Uses a simple branch and bound.
        n = len(adj)
        
        # Convert to bitmasks for speed if n <= 64
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            masks[i] = mask
        
        # independent_set(v, remaining, size)
        # v: current vertex to consider
        # remaining: bitmask of available vertices
        # size: number of vertices to pick
        import sys
        sys.setrecursionlimit(10000)
        
        # Memoization might be too large. Use simple recursion with pruning.
        # Pruning: if popcount(remaining) < size, return False.
        # Pruning: if v == n, return size == 0.
        
        def popcount(x):
            return bin(x).count('1')
            
        def search(v, remaining, size):
            if size == 0:
                return True
            if v == n:
                return False
            if popcount(remaining) < size:
                return False
            
            # Option 1: Exclude v
            if search(v + 1, remaining, size):
                return True
            
            # Option 2: Include v
            if (remaining & (1 << v)) != 0:
                new_remaining = remaining & ~(masks[v] | (1 << v))
                if search(v + 1, new_remaining, size - 1):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return search(0, all_mask, k)

    def try_construct(n, seed):
        random.seed(seed)
        # Start with empty graph
        adj = [set() for _ in range(n)]
        
        # Random order of edges to consider
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        # Greedy: add edge if it doesn't create a C4
        # This creates a sparse graph. We want to maximize edges to minimize alpha.
        # But we also want to ensure alpha < 11.
        # A random graph with appropriate density might work.
        
        # Better: Start with a known good structure?
        # Let's try a random regular-like graph.
        
        # Alternative: Use the polarity graph of PG(2, q) if n is close to q^2+q+1.
        # For q=6, n=43. The polarity graph of PG(2,6) has 43 vertices.
        # It is C4-free. What is its independence number?
        # Theorem: The independence number of the polarity graph of PG(2,q) is at most q+1? No.
        # Actually, for q even, the polarity graph is a C4-free graph.
        # The independence number is related to the maximum size of a set of points no two of which are orthogonal.
        # For q=6, the graph has 43 vertices.
        # Let's explicitly construct the polarity graph of PG(2,6).
        
        if n == 43:
            return construct_polarity_graph_43()
            
        # If n != 43, try random greedy
        for u, v in edges:
            # Check if adding (u,v) creates a C4
            common = adj[u] & adj[v]
            if len(common) == 0:
                adj[u].add(v)
                adj[v].add(u)
            # If len(common) > 0, adding edge creates a C4 (since u,v already have a common neighbor, and adding edge makes them adjacent, so the triangle u-v-w-u' ... wait.
            # C4 is u-v-w-x-u.
            # If u and v have a common neighbor w, then u-w-v is a path of length 2.
            # If we add edge (u,v), we get a triangle.
            # A C4 requires two distinct common neighbors?
            # No. A C4 is a cycle of length 4.
            # If u and v are adjacent, they cannot have a common neighbor w, because then u-v-w-u is a triangle, not a C4.
            # Wait. If u and v are adjacent, and they have a common neighbor w, then {u,v,w} is a triangle.
            # The definition of C4-free is that there are no 4 distinct vertices a,b,c,d such that a-b-c-d-a is a cycle.
            # This is equivalent to: no two vertices have more than one common neighbor.
            # Proof: If a and b have two common neighbors c and d, then a-c-b-d-a is a C4.
            # Conversely, if a-c-b-d-a is a C4, then a and b have common neighbors c and d.
            # So the condition is: for any u != v, |N(u) \cap N(v)| <= 1.
            # If we add edge (u,v), it does not change N(u) \cap N(v) directly, but it adds v to N(u) and u to N(v).
            # The new common neighbors of u and v are the old common neighbors plus... wait.
            # u and v are now adjacent.
            # The common neighbors of u and v are vertices w such that w is adjacent to u and w is adjacent to v.
            # Adding the edge (u,v) does not make u a neighbor of u, nor v a neighbor of v.
            # So N(u) \cap N(v) remains the same set of vertices w.
            # However, the condition is on the graph structure.
            # If we add edge (u,v), we must ensure that no C4 is formed.
            # A C4 involving the new edge (u,v) would look like u-v-w-x-u.
            # This requires w to be a neighbor of v and x to be a neighbor of u, and w-x to be an edge.
            # Also w != x.
            # This is equivalent to checking if there is a path of length 2 from u to v in the graph WITHOUT the edge (u,v)?
            # No.
            # Let's stick to the common neighbor rule.
            # Condition: For all pairs (a,b), |N(a) \cap N(b)| <= 1.
            # When we add edge (u,v):
            # 1. For any w, if w is in N(u) \cap N(v), the count is unchanged.
            # 2. But u and v are now neighbors. Do they have a common neighbor?
            #    Yes, any w in N(u) \cap N(v).
            #    If |N(u) \cap N(v)| >= 1, then u and v have at least 1 common neighbor.
            #    If |N(u) \cap N(v)| >= 2, then u and v have at least 2 common neighbors -> C4.
            #    So we must have |N(u) \cap N(v)| <= 1.
            #    This is the condition we checked!
            # 3. What about other pairs?
            #    Adding (u,v) increases N(u) by v and N(v) by u.
            #    Consider a pair (a,b) where a != u,v and b != u,v.
            #    N(a) and N(b) are unchanged.
            #    Consider pair (u,a) where a != v.
            #    N(u) gains v. N(a) is unchanged.
            #    New common neighbors: (Old N(u) \cap N(a)) U (if v in N(a) then {v} else {}).
            #    So the count increases by 1 if v is in N(a).
            #    We need new count <= 1.
            #    So if v in N(a), we need Old Count(u,a) == 0.
            #    Similarly, if u in N(b), we need Old Count(v,b) == 0.
            
            # So the greedy check "if len(common) == 0" is insufficient.
            # We need to check:
            # 1. len(adj[u] & adj[v]) <= 1
            # 2. For all a in adj[v]: len(adj[u] & adj[a]) == 0 (because v is now in adj[u], so if a is in adj[v] and adj[u] & adj[a] was 1, it becomes 2)
            #    Wait. If a is in adj[v], then v is in adj[a].
            #    New adj[u] contains v.
            #    So v is a common neighbor of u and a IF v is in adj[a] (which is true) AND v is in adj[u] (which is true now).
            #    So v becomes a new common neighbor of u and a.
            #    So we need (adj[u] & adj[a]) to be empty BEFORE adding (u,v)?
            #    No, we need (adj[u] \ {v} & adj[a] \ {u}) to be empty?
            #    Let's re-evaluate.
            #    Old common neighbors of u and a: C_old = adj[u] & adj[a].
            #    New adj[u] = adj[u] U {v}.
            #    New common neighbors of u and a: C_new = (adj[u] U {v}) & adj[a] = C_old U ({v} if v in adj[a] else {}).
            #    Since a is in adj[v], v is in adj[a].
            #    So C_new = C_old U {v}.
            #    We need |C_new| <= 1.
            #    So if v in C_old, impossible (but v not in adj[u] yet).
            #    So v not in C_old.
            #    So |C_new| = |C_old| + 1.
            #    We need |C_old| + 1 <= 1 => |C_old| == 0.
            #    So for all a in adj[v], adj[u] & adj[a] must be empty.
            #    Similarly, for all b in adj[u], adj[v] & adj[b] must be empty.
            
            if len(adj[u] & adj[v]) > 1:
                continue
            if len(adj[u] & adj[v]) == 1:
                # If they share 1 neighbor, adding edge is okay for the pair (u,v) itself.
                # But we still need to check the ripple effect.
                pass
            
            valid = True
            # Check ripple for neighbors of v
            for a in adj[v]:
                if adj[u] & adj[a]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check ripple for neighbors of u
            for b in adj[u]:
                if adj[v] & adj[b]:
                    valid = False
                    break
            if not valid:
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            
        # Check alpha
        if has_independent_set(adj, 11):
            return None
        else:
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}

    def construct_polarity_graph_43():
        # Construct polarity graph of PG(2,6)
        # Points are (x,y,z) in F_6^3 \ {0}, modulo scaling.
        # q=6. Number of points = 6^2 + 6 + 1 = 43.
        # Polarity: (x,y,z) is adjacent to (x',y',z') if xx' + yy' + zz' = 0.
        # Note: In characteristic 2, this is symmetric. In char 6 (not a field, 6 is composite), we use Z_6?
        # PG(2,q) requires q to be a prime power. 6 is not a prime power.
        # So PG(2,6) does not exist.
        # The best known C4-free graphs are often from finite fields or projective planes.
        # Prime powers near 43: 37 (prime), 49 (7^2).
        # For q=37, n = 37^2+37+1 = 1407. Too big.
        # We need n ~ 40.
        # The Erdos-Renyi construction for R(C4, k) uses random graphs.
        # For k=11, n=39 is known.
        # Let's try random search for n=40, 41, 42, 43.
        return None

    # Try n from 43 down to 39
    for n in range(43, 38, -1):
        # Try multiple random seeds
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            res = try_construct(n, seed)
            if res is not None:
                return res
        # If n failed, try next lower n
        # Note: try_construct might be slow.
        
    # Fallback to known 39-vertex construction if found?
    # We don't have it hardcoded.
    # The seed program returns n=30.
    # We should return the best found.
    # If none found, return the seed result to avoid 0 score.
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
