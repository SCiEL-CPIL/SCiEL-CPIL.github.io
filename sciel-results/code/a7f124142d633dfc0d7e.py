import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Fallback: The seed graph (Andrasfai graph on 26 vertices)
    def get_seed_graph():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Target: n = 39
    target_n = 39
    
    # Helper: Check if graph is triangle-free using bitmasks
    def is_triangle_free_bitmask(n, adj_mask):
        for u in range(n):
            # For each neighbor v of u, check if any neighbor of v is also a neighbor of u
            # Actually, simpler: for each u, check if any two neighbors of u are adjacent.
            # Or: for each edge (u,v), check if they have a common neighbor.
            # Let's stick to the standard check:
            # For each u, let N(u) be neighbors.
            # If there exists v, w in N(u) such that (v,w) is an edge, then triangle.
            # With bitmasks:
            # For each u:
            #   Let mask_u = adj_mask[u]
            #   For each v in mask_u:
            #     if adj_mask[v] & mask_u: return False
            # This is O(n * deg^2) effectively, but bit operations are fast.
            # Optimization: iterate only over set bits.
            mask_u = adj_mask[u]
            # We can iterate over neighbors of u
            # But checking all pairs is expensive.
            # Better: For each u, compute intersection of adj_mask[v] for v in N(u)?
            # Actually, the condition "triangle-free" is equivalent to:
            # For all u, the subgraph induced by N(u) is a matching? No, it's empty.
            # So N(u) must be an independent set.
            # Check if N(u) is independent:
            # For each v in N(u), if (adj_mask[v] & N(u)) != 0, then triangle.
            # We can do this:
            # Let common = 0
            # For v in N(u):
            #   common |= adj_mask[v]
            # If common & N(u) != 0, triangle.
            # This is O(deg(u)) bit ops.
            common = 0
            m = mask_u
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                common |= adj_mask[v]
                m ^= lsb
            if common & mask_u:
                return False
        return True

    # Helper: Check if alpha < 10 using Branch and Bound with Bitmasks
    def has_independent_set_of_size_k(n, adj_mask, k):
        # We want to find if there is an independent set of size k.
        # This is equivalent to finding a clique of size k in the complement.
        # Or just search for independent set in G.
        
        # B&B for Max Independent Set
        # State: current set (size), candidates (bitmask)
        # Pruning: if len(current) + popcount(candidates) < k, return False
        
        # Heuristic for vertex selection:
        # Pick vertex with highest degree in the induced subgraph of candidates?
        # Or lowest?
        # For Independent Set, picking a high degree vertex removes many candidates.
        # This might prune the "exclude" branch heavily?
        # Actually, standard MIS B&B picks a vertex, branches on Include/Exclude.
        # Include: remove v and its neighbors from candidates.
        # Exclude: remove v from candidates.
        
        # To speed up, we can pre-sort vertices or use a greedy lower bound.
        # Let's use a simple recursive search with bitmasks.
        
        def popcount(x):
            return x.bit_count()
            
        def dfs(candidates, current_size):
            if current_size >= k:
                return True
            if not candidates:
                return False
            # Pruning
            if current_size + popcount(candidates) < k:
                return False
            
            # Heuristic: Pick a vertex from candidates.
            # Let's pick the vertex with the highest degree in the subgraph induced by candidates.
            # This is expensive to compute each time.
            # Alternative: Pick the vertex with the lowest index?
            # Or use a greedy approximation to get a lower bound on MIS size to prune?
            # Let's try picking the vertex with max degree in candidates.
            
            # To avoid O(n^2) per node, we can just pick the first set bit or use a static order.
            # Let's try static order (lowest index) first for simplicity and speed, 
            # as n=39 is small.
            
            # Actually, a better heuristic for Independent Set is to pick the vertex 
            # with the *lowest* degree in the candidate set? 
            # If we include it, we remove few candidates.
            # If we exclude it, we remove 1.
            # High degree vertices are "constraining".
            
            # Let's stick to a simple strategy: pick the lowest set bit.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates: candidates without v and without neighbors of v
            # Note: adj_mask[v] contains neighbors in the full graph.
            # We need neighbors in the candidate set.
            # candidates & ~adj_mask[v] & ~(1 << v)
            new_candidates_include = candidates & ~adj_mask[v] & ~(1 << v)
            if dfs(new_candidates_include, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            new_candidates_exclude = candidates & ~(1 << v)
            if dfs(new_candidates_exclude, current_size):
                return True
            
            return False
        
        initial_candidates = (1 << n) - 1
        return dfs(initial_candidates, 0)

    def get_edges_from_shifts(n, S):
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append([i, j])
        return edges

    def build_adj_mask_from_shifts(n, S):
        adj_mask = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                adj_mask[i] |= (1 << j)
                # Symmetry: (i+j) % n is not s if s > n/2?
                # In circulant graph C_n(S) with S subset of 1..n-1,
                # edge exists if (v-u) % n in S or (u-v) % n in S.
                # If we define S as positive shifts 1..19, and n=39,
                # then for each i, j = (i+s) % n is a neighbor.
                # Also j = (i-s) % n = (i + n-s) % n is a neighbor.
                # But if we only iterate s in 1..19, we cover all pairs?
                # Yes, because if d is a distance, min(d, n-d) is in 1..19.
                # If s in S, then edge (i, i+s).
                # Is edge (i, i-s) covered?
                # i-s = i + (n-s). n-s is in 20..38.
                # We didn't include n-s in S.
                # So we must add both directions or ensure S is symmetric?
                # The definition of circulant graph C_n(S) usually requires S to be symmetric
                # or we just add edges for s in S and rely on undirected nature.
                # If we add edge (i, i+s), it's the same as (i+s, i).
                # So adj_mask[i] gets bit i+s.
                # adj_mask[i+s] should get bit i.
                # My loop: for i in range(n): for s in S: j = (i+s)%n.
                # This adds i->j.
                # It does NOT explicitly add j->i.
                # But adj_mask is for undirected graph, so we should set both.
                # However, if we just build adj_mask[i] |= 1<<j, 
                # then adj_mask[j] might not have bit i if (j-i)%n is not in S.
                # But (j-i)%n = s, which is in S.
                # Wait, if we iterate all i and all s in S, 
                # then for any edge (u,v) with (v-u)%n = s in S,
                # when i=u, j=v, we set adj_mask[u] |= 1<<v.
                # When i=v, is there an s' in S such that (u-v)%n = s'?
                # (u-v)%n = n-s.
                # If n-s is NOT in S, then adj_mask[v] will NOT get bit u from this loop.
                # So the graph would be directed? No, we want undirected.
                # So we must ensure that if s is in S, we treat the edge as undirected.
                # The standard way: for each i, for each s in S, add edge (i, (i+s)%n).
                # This generates the set of edges. Since the graph is undirected,
                # the edge (u,v) is the same as (v,u).
                # So adj_mask[u] should have v and adj_mask[v] should have u.
                # My loop adds u->v. It does not add v->u unless n-s is in S.
                # So we must manually add the reverse edge or ensure S contains both s and n-s?
                # No, S is just the set of "distances".
                # Correct construction:
                # for i in range(n):
                #   for s in S:
                #     j = (i+s) % n
                #     adj_mask[i] |= (1 << j)
                #     adj_mask[j] |= (1 << i)
                # This ensures symmetry.
                adj_mask[j] |= (1 << i)
        return adj_mask

    # Search for a valid circulant graph on n=39
    # We iterate over all subsets S of {1, ..., 19}.
    # There are 2^19 = 524,288 subsets.
    # We prioritize larger |S| because denser graphs have smaller alpha.
    
    shifts = list(range(1, 20))
    
    # Generate subsets by size descending
    # To avoid generating all 2^19 at once, we can iterate by size.
    # But 19 is small enough to just generate all and sort? 
    # Or use combinations.
    
    found = False
    best_graph = None
    
    # Pre-compute triangle-free check for shift sets?
    # Triangle free condition for C_n(S):
    # No a,b in S such that a+b in S or |a-b| in S (mod n, with reps 1..19).
    # Specifically:
    # 1. a+b <= 19 and a+b in S -> Triangle
    # 2. a+b > 19: (a+b)-39. If (a+b)-39 in S -> Triangle
    # 3. |a-b| in S -> Triangle
    
    def is_valid_shift_set(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b: continue
                # Check a+b
                s = a + b
                if s > 19:
                    s = 39 - s
                if s in S_set:
                    return False
                # Check |a-b|
                d = abs(a - b)
                if d in S_set:
                    return False
        return True

    # Iterate by size of S from 19 down to 1
    # This is O(2^19) total.
    # For each S, check triangle free.
    # If valid, build adj_mask, check alpha < 10.
    
    # Optimization: 
    # If |S| is small, alpha is likely large.
    # We can skip small S if we find a valid large S.
    # But we want to find *any* valid S.
    
    # Let's iterate through all subsets.
    # To make it faster, we can use bitmasks for S itself?
    # No, just use list of shifts.
    
    # We will iterate through all masks of 19 bits.
    # Mask m represents S.
    
    for mask in range(1 << 19):
        if time.time() > deadline:
            break
            
        # Count bits in mask
        size = mask.bit_count()
        
        # Optional: Skip if size is too small to hope for alpha < 10?
        # Average degree = 2 * size.
        # For n=39, if avg deg is low, alpha is high.
        # Turan bound: alpha >= n / (avg_deg + 1).
        # If alpha < 10, then n / (avg_deg + 1) < 10 => avg_deg + 1 > 3.9 => avg_deg > 2.9.
        # So size > 1.5. So size >= 2 is necessary.
        # This doesn't prune much.
        
        # Extract S
        S = []
        for i in range(19):
            if mask & (1 << i):
                S.append(i + 1)
                
        # Check triangle free
        if not is_valid_shift_set(S):
            continue
            
        # Build adj mask
        adj_mask = build_adj_mask_from_shifts(39, S)
        
        # Check alpha < 10
        if not has_independent_set_of_size_k(39, adj_mask, 10):
            # Valid!
            edges = get_edges_from_shifts(39, S)
            return {'n': 39, 'edges': edges}

    # If we didn't find one on 39, try n=38? Or return seed.
    # The task says maximize n. 39 is the target.
    # If we fail, we might try to construct n=38 or 37.
    # But the seed is 26.
    # Let's try a quick search for n=38 if 39 fails?
    # Given time constraints, if 39 fails, we probably don't have time for 38 search.
    # But we can return the seed.
    
    return get_seed_graph()

if __name__ == "__main__":
    # Quick test
    g = construct()
    print(f"Found graph with n={g['n']}, edges={len(g['edges'])}")
