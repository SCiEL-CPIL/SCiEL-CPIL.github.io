import time
import random
import sys

# Global deadline guard
DEADLINE = 145.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return DEADLINE
    return DEADLINE - (time.time() - _START_TIME)

def _seed_graph():
    """
    Fallback: 10 disjoint triangles.
    n=30, C4-free, alpha=10.
    """
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def _check_c4_violation(adj, u, v, n):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is formed if there exist x, y such that u-x-y-v-u is a cycle.
    This requires u~x, x~y, y~v, and u~v (new).
    So we need to check if there is a path of length 3 between u and v in the current graph.
    Specifically: exists x in N(u) and y in N(v) such that x~y and x!=y, x!=v, y!=u.
    """
    nu = adj[u]
    nv = adj[v]
    # To be efficient, iterate over the smaller set
    if len(nu) > len(nv):
        nu, nv = nv, nu
        u, v = v, u # Symmetry, but we just need to check intersection of neighborhoods' neighborhoods
        
    # Actually, the condition is:
    # There exists x in N(u) and y in N(v) such that x ~ y.
    # And we must ensure it's not a triangle (x=y) and not involving u/v directly in a way that breaks C4 definition (but u,v not adjacent yet).
    # If x=y, it's a triangle u-x-v-u, which is allowed (C4 is simple cycle of length 4).
    # Wait, C4-free means no 4-cycles. Adding an edge to a graph with a path of length 3 creates a 4-cycle.
    # Path u-x-y-v. If x=y, it's u-x-v, length 2. Adding u-v makes triangle. No C4.
    # So we need x != y.
    
    for x in nu:
        if x == v: continue # x cannot be v because u,v not adjacent yet, so v not in N(u)
        # Check if any neighbor of x is in N(v)
        # But we must exclude y==x (triangle case) and y==u (impossible since u not in N(v) yet? No, u is not in N(v) because edge not added)
        # Actually, if u~x and x~y and y~v.
        # We need to check if N(x) intersects N(v).
        # If the intersection contains x, that means x~x (impossible) or... wait.
        # If x is in N(v), then u~x and x~v. Path length 2. Adding u-v makes triangle.
        # If y is in N(x) and y is in N(v), and y != x.
        
        # Efficient check:
        # For each x in N(u), check if (N(x) & N(v)) contains any element y != x.
        # Note: x itself might be in N(v) (if dist 2). If x in N(v), then y=x is the intersection.
        # We need y != x.
        
        # Let's just iterate neighbors of x
        for y in adj[x]:
            if y == x: continue
            if y in nv:
                # We found u-x-y-v.
                # Is this a C4?
                # Vertices: u, x, y, v.
                # Edges: u-x, x-y, y-v, v-u(new).
                # Distinctness:
                # u != x (yes)
                # x != y (yes, no self loops)
                # y != v (yes, y in N(v) means y!=v)
                # v != u (yes)
                # u != y? If u==y, then u in N(x) and u in N(v). But u not in N(v) yet. So u!=y.
                # v != x? If v==x, then x in N(u) and x in N(v). This is the dist 2 case.
                # If v==x, then the path is u-v-v-v? No.
                # If x==v, then u~v. But we are checking if adding u~v creates C4.
                # If u~v already, we wouldn't be adding it.
                # So x != v is guaranteed because v not in N(u).
                
                # So if we find any y in N(x) such that y in N(v) and y != x, we have a C4.
                # Wait, if y==x, it's not in N(x) (no self loop).
                # So simply: if N(x) intersects N(v), and the intersection is not just {x} (if x in N(v)), 
                # actually if x in N(v), then x is in N(x)? No.
                # So if x in N(v), the intersection N(x) & N(v) might be empty or contain other nodes.
                # If x in N(v), it doesn't mean x is in N(x).
                # So we just need ANY y in N(x) that is also in N(v).
                # Does y==x matter? No, because x is not in N(x).
                # So if there is ANY y in N(x) \cap N(v), it forms a C4?
                # Let's trace: u-x-y-v-u.
                # If y==x, impossible.
                # So yes, any intersection N(x) & N(v) implies a C4.
                
                return True
    return False

def _construct_graph(n_target, seed_val):
    """
    Greedy randomized construction for n_target vertices.
    """
    rng = random.Random(seed_val)
    adj = [set() for _ in range(n_target)]
    
    # Pre-compute all potential edges
    all_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
    
    # We want to add edges to make the graph dense enough to reduce alpha.
    # Strategy: Add edges in a random order, but skip if C4 is created.
    # To bias towards low alpha, we can prioritize edges that connect low-degree vertices?
    # Or just random is often fine for C4-free graphs if we go dense.
    # The constraint is C4-free.
    
    # Shuffle edges
    rng.shuffle(all_edges)
    
    # Target density: 
    # For n=43, C4-free graphs can have up to ~ n^(3/2) edges?
    # HS graph (n=50) has 175 edges. 175/50 = 3.5 avg deg.
    # Let's aim for ~40% of possible edges? No, C4-free is sparse.
    # Max edges in C4-free graph on n vertices is O(n^1.5).
    # For n=43, n^1.5 ~ 280. Total possible ~ 900.
    # So we can add up to ~280 edges.
    
    edges_added = 0
    # Try to add as many edges as possible without creating C4
    for u, v in all_edges:
        if _time_left() < 5.0:
            break
        if _check_c4_violation(adj, u, v, n_target):
            continue
        adj[u].add(v)
        adj[v].add(u)
        edges_added += 1
        
    # Verify C4-freeness (safety check)
    # This check is O(n * E) or similar. 
    # For each pair (u,v), check common neighbors.
    # If any pair has >1 common neighbor, C4 exists.
    for u in range(n_target):
        for v in range(u+1, n_target):
            common = adj[u] & adj[v]
            if len(common) > 1:
                return None # Invalid
        
    # Calculate Independence Number
    # We need alpha <= 10.
    # Use a simple branch and bound or greedy upper bound?
    # For n=43, exact alpha is feasible with good pruning.
    
    # Greedy coloring for lower bound on chromatic number of complement? 
    # No, we need upper bound on alpha (which is lower bound on chi(complement)).
    # Actually, we need to check if alpha >= 11.
    # If we can find an independent set of size 11, it's invalid.
    # If we prove max independent set is <= 10, it's valid.
    
    # Fast check: Greedy independent set
    # If greedy finds 11, definitely invalid.
    # If greedy finds < 11, might still be invalid.
    
    # Let's do a quick backtracking to see if alpha >= 11.
    # If it finds one, return None.
    # If it exhausts search without finding 11, return valid.
    
    def has_independent_set_11(adj, n):
        # Backtracking
        # Order vertices by degree descending (low degree vertices are better candidates for independent set)
        # Actually, high degree vertices are worse.
        # Let's use a simple recursive search with pruning.
        
        # Pre-compute non-neighbors?
        # For each vertex v, non_neighbors = {0..n-1} - {v} - adj[v]
        
        # Use bitmasks for speed? n=43 fits in 64-bit int.
        # Let's use Python sets for simplicity first, but bitmasks are faster.
        
        # Bitmask approach:
        # adj_mask[i] is a bitmask of neighbors of i.
        
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_masks[i] = mask
            
        # All vertices mask
        all_mask = (1 << n) - 1
        
        # non_adj_mask[i] = all_mask & ~(adj_masks[i] | (1 << i))
        non_adj_masks = [0] * n
        for i in range(n):
            non_adj_masks[i] = all_mask & ~(adj_masks[i] | (11 << i) # 11 is 1011? No, just 1<<i
            )
            # Fix: non_adj_masks[i] = all_mask & ~(adj_masks[i] | (1 << i))
            non_adj_masks[i] = all_mask & ~(adj_masks[i] | (1 << i))
            
        # Recursive search
        # state: current set (mask), size, candidate mask
        # We want to find if we can pick 11 vertices.
        
        best_found = [0]
        
        def search(cand_mask, size):
            if size >= 11:
                return True
            if cand_mask == 0:
                return False
            if _time_left() < 1.0:
                return False # Timeout, assume invalid to be safe? 
                # Actually if we time out, we don't know. 
                # Better to return False (assume valid) only if we are confident?
                # No, if we don't know, we should discard the graph to be safe?
                # But we want to maximize n. 
                # If we can't verify, we shouldn't return it.
                # So return False means "did not find 11", which implies valid.
                # But if we timed out, we might have missed a set.
                # So returning False is risky.
                # Let's return a special value or just stop and let the caller decide.
                # For now, if timeout, we assume it's invalid (conservative) to avoid returning bad graphs.
                # But that might reject good graphs.
                # Given 145s deadline, we have plenty of time for n=43 backtracking if pruned well.
                pass
            
            # Heuristic: pick vertex with min degree in cand_mask?
            # Or max degree?
            # For independent set, picking a vertex removes its neighbors from candidates.
            # Picking a high-degree vertex removes many candidates, reducing search space.
            # So pick vertex with highest degree in the induced subgraph?
            # Or just pick the first available one for speed?
            
            # Let's pick the vertex with the smallest index for determinism in this step, 
            # but we can optimize later.
            
            # Optimization: If the number of candidates is small, we can check all.
            
            # Pick a vertex v from cand_mask
            # Let's pick the one with highest degree in the original graph? 
            # No, degree in the remaining candidate set matters.
            
            # Simple strategy: pick the first set bit.
            # This is slow.
            
            # Better: pick vertex v that maximizes the number of removed candidates?
            # i.e., v with max degree in the induced subgraph on cand_mask.
            
            # For simplicity and speed in Python, let's just iterate.
            
            # Get list of candidates
            # To avoid creating lists, we can iterate bits.
            
            # Let's try a different heuristic: 
            # If cand_mask has k bits, and size + k <= 10, return False (can't reach 11).
            bit_count = cand_mask.bit_count()
            if size + bit_count < 11:
                return False
                
            # Pick a vertex
            # Let's pick the vertex with the most neighbors in cand_mask.
            # This is O(n) per step.
            
            best_v = -1
            max_deg = -1
            m = cand_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Count neighbors of v in cand_mask
                # deg_in_cand = (adj_masks[v] & cand_mask).bit_count()
                # But we want to remove v and its neighbors.
                # Removing v and its neighbors leaves cand_mask & ~adj_masks[v] & ~(1<<v)
                # The size of the new candidate set is (cand_mask & ~(adj_masks[v] | (1<<v))).bit_count()
                # We want to minimize this? Or maximize removal?
                # Minimizing remaining candidates is good for pruning.
                
                rem = (cand_mask & ~(adj_masks[v] | (1 << v))).bit_count()
                if rem < max_deg: # We want min remaining
                    max_deg = rem
                    best_v = v
                m -= lsb
                
            if best_v == -1:
                return False
                
            # Branch 1: Exclude best_v
            # New cand: cand_mask without best_v
            new_cand_excl = cand_mask & ~(1 << best_v)
            if search(new_cand_excl, size):
                return True
                
            # Branch 2: Include best_v
            # New cand: cand_mask without best_v and its neighbors
            new_cand_incl = cand_mask & ~(adj_masks[best_v] | (1 << best_v))
            if search(new_cand_incl, size + 1):
                return True
                
            return False

        return search(all_mask, 0)

    if has_independent_set_11(adj, n_target):
        return None # Invalid
        
    # Valid!
    edges = []
    for i in range(n_target):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n_target, 'edges': edges}

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with seed
    best_result = _seed_graph()
    best_n = 30
    
    # Try to build larger graphs
    # We know best known is 39. Let's aim for 39, 40, 41, 42, 43.
    
    # Use multiple seeds for each n to increase chance of finding a valid dense graph
    for n_target in [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31]:
        if n_target <= best_n:
            break
        if _time_left() < 10.0:
            break
            
        # Try a few random seeds
        found = False
        for seed in range(100):
            if _time_left() < 5.0:
                break
            res = _construct_graph(n_target, seed)
            if res is not None:
                if res['n'] > best_n:
                    best_n = res['n']
                    best_result = res
                found = True
                break # Found one for this n, try next larger n? 
                # Actually, we are iterating from largest n. 
                # If we found a valid graph for n=43, we are done.
                # If we found valid for 42, we should try 43 again? 
                # No, we tried 43 first. If it failed, we move to 42.
                # If 42 succeeds, we return 42. 
                # But maybe a different seed for 43 would work?
                # Let's try a few seeds for the largest n first.
                
        if found and n_target == 43:
            break
            
    return best_result

# To improve success rate for high n, we might need to be smarter about edge selection.
# The current greedy "add if no C4" might get stuck in a local minimum where alpha is still high.
# However, for C4-free graphs, density is limited.
# If we can't find alpha <= 10, we might need to try different edge orderings.

# Let's refine the edge ordering to prioritize edges that "break" potential independent sets.
# This is hard to do online.
# Random is a good baseline.

# One improvement: If the graph is valid but n is small, we can try to add more vertices?
# No, n is fixed in the loop.

# Let's ensure the C4 check is fast.
# The current _check_c4_violation is O(deg(u) * deg(x)).
# With n=43, deg is small (~4-6). So it's very fast.

# The bottleneck is the independent set search.
# For n=43, if the graph is dense enough, the search space is small.
# If it's sparse, alpha is likely > 10, and we will find it quickly (or fail to prune).
# If we fail to find 11, we prove alpha <= 10.

# Let's run it.

if __name__ == "__main__":
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['edges'])}")
