import time
import random
import sys

DEADLINE = 115  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    
    # Try to find a valid graph for n from 43 down to 39
    # The task states best known is 39, but we try higher first.
    
    for n in range(43, 38, -1):
        if time.time() > DEADLINE - 5:
            break
            
        # Try multiple random trials for this n
        # Use a fixed number of trials to balance time
        trials = 200 if n > 40 else 500
        
        for trial in range(trials):
            if time.time() > DEADLINE - 2:
                break
                
            # Generate a random C4-free graph
            # We use a greedy approach: add edges if they don't create a C4
            # Randomize the order of vertex pairs to get different graphs
            rng = random.Random(trial * 1000 + n)
            
            adj = [0] * n  # Bitset representation of adjacency
            
            # List of all pairs
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(pairs)
            
            # Greedily add edges
            # To make the graph denser (lower alpha), we add edges if possible.
            # We can also skip some edges to avoid creating dense clusters that might have high alpha?
            # No, denser graph usually has lower alpha.
            # However, C4-free constraint limits density.
            
            for u, v in pairs:
                # Check if adding (u,v) creates a C4
                # Common neighbors of u and v
                # If |N(u) & N(v)| >= 2, adding (u,v) creates a C4?
                # No, adding (u,v) creates a C4 if there is a path u-w-x-v of length 3.
                # This means w is neighbor of u and x is neighbor of v?
                # Actually, C4 is u-v-w-x-u.
                # If we add edge (u,v), it creates a C4 if there exists w, x such that
                # u-w, w-x, x-v are edges.
                # This is equivalent to: there is a vertex w adjacent to u and a vertex x adjacent to v
                # such that w-x is an edge? No.
                
                # Let's re-verify the C4 condition.
                # A C4 is a cycle of length 4.
                # Adding edge (u,v) creates a C4 if there is a path of length 3 between u and v.
                # Path: u - w - x - v.
                # Edges: (u,w), (w,x), (x,v).
                # This requires w in N(u), x in N(v), and (w,x) in E.
                
                # Alternative check:
                # If we add (u,v), then for any common neighbor w of u and v, we don't have a C4 involving u,v,w directly.
                # The condition "No two vertices may have two common neighbours" is for the final graph.
                # If the graph is C4-free, then for any u != v, |N(u) & N(v)| <= 1.
                # If we add (u,v), we need to ensure that for all w, v, the new graph satisfies this.
                # Actually, it's easier to check: does adding (u,v) create a C4?
                # A C4 involving the new edge (u,v) must use (u,v) and a path of length 3 from u to v.
                # Path u-w-x-v.
                # So we need to check if there exists w in N(u) and x in N(v) such that (w,x) is an edge.
                
                # Let's compute common neighbors of u and v in the current graph? No.
                # Let's check if there is a path of length 3.
                # N(u) is a bitset. N(v) is a bitset.
                # We need to know if any edge exists between N(u) and N(v).
                # This is slow to check for every pair.
                
                # Simpler C4-free construction:
                # Iterate u from 0 to n-1.
                # For each v > u, add edge (u,v) if it doesn't create a C4.
                # Condition: |N(u) & N(v)| <= 1 in the graph formed by vertices {0..u-1}?
                # No, the graph is built incrementally.
                # If we add edges in a specific order, we can maintain the invariant.
                
                # Let's use the standard incremental check:
                # When adding edge (u,v), check if there is a vertex w such that (u,w) and (v,w) are edges?
                # If so, u and v share a neighbor w. If they share TWO neighbors, we have a C4.
                # So, before adding (u,v), we must ensure that the number of common neighbors of u and v is <= 1.
                # Wait, if they share 1 neighbor w, adding (u,v) creates a triangle u-v-w-u. That's allowed.
                # If they share 2 neighbors w, x, adding (u,v) creates u-v-w-x-u? No, u-w and v-w and u-x and v-x exist.
                # The cycle u-v-w-x-u requires edges (u,v), (v,w), (w,x), (x,u).
                # We have (u,w), (v,w), (u,x), (v,x).
                # If we add (u,v), do we have a C4?
                # The cycle u-v-w-x-u uses edges (u,v), (v,w), (w,x), (x,u).
                # We don't necessarily have (w,x).
                # However, the condition "No two vertices have two common neighbors" is equivalent to C4-free.
                # If u and v have common neighbors w and x, then w and x are both in N(u) and N(v).
                # This implies the subgraph induced by {u,v,w,x} contains a C4?
                # Not necessarily. It contains a K_{2,2} between {u,v} and {w,x}.
                # A K_{2,2} is a C4.
                # So yes, if |N(u) & N(v)| >= 2, the graph contains a C4.
                
                # So the check is: before adding (u,v), ensure |N(u) & N(v)| <= 1.
                
                common = adj[u] & adj[v]
                if common.bit_count() <= 1:
                    # Safe to add
                    # We add it to maximize density
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Now we have a maximal C4-free graph (given the order).
            # Check if alpha < 11
            if _alpha_less_than(adj, n, 11):
                # Convert bitsets to edge list
                edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
    
    # Fallback: return a known valid graph if we didn't find one in time
    # A simple valid graph: 10 disjoint triangles (n=30, alpha=10)
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def _alpha_less_than(adj, n, target):
    """
    Check if independence number is < target using branch and bound with bitsets.
    adj is a list of integers (bitsets).
    Returns True if alpha < target, False if alpha >= target.
    """
    # We want to find an independent set of size >= target.
    # If we find one, return False.
    # If we exhaust search without finding one, return True.
    
    best_size = 0
    
    # Use a recursive function
    # candidates: bitset of vertices that can still be added
    # current_size: size of current independent set
    
    # Optimization: if current_size + popcount(candidates) <= best_size, prune.
    # But we are looking for size >= target.
    # So if current_size >= target, we found it.
    # If current_size + popcount(candidates) < target, we can't reach target from this branch.
    
    def backtrack(candidates, current_size):
        # If we have reached target, we found an independent set
        if current_size >= target:
            return True
        
        # If we can't reach target even with all remaining candidates
        if current_size + candidates.bit_count() < target:
            return False
        
        # If no candidates, we failed this branch
        if candidates == 0:
            return False
        
        # Pick a vertex
        # Heuristic: pick vertex with highest degree in the induced subgraph on candidates
        # Or just pick the first one for simplicity?
        # Picking a high-degree vertex removes more candidates in the "include" branch, which is good.
        
        # To find a high degree vertex quickly:
        # We can just pick the lowest set bit for now, or iterate.
        # Let's pick the vertex with the most neighbors within candidates.
        
        v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lb = temp & -temp
            idx = lb.bit_length() - 1
            # Degree of idx in candidates
            deg = (adj[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lb
        
        if v == -1:
            return False
            
        # Branch 1: Include v
        # New candidates = candidates without v and without N(v)
        new_candidates = candidates & ~(1 << v) & ~adj[v]
        if backtrack(new_candidates, current_size + 1):
            return True
            
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        if backtrack(new_candidates, current_size):
            return True
            
        return False
    
    initial_candidates = (1 << n) - 1
    return not backtrack(initial_candidates, 0)

if __name__ == "__main__":
    res = construct()
    print(res['n'])
