import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    # M[u] is a bitmask of neighbors of u
    M = [0] * n
    
    def get_neighbors(u):
        return [i for i in range(n) if (M[u] >> i) & 1]

    def count_violations_for_edge(u, v):
        """
        Counts the number of K3,3 subgraphs containing edge (u, v).
        A K3,3 containing (u,v) requires:
        Partitions X={u, a, b}, Y={v, c, d}
        such that all edges between X and Y exist.
        This implies a, b in N(u) and a, b in N(v) => a, b in N(u)∩N(v).
        c, d in N(v) and c, d in N(u) => c, d in N(u)∩N(v).
        And the edges between {a,b} and {c,d} must exist.
        """
        common_mask = M[u] & M[v]
        # Exclude u and v themselves if they are in the mask (they shouldn't be self-loops)
        common_mask &= ~(1 << u)
        common_mask &= ~(1 << v)
        
        common = [i for i in range(n) if (common_mask >> i) & 1]
        
        if len(common) < 4:
            return 0
            
        count = 0
        # We need to choose 2 vertices for X\{u} (call them a,b) and 2 for Y\{v} (call them c,d)
        # a,b,c,d must be distinct? 
        # X and Y are disjoint sets in a bipartite graph. 
        # So {a,b} and {c,d} must be disjoint.
        # Also a,b,c,d must be distinct from u,v (already handled).
        
        # Iterate over all pairs (a,b) from common
        # Then iterate over all pairs (c,d) from common \ {a,b}
        # Check if K2,2 exists between {a,b} and {c,d}
        
        # Optimization: Precompute adjacency between common vertices?
        # Or just check bitmasks.
        
        pairs_ab = list(combinations(common, 2))
        for a, b in pairs_ab:
            # Mask of a and b
            mask_ab = (1 << a) | (1 << b)
            # Candidates for c,d are common vertices not in {a,b}
            candidates_mask = common_mask & ~mask_ab
            candidates = [i for i in range(n) if (candidates_mask >> i) & 1]
            
            if len(candidates) < 2:
                continue
                
            # Check if a and b are connected to all potential c,d
            # Actually, we just need to find if there exists a pair (c,d) in candidates
            # such that a-c, a-d, b-c, b-d exist.
            
            # Let's iterate pairs of candidates
            for c, d in combinations(candidates, 2):
                # Check edges a-c, a-d, b-c, b-d
                if ((M[a] >> c) & 1) and ((M[a] >> d) & 1) and ((M[b] >> c) & 1) and ((M[b] >> d) & 1):
                    count += 1
        return count

    def remove_edge(u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)

    def add_edge(u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)

    def count_edges():
        return sum(M[i].bit_count() for i in range(n)) // 2

    def is_valid():
        # Check if graph is K3,3 free
        # We can use the same logic: if any edge has violation count > 0, it's invalid
        # But for speed, we can just check if total violations is 0
        # However, count_violations_for_edge is O(|K|^4) roughly.
        # For n=20, max degree 19, |K| <= 18. 18C2 * 16C2 ~ 153 * 120 ~ 18000 ops per edge.
        # 190 edges * 18000 = 3.4M ops. Might be slow in Python loop.
        # Let's use a faster global check or just rely on the removal process.
        # Actually, if we remove edges based on count, we stop when all counts are 0.
        pass

    def break_down(density=0.8):
        """
        Start with a random dense graph and remove edges until K3,3-free.
        """
        # Initialize
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    add_edge(u, v)
        
        # Iteratively remove edges with highest violation count
        # We need to recompute counts after each removal
        # To speed up, we can batch remove if possible, but sequential is safer for correctness
        
        # Precompute list of edges
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (M[u] >> v) & 1]
        
        # We will maintain a list of (count, u, v)
        # But counts change after removal.
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Find edge with max violation count
            max_count = 0
            max_edge = None
            
            # Sample a subset of edges to find a good candidate to speed up
            # Or just check all? 190 edges is small.
            current_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (M[u] >> v) & 1]
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                c = count_violations_for_edge(u, v)
                if c > max_count:
                    max_count = c
                    max_edge = (u, v)
                    
            if max_count == 0:
                break
                
            remove_edge(max_edge[0], max_edge[1])
            
        return count_edges()

    def local_search():
        """
        Try to add edges or swap to increase count.
        """
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try adding any missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((M[u] >> v) & 1)]
            random.shuffle(missing)
            
            # To check if adding (u,v) creates a K3,3, we just need to check if 
            # count_violations_for_edge(u,v) > 0 in the graph WITH the edge added.
            # But adding the edge changes the common neighbors for other edges too?
            # No, adding (u,v) only affects K3,3s containing (u,v).
            # Wait, adding (u,v) might create new K3,3s that do NOT contain (u,v)?
            # No. A K3,3 is a subgraph. If it didn't exist before, and we add one edge,
            # the new K3,3 must contain that edge.
            # So we just need to check if adding (u,v) creates any K3,3 containing (u,v).
            
            for u, v in missing:
                # Temporarily add
                add_edge(u, v)
                # Check if this creates a violation
                c = count_violations_for_edge(u, v)
                if c == 0:
                    # Valid addition
                    improved = True
                    break
                else:
                    remove_edge(u, v)
            
            if improved:
                continue

            # 2. Try swaps: Remove (a,b), add (c,d)
            # This is more complex.
            # Let's stick to single additions for now, maybe add 2-opt later if time permits.
            # Given the time, single addition from a good seed is likely sufficient.
            
            # If no single edge can be added, try a random restart of local search?
            # Or just break.
            break
            
        return count_edges()

    best_edges = 0
    best_M = [0] * n
    
    random.seed(42)
    
    # Run multiple restarts with different densities
    densities = [0.7, 0.8, 0.9, 0.95]
    
    for density in densities:
        if time.time() > deadline - 10:
            break
            
        # Reset graph
        M = [0] * n
        
        # Break-down phase
        break_down(density)
        
        # Local search phase
        local_search()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
            
    # Final polish: try to add more edges to the best solution
    if time.time() < deadline - 5:
        M = best_M[:]
        local_search()
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]

    # Construct output
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges

# Quick test to ensure no exceptions
if __name__ == "__main__":
    res = construct()
    print(len(res))
