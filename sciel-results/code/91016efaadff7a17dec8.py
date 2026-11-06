"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    n = 31
    target_edges = 91
    
    # Helper: count 4-cycles in a graph given adjacency matrix
    def count_c4(adj):
        # A 4-cycle is a set of 4 distinct vertices a,b,c,d with edges ab, bc, cd, da
        # We count each 4-cycle once by ordering: a < b, a < c, and a < d
        # Actually, let's use the standard method: for each pair (u,v) that is NOT an edge,
        # count common neighbors. Each 4-cycle contributes to exactly one non-edge pair?
        # No. Let's use: number of C4 = sum over all pairs {u,v} of C(common_neighbors(u,v), 2)
        # But this counts each C4 exactly once? 
        # A C4 is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For pair (u,w), common neighbors are v and x. So C(2,2) = 1.
        # For pair (v,x), common neighbors are u and w. So C(2,2) = 1.
        # So each C4 is counted twice in the sum over ALL pairs.
        # So C4_count = (1/2) * sum_{u<v} C(common(u,v), 2)
        # But this includes pairs (u,v) that ARE edges. If u-v is an edge, common neighbors
        # form triangles with u-v, not 4-cycles. Wait, no.
        # If u and v are adjacent, and w,x are common neighbors, then u-w-v-x-u is a 4-cycle
        # ONLY IF u-x and w-v are edges? No.
        # Let's reconsider. A 4-cycle is a cycle of length 4.
        # The formula sum_{u<v} C(common(u,v), 2) counts the number of "diamonds" or 
        # pairs of triangles sharing an edge? No.
        # Actually, sum_{u<v} C(common(u,v), 2) counts the number of 4-cycles where 
        # u and v are OPPOSITE vertices. Each 4-cycle has exactly 2 pairs of opposite vertices.
        # So total C4 = (1/2) * sum_{u<v} C(common(u,v), 2).
        # This is correct regardless of whether u-v is an edge or not.
        
        # Efficient computation using matrix multiplication:
        # common(u,v) = (A @ A)[u,v] - A[u,v] (if A is 0-1 adjacency, no self-loops)
        # Actually (A^2)[u,v] = number of length-2 paths from u to v.
        # If u != v, this equals common(u,v) + (1 if u~v else 0)? No.
        # (A^2)[u,v] = sum_w A[u,w] A[w,v]. This counts common neighbors w.
        # If u and v are adjacent, w can be u or v? No, A[u,u]=0, A[v,v]=0.
        # So (A^2)[u,v] = number of common neighbors of u and v.
        
        A2 = adj @ adj
        # For u < v, common = A2[u,v]
        # C(common, 2) = common * (common - 1) / 2
        # Sum over u < v
        # We can compute the full sum and divide by 2, but we need to be careful.
        # Let's just sum over upper triangle.
        upper = np.triu_indices(n, k=1)
        commons = A2[upper]
        c4_count = np.sum(commons * (commons - 1)) // 2
        return c4_count
    
    def greedy_c4_free_base():
        """Construct a C4-free graph with 90 edges using greedy randomized search."""
        best_adj = None
        best_c4 = float('inf')
        best_edges = []
        
        # Try multiple seeds to find a good C4-free base
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            # Use different seeds for diversity
            random.seed(attempt * 12345 + 999)
            adj = np.zeros((n, n), dtype=np.int32)
            edges = []
            
            # Greedy: add edges in random order, skip if it creates a C4
            candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if len(edges) >= 90:
                    break
                
                # Check if adding (u,v) creates a C4
                # We need to check if there is a path of length 3 between u and v
                # i.e., exists w, x such that u~w, w~x, x~v
                # w != x, w != v, x != u
                
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                creates_c4 = False
                # Optimization: Check intersection of neighbors of u and neighbors of v? No.
                # We need w in N(u) and x in N(v) such that w~x.
                # Iterate over smaller set?
                if len(neighbors_u) < len(neighbors_v):
                    for w in neighbors_u:
                        if w == v: continue
                        # Check if w is connected to any neighbor of v (excluding u, w)
                        # adj[w] is the row.
                        # We want x in N(v) such that adj[w,x]==1 and x!=u and x!=w.
                        # Since w is in N(u), u is in N(w).
                        # We can check adj[w, neighbors_v]
                        # But we must exclude u and w.
                        # adj[w, u] is 1. adj[w, w] is 0.
                        # So we just check if any x in neighbors_v (x != u) has adj[w,x]==1.
                        # If x == w, adj[w,w]=0.
                        # So:
                        valid_xs = neighbors_v[neighbors_v != u]
                        if len(valid_xs) > 0:
                            if np.any(adj[w, valid_xs] == 1):
                                creates_c4 = True
                                break
                else:
                    for x in neighbors_v:
                        if x == u: continue
                        valid_ws = neighbors_u[neighbors_u != x]
                        if len(valid_ws) > 0:
                            if np.any(adj[x, valid_ws] == 1):
                                creates_c4 = True
                                break
                
                if not creates_c4:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edges.append((u, v))
            
            if len(edges) == 90:
                # It's C4-free by construction, so c4 should be 0
                # But let's verify to be safe
                c4 = count_c4(adj)
                if c4 < best_c4:
                    best_c4 = c4
                    best_adj = adj.copy()
                    best_edges = list(edges)
            
            if best_c4 == 0:
                break
        
        return best_adj, best_edges
    
    # Step 1: Find a C4-free base with 90 edges
    base_adj, base_edges = greedy_c4_free_base()
    
    if base_adj is None:
        # Fallback: use a simpler construction (likely to have C4s, but valid)
        base_adj = np.zeros((n, n), dtype=np.int32)
        base_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if len(base_edges) >= 90:
                    break
                base_adj[i, j] = 1
                base_adj[j, i] = 1
                base_edges.append((i, j))
            if len(base_edges) >= 90:
                break
    
    # Step 2: Add the 91st edge that creates the fewest new 4-cycles
    best_total_c4 = float('inf')
    best_final_edges = None
    
    # Try each possible edge not in base
    existing_edges = set(base_edges)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidate_edges = [e for e in all_possible if e not in existing_edges]
    
    # To speed up, we can estimate the number of new C4s created by adding (u,v)
    # New C4s are formed by paths of length 3 between u and v in the base graph.
    # Number of such paths = sum_{w in N(u), x in N(v)} (1 if w~x else 0) - (1 if u~v else 0)?
    # Actually, we just need to count paths u-w-x-v.
    # This is equivalent to (A^3)[u,v] in the base graph?
    # A^3[u,v] counts walks of length 3.
    # Since the graph is simple, walks of length 3 are paths u-w-x-v where u!=w!=x!=v and u!=x, w!=v?
    # If the graph is C4-free, there are no cycles of length 4.
    # So any walk of length 3 is a simple path?
    # u-w-x-v. If u=x, then u-w-u-v (backtrack).
    # If w=v, u-v-v (invalid).
    # If w=x, u-w-w-v (invalid).
    # So in a simple graph, length 3 walks are either simple paths or backtrackings.
    # Backtracking: u-w-u-v (requires u~w and u~v). But u,v are non-adjacent in base.
    # So u~v is false.
    # Thus, all length 3 walks between non-adjacent u,v are simple paths u-w-x-v.
    # So the number of new C4s created by adding (u,v) is exactly (A_base^3)[u,v].
    
    # Let's compute A_base^3 once.
    A_base = base_adj
    A_base_2 = A_base @ A_base
    A_base_3 = A_base_2 @ A_base
    
    for u, v in candidate_edges:
        if time.time() > deadline - 5:
            break
        
        # Number of new C4s = number of length-3 paths between u and v
        new_c4s = A_base_3[u, v]
        
        # Total C4s = C4s in base (0) + new_c4s
        total_c4 = new_c4s
        
        if total_c4 < best_total_c4:
            best_total_c4 = total_c4
            best_final_edges = base_edges + [(u, v)]
    
    if best_final_edges is None:
        # Fallback: just add the first candidate
        if candidate_edges:
            best_final_edges = base_edges + [candidate_edges[0]]
        else:
            best_final_edges = base_edges
    
    return [[u, v] for u, v in best_final_edges]
# EVOLVE-BLOCK-END
