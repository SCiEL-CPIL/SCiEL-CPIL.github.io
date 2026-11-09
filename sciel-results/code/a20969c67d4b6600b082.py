"""
Strategy: Simulated Annealing with Incremental C4 Delta Calculation.
We start with a random 91-edge graph and perform edge swaps (remove one, add one)
to minimize the number of C4s.
The key to speed is calculating the change in C4 count (delta) in O(n) time per swap
candidate, rather than O(n^3) or O(n^4).

C4 count formula:
Total C4s = sum_{u < v} C(common(u, v), 2)
where common(u, v) is the number of common neighbors of u and v.

When we swap edge (a, b) for (c, d):
1. Remove (a, b):
   - For all w != a, b: common(a, w) decreases by 1 if b is a common neighbor? No.
   - common(a, w) is the number of neighbors of a that are also neighbors of w.
   - Since b is a neighbor of a, b contributes to common(a, w) iff b is a neighbor of w.
   - So for each w in N(b) \ {a}, common(a, w) decreases by 1.
   - Similarly for each w in N(a) \ {b}, common(b, w) decreases by 1.
   - Also, common(a, b) is not defined in the sum (since a, b are adjacent, they are not in the sum if we sum over non-edges?
     Wait, the formula sum_{u<v} C(common(u,v), 2) counts C4s for ALL pairs u, v?
     Let's check.
     A C4 is u-v-w-x-u.
     The pairs of opposite vertices are (u, w) and (v, x).
     u and w are not adjacent in the C4 (unless there are chords, but we count subgraphs, so chords don't matter for the existence of the cycle, but they affect common neighbors).
     Actually, the standard formula for the number of C4s in a graph G is:
     N_C4 = 1/4 * sum_{u, v} C(common(u, v), 2)
     This sum is over all ordered pairs (u, v) with u != v? Or all u, v?
     If u, v are adjacent, can they be opposite in a C4? No, opposite vertices in a C4 are not adjacent in the cycle, but they might be adjacent in the graph (chord).
     However, the formula 1/4 sum_{u,v} C(common(u,v), 2) counts each C4 exactly 4 times (once for each pair of opposite vertices).
     Does it require u, v to be non-adjacent? No.
     If u, v are adjacent and have 2 common neighbors w, x, then u-w-v-x-u is a C4?
     Edges: (u, w), (w, v), (v, x), (x, u). Yes, this is a C4.
     So the formula works for all pairs u, v.
     
     So, when we remove edge (a, b):
     - common(a, w) decreases by 1 for all w in N(b) \ {a}.
     - common(b, w) decreases by 1 for all w in N(a) \ {b}.
     - Note: common(a, b) is not in the sum? No, the sum is over all u, v.
     - But wait, if we remove (a, b), does common(a, b) change?
       common(a, b) is the number of common neighbors. It does not depend on the edge (a, b) itself.
       So common(a, b) is unchanged by removing (a, b).
       However, the term C(common(a, b), 2) in the sum is for the pair (a, b).
       Since (a, b) is an edge, it is included in the sum.
       
     Let's refine the delta calculation.
     Let S = sum_{u, v} C(common(u, v), 2).
     Change in S when removing (a, b):
     The common neighbor counts that change are those involving a or b.
     Specifically:
     For any w:
     - If w is a neighbor of b (and w != a), then common(a, w) decreases by 1 (because b was a common neighbor, but now a is not adjacent to b, so b is no longer a neighbor of a? No, b IS a neighbor of a in the old graph. After removal, b is NOT a neighbor of a. So b is no longer a common neighbor of a and w. Correct.)
     - If w is a neighbor of a (and w != b), then common(b, w) decreases by 1.
     
     So the change in S is:
     Delta_remove = - [ sum_{w in N(b)\{a}} (C(c_a_w, 2) - C(c_a_w - 1, 2)) + sum_{w in N(a)\{b}} (C(c_b_w, 2) - C(c_b_w - 1, 2)) ]
     where c_a_w is the old common(a, w).
     Note: C(k, 2) - C(k-1, 2) = k - 1.
     So Delta_remove = - [ sum_{w in N(b)\{a}} (common(a, w) - 1) + sum_{w in N(a)\{b}} (common(b, w) - 1) ]
     
     Similarly, when adding (c, d):
     Delta_add = + [ sum_{w in N(d)\{c}} (common(c, w) - 1) + sum_{w in N(c)\{d}} (common(d, w) - 1) ]
     BUT, we must be careful. The common counts used in Delta_add should be the ones AFTER the removal of (a, b).
     If (c, d) is disjoint from (a, b) and their neighborhoods, the common counts are unchanged.
     If they overlap, it's complicated.
     
     To simplify, we can restrict swaps to be "local" or just recompute the relevant common counts.
     Or, we can use the fact that n=31 is small.
     We can maintain the common neighbor matrix C.
     C[u, v] = common(u, v).
     
     When removing (a, b):
     1. For each w in N(b) \ {a}:
        C[a, w] -= 1
        C[w, a] -= 1
        Update S: S -= (C[a, w]_old - 1)  [since C(k,2)-C(k-1,2) = k-1, and k is the old value]
        Wait, S contains C(C[a, w], 2).
        New term is C(C[a, w]_new, 2) = C(C[a, w]_old - 1, 2).
        Difference = C(k, 2) - C(k-1, 2) = k - 1.
        So S decreases by (k - 1).
        
     2. For each w in N(a) \ {b}:
        C[b, w] -= 1
        C[w, b] -= 1
        S decreases by (C[b, w]_old - 1).
        
     3. What about C[a, b]?
        C[a, b] is the number of common neighbors. It doesn't change when we remove (a, b).
        So the term C(C[a, b], 2) in S remains the same.
        
     So total change in S for removal:
     Delta_S_remove = - sum_{w in N(b)\{a}} (C[a, w]_old - 1) - sum_{w in N(a)\{b}} (C[b, w]_old - 1)
     
     When adding (c, d) (after removal):
     The common counts C have been updated.
     1. For each w in N(d)\{c} (in the graph AFTER removal):
        C[c, w] += 1
        C[w, c] += 1
        S increases by (C[c, w]_current - 1)  [where current is BEFORE the increment]
        Wait. New term C(k+1, 2). Old term C(k, 2).
        Difference = C(k+1, 2) - C(k, 2) = k.
        So S increases by k, where k is the value of C[c, w] BEFORE adding the edge.
        
     2. For each w in N(c)\{d}:
        C[d, w] += 1
        C[w, d] += 1
        S increases by C[d, w]_current.
        
     So Delta_S_add = sum_{w in N(d)\{c}} C[c, w]_current + sum_{w in N(c)\{d}} C[d, w]_current
     
     Total Delta_S = Delta_S_remove + Delta_S_add.
     Note: If the sets of vertices affected by removal and addition overlap, we must be careful.
     Specifically, if a is in N(d) or N(c), etc.
     Given n=31, we can just implement this carefully.
     
     However, there is a simpler approach for n=31.
     We can just recompute the C4 count for a candidate swap in O(n^2) or less?
     No, O(n^3) is 31^3 = 29791.
     If we do 10,000 swaps, that's 300 million ops. In Python/NumPy, this might be slow.
     But the incremental method is O(n) per swap. 31 * 10,000 = 310,000 ops. Very fast.
     
     Let's implement the incremental method.
"""

import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140  # Safety margin
    
    rng = random.Random(123)
    
    def count_c4(adj):
        """
        Count C4s using the formula 1/4 * sum_{u,v} C(common(u,v), 2).
        adj is a 0/1 matrix.
        """
        A = adj.astype(np.int32)
        C = A @ A
        # Sum C(C[u,v], 2) for all u, v
        # C(C[u,v], 2) = C[u,v] * (C[u,v] - 1) // 2
        total = np.sum(C * (C - 1)) // 2
        return total // 4

    def get_common_matrix(adj):
        A = adj.astype(np.int32)
        return A @ A

    # Initialize with a random graph
    # Or better, start with a sparse random graph and add edges?
    # Let's start with a random 91-edge graph.
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_adj = None
    best_c4 = float('inf')
    
    # Run multiple restarts or one long SA
    # Let's do one long SA with random restarts if stuck.
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        edges = rng.sample(all_pairs, m)
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        C = get_common_matrix(adj)
        current_c4 = count_c4(adj)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
        
        # Simulated Annealing
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        step = 0
        while T > T_min and time.time() < deadline - 5:
            step += 1
            
            # Pick a random edge to remove
            # We need a list of current edges
            # Maintaining a list of edges is O(1) for sampling if we use a set or list.
            # Let's just sample a random edge from the adjacency matrix.
            # Or maintain a list.
            # For simplicity and speed, let's maintain a list of edges.
            
            # Actually, sampling a random edge from a 31x31 matrix is fast.
            while True:
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u < v and adj[u, v] == 1:
                    break
                if v < u and adj[v, u] == 1:
                    u, v = v, u
                    break
            
            # Pick a random non-edge to add
            while True:
                c = rng.randint(0, n - 1)
                d = rng.randint(0, n - 1)
                if c < d and adj[c, d] == 0:
                    break
                if d < c and adj[d, c] == 0:
                    c, d = d, c
                    break
            
            # Calculate Delta
            # Remove (u, v)
            # Affected vertices: N(v)\{u} and N(u)\{v}
            
            # We need the current common counts C.
            # Note: C[u, w] is the number of common neighbors of u and w.
            
            # 1. Delta from removing (u, v)
            delta_remove = 0
            # For w in N(v) \ {u}:
            # C[u, w] decreases by 1.
            # S decreases by (C[u, w]_old - 1)
            for w in range(n):
                if w != u and adj[v, w] == 1:
                    delta_remove -= (C[u, w] - 1)
            
            # For w in N(u) \ {v}:
            # C[v, w] decreases by 1.
            # S decreases by (C[v, w]_old - 1)
            for w in range(n):
                if w != v and adj[u, w] == 1:
                    delta_remove -= (C[v, w] - 1)
            
            # 2. Delta from adding (c, d)
            # We must use the updated C values?
            # The formula for Delta_add uses C[c, w] and C[d, w] AFTER the removal.
            # So we need to update C first, or calculate the effect of removal on C[c, w] and C[d, w].
            
            # Let's update C temporarily to get the correct values for Delta_add.
            # This is O(n) per swap.
            
            # Save old values if we need to revert?
            # We can just calculate the new C values for the affected entries.
            
            # Affected entries in C due to removal of (u, v):
            # C[u, w] for w in N(v)\{u}
            # C[v, w] for w in N(u)\{v}
            
            # Let's create a dictionary of changes or just a list of (i, j, delta)
            changes = []
            
            for w in range(n):
                if w != u and adj[v, w] == 1:
                    changes.append((u, w, -1))
                    changes.append((w, u, -1))
                if w != v and adj[u, w] == 1:
                    changes.append((v, w, -1))
                    changes.append((w, v, -1))
            
            # Apply changes to C (temporarily)
            for i, j, d in changes:
                C[i, j] += d
            
            # Now calculate Delta_add
            delta_add = 0
            # For w in N(d) \ {c} (in the graph with (u, v) removed):
            # Note: The graph structure for neighbors is the same, except (u, v) is gone.
            # So N(d) is the same, unless d is u or v?
            # If d == u, then N(d) changes? No, N(d) is the set of neighbors.
            # Removing (u, v) changes N(u) and N(v).
            # If d == u, then N(d) is N(u) without v.
            # If d == v, then N(d) is N(v) without u.
            # If d is neither, N(d) is unchanged.
            
            # Let's handle the neighbor sets carefully.
            
            # Define a function to get neighbors, considering the removal of (u, v)
            def get_neighbors(x, exclude_edge_u, exclude_edge_v):
                neighbors = []
                for w in range(n):
                    if adj[x, w] == 1:
                        # If (x, w) is the removed edge, skip
                        if (x == exclude_edge_u and w == exclude_edge_v) or (x == exclude_edge_v and w == exclude_edge_u):
                            continue
                        neighbors.append(w)
                return neighbors
            
            N_d = get_neighbors(d, u, v)
            N_c = get_neighbors(c, u, v)
            
            for w in N_d:
                if w != c:
                    delta_add += C[c, w]  # C[c, w] is the value AFTER removal
            
            for w in N_c:
                if w != d:
                    delta_add += C[d, w]  # C[d, w] is the value AFTER removal
            
            total_delta = delta_remove + delta_add
            
            # Acceptance
            if total_delta < 0 or rng.random() < np.exp(-total_delta / T):
                # Accept
                # Apply changes to adj
                adj[u, v] = 0
                adj[v, u] = 0
                adj[c, d] = 1
                adj[d, c] = 1
                
                # C is already updated for removal.
                # Now update C for addition of (c, d).
                # For w in N(d) \ {c}: C[c, w] += 1
                # For w in N(c) \ {d}: C[d, w] += 1
                
                for w in N_d:
                    if w != c:
                        C[c, w] += 1
                        C[w, c] += 1
                for w in N_c:
                    if w != d:
                        C[d, w] += 1
                        C[w, d] += 1
                
                current_c4 += total_delta
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                # Reject
                # Revert C changes
                for i, j, d in changes:
                    C[i, j] -= d
            
            T *= cooling
        
        # If we found 0, we can stop?
        # ex(31, C4) = 90, so 91 edges must have at least 1 C4.
        # So 0 is impossible.
        if best_c4 <= 1:
            break

    # Extract edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges

if __name__ == "__main__":
    edges = construct()
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A = adj.astype(np.int32)
    C = A @ A
    total = np.sum(C * (C - 1)) // 2
    print(f"C4: {total // 4}")
