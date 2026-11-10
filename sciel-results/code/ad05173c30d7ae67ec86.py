import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 16
    
    def check_q3(adj, u, v):
        """
        Check if adding edge (u,v) creates a Q3.
        """
        # u and v are non-adjacent in current adj
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # Q3 vertices: 0,1,2,3,4,5,6,7
        # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        # Let's say edge (u,v) corresponds to (0,1) in Q3
        # Then:
        # 0=u, 1=v
        # Neighbors of 0 in Q3: 1, 2, 4. So 2, 4 are neighbors of u (but not v)
        # Neighbors of 1 in Q3: 0, 3, 5. So 3, 5 are neighbors of v (but not u)
        # 2 and 4 are not adjacent. 3 and 5 are not adjacent.
        # 3 is adjacent to 2 and 7.
        # 5 is adjacent to 4 and 7.
        # 6 is adjacent to 4 and 7.
        # 2 and 6 are not adjacent? In Q3, 2=(100), 6=(110). They differ in bit 1. So NOT adjacent.
        # Wait. Q3 edges connect vertices differing in exactly 1 bit.
        # 2=(100), 6=(110). Differ in bit 1 (value 2). So 2 and 6 ARE adjacent? No.
        # 100 and 110 differ in the 2's place. So they are adjacent?
        # Binary: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111.
        # Edge (2,6): 010 and 110. Differ in bit 2 (value 4). So they are adjacent.
        # My previous listing of edges:
        # (0,1) 000-001
        # (0,2) 000-010
        # (0,4) 000-100
        # (1,3) 001-011
        # (1,5) 001-101
        # (2,3) 010-011
        # (2,6) 010-110
        # (3,7) 011-111
        # (4,5) 100-101
        # (4,6) 100-110
        # (5,7) 101-111
        # (6,7) 110-111
        # Total 12 edges. Correct.
        
        # So:
        # u=0, v=1
        # Neighbors of u in Q3 (excluding v): 2, 4
        # Neighbors of v in Q3 (excluding u): 3, 5
        # Let a2, a4 be neighbors of u. (2,4)
        # Let a3, a5 be neighbors of v. (3,5)
        # a2 must be adjacent to a3 and a6 (where a6 is 6).
        # a4 must be adjacent to a5 and a6.
        # a3 must be adjacent to a7.
        # a5 must be adjacent to a7.
        # a6 must be adjacent to a7.
        # a2, a4 not adjacent.
        # a3, a5 not adjacent.
        # a2, a5 not adjacent? 2=(010), 5=(101). Differ in 3 bits. Not adjacent.
        # a4, a3 not adjacent? 4=(100), 3=(011). Differ in 3 bits. Not adjacent.
        
        # Algorithm:
        # 1. Find candidates for a2, a4 from neighbors of u.
        # 2. For each pair (a2, a4) that are NOT adjacent:
        #    3. Find candidates for a3 from neighbors of v AND neighbors of a2.
        #    4. Find candidates for a5 from neighbors of v AND neighbors of a4.
        #    5. For each pair (a3, a5) that are NOT adjacent and distinct from a2,a4:
        #       6. Find candidates for a7 from neighbors of a3 AND neighbors of a5.
        #       7. Find candidates for a6 from neighbors of a2 AND neighbors of a4.
        #       8. Check if there is an a7 in cand_a7 and a6 in cand_a6 such that a6 and a7 are adjacent.
        
        # Optimization:
        # Precompute intersection of neighbors.
        
        # a2, a4 must be in N(u) \ {v}
        # a2, a4 not adjacent.
        
        c_u = neigh_u
        c_v = neigh_v
        
        # Iterate a2 in c_u
        a2_bits = c_u
        while a2_bits:
            a2_bit = a2_bits & -a2_bits
            a2 = a2_bit.bit_length() - 1
            a2_bits -= a2_bit
            
            # a4 in c_u, a4 != a2, not adjacent to a2
            a4_bits = c_u & ~(1 << a2) & ~adj[a2]
            while a4_bits:
                a4_bit = a4_bits & -a4_bits
                a4 = a4_bit.bit_length() - 1
                a4_bits -= a4_bit
                
                # a3 in N(v) \ {u, a2, a4} AND N(a2)
                a3_cand = c_v & ~(1 << a2) & ~(1 << a4) & adj[a2]
                # a5 in N(v) \ {u, a2, a4} AND N(a4)
                a5_cand = c_v & ~(1 << a2) & ~(1 << a4) & adj[a4]
                
                if not a3_cand or not a5_cand:
                    continue
                
                a3_bits = a3_cand
                while a3_bits:
                    a3_bit = a3_bits & -a3_bits
                    a3 = a3_bit.bit_length() - 1
                    a3_bits -= a3_bit
                    
                    # a5 != a3
                    a5_check = a5_cand & ~(1 << a3)
                    if not a5_check:
                        continue
                    
                    a5_bits = a5_check
                    while a5_bits:
                        a5_bit = a5_bits & -a5_bits
                        a5 = a5_bit.bit_length() - 1
                        a5_bits -= a5_bit
                        
                        # a7 in N(a3) \ {u,v,a2,a4,a5} AND N(a5)
                        used_mask = (1 << u) | (1 << v) | (1 << a2) | (1 << a4) | (1 << a3) | (1 << a5)
                        a7_cand = (adj[a3] & adj[a5]) & ~used_mask
                        
                        # a6 in N(a2) \ {u,v,a3,a4,a5} AND N(a4)
                        a6_cand = (adj[a2] & adj[a4]) & ~((1 << u) | (1 << v) | (1 << a3) | (1 << a4) | (1 << a5))
                        
                        if not a7_cand or not a6_cand:
                            continue
                        
                        # Check if any a6 in a6_cand is adjacent to any a7 in a7_cand
                        a6_bits = a6_cand
                        while a6_bits:
                            a6_bit = a6_bits & -a6_bits
                            a6 = a6_bit.bit_length() - 1
                            a6_bits -= a6_bit
                            
                            if adj[a6] & a7_cand:
                                return True
        return False

    def is_q3_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    if check_q3(adj, u, v):
                        return False
        return True

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def edges_from_adj(adj):
        res = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    res.append([u, v])
        return res

    # Strategy: Greedy Construction + Local Search (1-flip and 2-flip)
    # Start with empty graph. Add edges if no Q3.
    # Then try to add more.
    # Then try swaps: remove edge e1, add edge e2.
    
    def greedy_build(adj):
        # Try to add every missing edge
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        # Try adding (u,v)
                        if not check_q3(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
        return adj

    def local_search(adj, deadline):
        # 1-flip: remove edge
        # 2-flip: remove e1, add e2
        # We want to maximize edges.
        # Start: Greedy add all possible.
        adj = greedy_build(adj)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try 2-flip: remove e1, add e2
            # Iterate over all pairs of edges (e1, e2) where e1 is present, e2 is absent.
            # This is O(E * (E-E)) which is 120*120 = 14400 checks.
            # Each check is fast.
            
            # To avoid missing improvements, we can do:
            # For each edge e1 present:
            #   Remove e1
            #   For each edge e2 absent:
            #      Try add e2
            #      If valid: accept (net change 0 edges? No, we removed 1 added 1. Same count.)
            #      Wait, we want to INCREASE edges.
            #      If we remove e1 and add e2, count is same.
            #      We only accept if it allows us to add MORE edges afterwards?
            #      That's complex.
            
            # Better: Just try to add any absent edge.
            # If we get stuck, we are at a local maximum where no single edge can be added.
            # Can we swap?
            # If we swap, we might unblock an edge addition.
            
            # Let's try:
            # 1. Try to add all absent edges. (Done in greedy_build)
            # 2. If no edges added, try swaps.
            
            # For each present edge e1:
            #   Temporarily remove e1.
            #   Try to add all absent edges (including e1? No, e1 is removed).
            #   If we can add e1 back OR add a different edge e2, and the resulting graph has >= edges as before:
            #      If we add e2 (and not e1), we have same count but different structure.
            #      If we can add e2 AND e1, we have +1 edge. (Accept immediately)
            #      If we can only add e2, we have same count. Accept if it allows further improvements?
            #      This is "kick".
            
            # Let's implement a simple kick:
            # Pick a random present edge e1 to remove.
            # Try to greedily add edges.
            # If we can add e1 back, plus some others, great.
            # If we can't add e1 back, but added others, maybe good.
            
            # For now, let's just do:
            # Try to add all absent edges.
            if True:
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            if not check_q3(adj, u, v):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                improved = True
                if improved:
                    continue
            
            # If no improvement, try a kick.
            # Remove a random edge.
            edges_present = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u] & (1 << v):
                        edges_present.append((u,v))
            
            if not edges_present:
                break
            
            # Try up to 5 random kicks
            for _ in range(5):
                u1, v1 = random.choice(edges_present)
                # Remove
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Greedy add
                adj = greedy_build(adj)
                
                # If we added more than 1 edge (net gain), or same count (swap), accept?
                # Let's count edges.
                # Actually greedy_build returns the graph.
                # We lost (u1,v1) but maybe gained others.
                # Let's just accept if valid. It might open up paths.
                # To be safe, only accept if edge count didn't decrease?
                # Or just accept and hope?
                # Let's accept if edge count >= previous count - 1?
                # No, let's just run greedy_build again on the result.
                
                # Check if we can re-add (u1,v1)
                # Doesn't matter. We just keep the result.
                
                # If this kick led to a dead end, we might want to revert.
                # But without backtracking, we might get stuck in bad local minima.
                # Let's stick to "improve only" for now.
                # If greedy_build(adj) has more edges than before kick, accept.
                # Else revert?
                
                # Let's implement:
                # prev_count = count_edges(adj_before_kick)
                # new_count = count_edges(adj_after_kick)
                # if new_count > prev_count: accept
                # else: revert
                
                # But we already mutated adj.
                # Let's save copy.
                
                # Optimization: Don't save full copy. Just count.
                
                # Let's refine:
                # We will implement a proper "best improvement" local search later.
                # For now, this loop might not improve much.
                
                # Let's try a different structure:
                # 1. Greedy
                # 2. While improved:
                #      Try all swaps (remove e1, add e2).
                #      If a swap increases edge count, do it.
                #      If no swap increases, try a random kick (remove e1), greedy fill, if result better, accept.
                
                break # Break kick loop for now, structure needs work
        
        return adj

    # Main
    best_adj = [0] * n
    best_score = 0
    
    # Multiple restarts
    while time.time() < deadline - 10:
        adj = [0] * n
        # Random initial edges?
        # Or empty.
        
        # Greedy
        adj = greedy_build(adj)
        
        # Local Search
        adj = local_search(adj, deadline)
        
        score = count_edges(adj)
        if score > best_score:
            best_score = score
            best_adj = [m for m in adj]
        
        # If we reached a high score, break?
        # ex(16, Q3) is likely around 60-70.
        # Let's see.
        if best_score > 60:
            # Try to polish best_adj
            pass

    # Final Polish on best_adj
    best_adj = greedy_build(best_adj)
    
    if not is_q3_free(best_adj):
        # Fallback
        return []
        
    return edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
