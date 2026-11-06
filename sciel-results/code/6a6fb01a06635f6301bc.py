import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # 1. Generate Candidates
    # Candidates must share a factor with 255 (3, 5, 17)
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
            
    n = len(candidates)
    # n should be 127
    
    # 2. Build Adjacency Bitmasks
    # adj[i] is a bitmask where bit j is 1 if candidates[i] and candidates[j] are connected
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Maximum Clique Search (Tomita's Algorithm)
    # We use a recursive branch and bound with coloring.
    
    best_clique = []
    
    def expand(clique, color_class):
        # color_class is a list of (vertex, color)
        # We maintain the invariant that vertices in color_class are candidates for addition
        # and are colored such that color(v) is an upper bound on the clique size extendable from v
        
        if not color_class:
            return
        
        # If current clique size + max color > best size, we might improve
        # Actually, the standard Tomita algorithm handles the bound inside the recursion.
        
        # Pick a vertex v from the last color class (highest color)
        # To simplify, we can just pick the last vertex in the list, but Tomita's picks strategically.
        # A simpler version:
        # Let's use a standard Bron-Kerbosch with pivoting and bitmasks, which is often faster to implement correctly and sufficient for N=127.
        
        pass

    # Let's implement a robust Bron-Kerbosch with pivoting using bitmasks.
    
    best_size = 0
    best_mask = 0
    
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0 and X_mask == 0:
            if bin(R_mask).count('1') > best_size:
                best_size = bin(R_mask).count('1')
                best_mask = R_mask
            return
            
        # Choose pivot u from P_mask U X_mask to minimize branching
        # Heuristic: pick vertex with max degree in P_mask
        # For simplicity and speed in Python, we can iterate through vertices in P_mask
        # A better pivot: pick u in P_mask | X_mask with max popcount(adj[u] & P_mask)
        
        # To keep it simple and fast enough for 127 nodes, we can just iterate through P_mask.
        # However, pivoting is crucial for speed.
        
        # Let's find a pivot u in P_mask | X_mask
        # We want to maximize |adj[u] & P_mask|
        # We'll just pick the first one for now? No, that's slow.
        # Let's pick the vertex in P_mask with the highest degree in P_mask.
        
        # Optimization: Precompute degrees? Or just calculate on fly.
        # With 127 nodes, calculating on fly is fine.
        
        # To avoid O(N^2) overhead in pivot selection at every node, we can use a simpler heuristic or just iterate.
        # Given the density of the graph (many multiples share factors), the graph is likely dense.
        # Dense graphs are easy for clique finding.
        
        # Let's try a simple recursive search with bitmasks.
        
        # Select a pivot u
        # If P_mask is empty, we handle above.
        # We iterate through vertices in P_mask.
        # To speed up, we can pick u = argmax_{u in P_mask | X_mask} popcount(adj[u] & P_mask)
        
        # Let's implement the pivot selection properly.
        # Candidates for pivot: P_mask | X_mask
        # But we only recurse on vertices in P_mask \ adj[u]
        
        # Simplified approach for this specific small N:
        # Just iterate through all v in P_mask.
        # If the graph is dense, the tree is shallow.
        
        # Let's try to be a bit smarter.
        # Pick u in P_mask | X_mask with max common neighbors in P_mask.
        
        # For performance in Python, avoiding complex pivot selection logic might be better if the graph is dense enough.
        # Let's just iterate through P_mask.
        
        # To ensure we don't timeout, we add a simple check.
        
        # Iterate through vertices in P_mask
        # We need to iterate through bits set in P_mask
        
        # Standard trick:
        # while P_mask:
        #    v = lowest set bit
        #    P_mask ^= (1<<v)
        #    ...
        
        # But we need to pass P_mask & adj[v] to the next level.
        
        # Let's use the pivot to reduce branching.
        # u = select_pivot(P_mask | X_mask, P_mask)
        # candidates = P_mask & ~adj[u]
        # if candidates == 0: return (because all neighbors of u are in P, so any clique must contain u, which is handled by other branches? No, if P \ adj[u] is empty, we can return early? No, that's for the case where we force u. Standard BK with pivot:
        # Recurse on R U {v} for v in P \ adj[u]
        # If P \ adj[u] is empty, we still need to check if R is a clique? No, the base case P=0 handles that.
        # Wait, if P \ adj[u] is empty, it means every vertex in P is adjacent to u.
        # Then any clique in P can be extended by u?
        # Actually, if P \ adj[u] is empty, we can just return? No.
        # The theorem says: if P \ adj[u] is empty, then R is a maximal clique?
        # Yes, if P \ adj[u] is empty, then all vertices in P are neighbors of u.
        # But we don't necessarily have to include u.
        # However, Bron-Kerbosch with pivot says:
        # If P \ adj[u] is empty, we can stop?
        # Let's check. If P \ adj[u] is empty, then for any v in P, v is in adj[u].
        # This doesn't mean R is maximal.
        # Actually, the standard optimization is:
        # Let U = P \ adj[u]
        # For each v in U:
        #    BK(R U {v}, P & adj[v], X & adj[v])
        #    P = P \ {v}
        #    X = X U {v}
        
        # If U is empty, the loop doesn't run. Is that correct?
        # Yes, if U is empty, it implies that every vertex in P is adjacent to u.
        # This doesn't prevent a larger clique from existing in R U P.
        # Wait, if U is empty, we don't recurse. Do we miss a clique?
        # No, because any clique in R U P must contain a vertex from P.
        # If all vertices in P are adjacent to u, does that help?
        # Actually, the pivot optimization relies on the fact that if P \ adj[u] is empty, we can return immediately?
        # Let's verify.
        # If P \ adj[u] is empty, then P subset of adj[u].
        # This means u is connected to all candidates.
        # It doesn't mean we can't form a clique.
        # However, if we don't recurse, we might miss the clique R U P?
        # No, the base case `if P_mask == 0` catches the maximal cliques.
        # If we skip the loop, we just return.
        # This implies that R is a maximal clique?
        # Not necessarily.
        # Let's stick to the standard algorithm without the "early return if U empty" optimization unless I'm sure.
        # Actually, if U is empty, the loop `for v in U` is empty, so we return.
        # This is correct because the algorithm is designed to enumerate all maximal cliques.
        # If P \ adj[u] is empty, it turns out that no new maximal clique can be found by adding vertices from P to R?
        # I will assume the standard implementation is correct.
        
        pass

    # I will implement the Tomita Color Sort algorithm as it is more robust for finding the MAXIMUM clique (not just enumerating all).
    
    # Tomita's Algorithm:
    # 1. Color sort the vertices in P.
    # 2. Recurse on vertices in reverse color order.
    # 3. Use the color as a bound.
    
    # This is complex to implement correctly from scratch in a short time.
    # Given N=127, a simple Bron-Kerbosch with bitmasks and a good pivot might be fast enough.
    # Or even simpler: Greedy + Local Search?
    # No, we want the exact maximum.
    
    # Let's try a different angle.
    # The set of multiples of 3 has size 85.
    # Can we get bigger?
    # If we take all multiples of 3, size 85.
    # If we take all multiples of 5, size 51.
    # If we take all multiples of 17, size 15.
    
    # What if we take $P_3 \cup P_5$?
    # Elements in $P_3 \setminus P_5$ are not connected to elements in $P_5 \setminus P_3$ unless they share a factor.
    # Example: 3 (in $P_3 \setminus P_5$) and 5 (in $P_5 \setminus P_3$). gcd(3,5)=1. Not connected.
    # So $P_3 \cup P_5$ is NOT a clique.
    
    # We need to pick a subset of $P_3$ and a subset of $P_5$ such that every element in $S_3 \subset P_3$ is connected to every element in $S_5 \subset P_5$.
    # This means for all $a \in S_3, b \in S_5$, $\gcd(a,b) > 1$.
    
    # This is a "biclique" like problem within the bipartite graph between $P_3 \setminus P_5$ and $P_5 \setminus P_3$, plus the intersection $P_3 \cap P_5$ which connects to everyone.
    
    # Let $A = P_3 \setminus P_5$, $B = P_5 \setminus P_3$, $C = P_3 \cap P_5$.
    # $C$ is a clique and connects to all in $A$ and $B$.
    # We want to pick $S_A \subseteq A, S_B \subseteq B$ such that $S_A \cup S_B \cup C$ is a clique.
    # This requires $S_A$ and $S_B$ to be fully connected to each other.
    # i.e., $\gcd(a,b) > 1$ for all $a \in S_A, b \in S_B$.
    
    # Since $C$ is small ($|C|=17$), and $A, B$ are larger, we can try to maximize $|S_A| + |S_B| + |C|$.
    # Note: We can also include elements from $P_{17} \setminus (P_3 \cup P_5)$.
    # Let $D = P_{17} \setminus (P_3 \cup P_5)$.
    # Elements in $D$ must be connected to $S_A, S_B, C$.
    
    # This structure suggests that the maximum clique is likely composed of:
    # 1. All of $C$ (multiples of 15).
    # 2. A large subset of $A$ (multiples of 3, not 5).
    # 3. A large subset of $B$ (multiples of 5, not 3).
    # 4. A large subset of $D$ (multiples of 17, not 3 or 5).
    
    # However, $A$ and $B$ are not fully connected to each other.
    # We can just run the Max Clique algorithm on the 127 nodes. It's the safest bet.
    
    # I will implement a simple recursive max clique finder with bitmasks.
    
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    def search(R_mask, P_mask, X_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0 and X_mask == 0:
            sz = popcount(R_mask)
            if sz > best_size:
                best_size = sz
                best_mask = R_mask
            return
            
        # Pivot selection
        # Choose u in P_mask | X_mask with max popcount(adj[u] & P_mask)
        # For speed, we can just pick the first vertex in P_mask | X_mask?
        # No, that's bad.
        # Let's pick u in P_mask with max degree in P_mask.
        
        # To avoid O(N^2) per node, we can precompute degrees?
        # Or just use a heuristic.
        
        # Given N=127, let's just iterate through P_mask and find the best pivot.
        # This is O(N^2) per node.
        # Depth is at most N.
        # Worst case O(N^3) or more?
        # But the graph is dense, so branching factor is low.
        
        # Let's try to pick u = argmax_{v in P_mask | X_mask} popcount(adj[v] & P_mask)
        
        # Implementation:
        # candidates_for_pivot = P_mask | X_mask
        # best_u = -1
        # max_deg = -1
        # temp = candidates_for_pivot
        # while temp:
        #    v = (temp & -temp).bit_length() - 1
        #    temp ^= (1 << v)
        #    deg = popcount(adj[v] & P_mask)
        #    if deg > max_deg:
        #       max_deg = deg
        #       best_u = v
        #       if deg == popcount(P_mask): # u is connected to all in P
        #           break
            
        # if best_u == -1: return # Should not happen if P_mask != 0
        
        # U = P_mask & ~adj[best_u]
        
        # while U:
        #    v = (U & -U).bit_length() - 1
        #    U ^= (1 << v)
        #    search(R_mask | (1 << v), P_mask & adj[v], X_mask & adj[v])
        #    P_mask ^= (1 << v)
        #    X_mask |= (1 << v)
        
        # This is the standard algorithm.
        
        pass

    # Let's write the actual code.

    # Pre-compute adj bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_size = 0
    best_mask = 0

    def popcount(x):
        return bin(x).count('1')

    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0 and X_mask == 0:
            sz = popcount(R_mask)
            if sz > best_size:
                best_size = sz
                best_mask = R_mask
            return
            
        # Select pivot
        # We want u in P_mask | X_mask
        # To optimize, we pick u in P_mask | X_mask that maximizes |adj[u] & P_mask|
        
        # If P_mask is empty, we return above.
        
        # Heuristic: Just pick the first vertex in P_mask?
        # No, let's pick the one with max degree in P_mask.
        
        # To make it faster, we can just iterate through P_mask.
        # If the graph is dense, the number of nodes visited is small.
        
        # Let's try a simpler pivot: pick u from P_mask | X_mask with max degree in P_mask.
        
        # For performance, I'll use a slightly simplified version:
        # Pick u from P_mask.
        
        # Actually, for N=127, even a naive pivot (first vertex) might be fast enough if the graph is dense.
        # Let's try to be a bit smarter.
        
        # Find u in P_mask | X_mask with max popcount(adj[u] & P_mask)
        
        # To avoid heavy computation, I'll just pick u = 0 if 0 in P_mask | X_mask, else first bit.
        # No, that's risky.
        
        # Let's use the standard max-degree pivot.
        
        # Get list of vertices in P_mask | X_mask
        # This is O(N)
        
        # Calculate degrees
        # This is O(N)
        
        # Total O(N^2) per node.
        
        # Let's hope the tree is small.
        
        # Alternative: Use a library? No, standard lib only.
        
        # I will implement the max-degree pivot.
        
        # To speed up, I can precompute the list of bits for each mask? No, masks change.
        
        # Let's just do it.
        
        # Find pivot u
        pivot_candidates = P_mask | X_mask
        best_u = -1
        max_deg = -1
        
        temp = pivot_candidates
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            deg = popcount(adj[v] & P_mask)
            if deg > max_deg:
                max_deg = deg
                best_u = v
                if deg == popcount(P_mask):
                    break
                    
        if best_u == -1:
            return
            
        # U = P_mask & ~adj[best_u]
        U = P_mask & ~adj[best_u]
        
        while U:
            v_bit = U & -U
            v = v_bit.bit_length() - 1
            U ^= v_bit
            
            # Recurse
            bron_kerbosch(R_mask | v_bit, P_mask & adj[v], X_mask & adj[v])
            
            # Update P and X
            P_mask ^= v_bit
            X_mask |= v_bit

    # Initialize
    # Start with R=0, P=all nodes, X=0
    initial_P = (1 << n) - 1
    bron_kerbosch(0, initial_P, 0)
    
    # Extract result
    if best_mask == 0:
        # Fallback: return just 255
        return [255]
        
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(candidates[i])
            
    # Ensure 255 is in result?
    # The problem requires 255 in the set.
    # Since 255 is connected to all, it should be in the max clique.
    # But if the max clique found doesn't have 255, we can add it?
    # No, if 255 is not in the clique, adding it might break it?
    # No, 255 is connected to ALL nodes in C.
    # So if K is a clique in C, K U {255} is also a clique in C.
    # So the maximum clique MUST contain 255 (or there is an equal size one that does).
    # So we can just return the result.
    
    return result

# EVOLVE-BLOCK-END
