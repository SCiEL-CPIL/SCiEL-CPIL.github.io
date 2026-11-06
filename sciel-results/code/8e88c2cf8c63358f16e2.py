import time
import sys

def construct():
    start_time = time.time()
    limit = 55.0 # Safety margin for 60s limit
    
    n = 256
    
    # Weights: 1/i
    # Using float for speed, but we need exact fraction for details? 
    # The task says "score is its floating serialization with exact numerator/denominator in details".
    # The return is just the list. The evaluator calculates the score.
    weights = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Build Conflict Graph G
    # u, v conflict if they form a 3-AP with some w in [1, n]
    # adj_conflict[i] is a bitmask of j such that i, j conflict
    adj_conflict = [0] * (n + 1)
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            # Check if u, v are in a 3-AP
            # 1. w = 2u - v
            w = 2 * u - v
            if 1 <= w <= n and w != u and w != v:
                adj_conflict[u] |= (1 << v)
                adj_conflict[v] |= (1 << u)
                continue
            # 2. w = 2v - u
            w = 2 * v - u
            if 1 <= w <= n and w != u and w != v:
                adj_conflict[u] |= (1 << v)
                adj_conflict[v] |= (1 << u)
                continue
            # 3. w = (u + v) / 2
            if (u + v) % 2 == 0:
                w = (u + v) // 2
                if 1 <= w <= n and w != u and w != v:
                    adj_conflict[u] |= (1 << v)
                    adj_conflict[v] |= (1 << u)
    
    # Build Complement Graph G_c for Max Clique
    # adj_complement[i] is a bitmask of j such that i, j do NOT conflict (and i != j)
    # In G_c, an edge means "compatible".
    # We want Max Weight Clique in G_c.
    adj_complement = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # All nodes except i and its conflicts
        all_nodes_mask = (1 << (n + 1)) - 2 # Bits 1..n set
        # Remove i
        all_nodes_mask &= ~(1 << i)
        # Remove conflicts
        all_nodes_mask &= ~adj_conflict[i]
        adj_complement[i] = all_nodes_mask

    # Max Weight Clique Branch and Bound on G_c
    # State:
    # P: Bitmask of candidate vertices
    # Q: Bitmask of current clique vertices
    # bound: Upper bound on weight of Q + P
    
    best_weight = 0.0
    best_clique_mask = 0
    
    # Helper to get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]

    # Greedy Coloring for Upper Bound
    # We color the subgraph induced by P in G_c.
    # We assign colors such that adjacent vertices have different colors.
    # The number of colors is an upper bound on the clique size? No.
    # For Max Clique, if we color the graph, any clique can have at most one vertex of each color.
    # So, if we have color classes C_1, ..., C_k, the max weight clique is <= sum of max weights in each C_i.
    # We need to order the vertices in P to get a tight bound.
    # Standard Tomita: Order vertices by weight (descending) or degree.
    
    def color_bound(P_mask, Q_weight):
        """
        Returns an upper bound for the weight of any clique extending Q using vertices from P.
        Uses greedy coloring on the subgraph induced by P in G_c.
        """
        if P_mask == 0:
            return Q_weight
        
        # Extract vertices in P
        P_list = []
        temp = P_mask
        while temp:
            low_bit = temp & -temp
            v = low_bit.bit_length() - 1
            P_list.append(v)
            temp &= temp - 1
            
        # Sort by weight descending to get better bounds earlier? 
        # Or sort by degree in P?
        # Let's sort by weight descending.
        P_list.sort(key=lambda x: weights[x], reverse=True)
        
        colors = {} # v -> color_id
        color_max_weights = [] # max weight in each color class
        
        for v in P_list:
            # Find the smallest color that v can take
            # v can take color c if no neighbor of v in P has color c
            used_colors = set()
            # Check neighbors of v in P
            neighbors_in_P = adj_complement[v] & P_mask
            temp_n = neighbors_in_P
            while temp_n:
                low_bit = temp_n & -temp_n
                u = low_bit.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                temp_n &= temp_n - 1
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c >= len(color_max_weights):
                color_max_weights.append(weights[v])
            else:
                if weights[v] > color_max_weights[c]:
                    color_max_weights[c] = weights[v]
        
        # Upper bound: Q_weight + sum(color_max_weights)
        # Note: This bound is valid because any clique in P can pick at most one vertex from each color class.
        # The maximum weight it can pick from color class c is at most color_max_weights[c].
        return Q_weight + sum(color_max_weights)

    # Recursive B&B
    # To avoid recursion depth issues, we can use a stack, but 256 depth is fine for Python if limit is set.
    # However, Python recursion is slow. Let's use a stack-based approach or careful recursion.
    # Given n=256, recursion depth might be large.
    
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(P_mask, Q_mask, Q_weight):
        nonlocal best_weight, best_clique_mask
        
        # Time check
        if time.time() - start_time > limit - 2:
            return
            
        if P_mask == 0:
            if Q_weight > best_weight:
                best_weight = Q_weight
                best_clique_mask = Q_mask
            return
        
        # Calculate Upper Bound
        # To speed up, we can check a simpler bound first: Q_weight + sum(weights of P)
        # But coloring bound is tighter.
        ub = color_bound(P_mask, Q_weight)
        
        if ub <= best_weight:
            return
            
        # Select vertex to branch on
        # Standard strategy: Pick the last vertex in the ordering used for coloring.
        # In our color_bound, we sorted P_list by weight desc.
        # Let's re-extract the sorted list to pick the last one (lowest weight among top candidates?)
        # Actually, Tomita picks the vertex with the highest color index? 
        # Or simply the last one in the sorted order.
        
        P_list = []
        temp = P_mask
        while temp:
            low_bit = temp & -temp
            v = low_bit.bit_length() - 1
            P_list.append(v)
            temp &= temp - 1
        P_list.sort(key=lambda x: weights[x], reverse=True)
        
        # We branch on the last vertex in the sorted list (lowest weight in this ordering)
        # This is often effective as it handles "heavy" vertices first in the bound calculation.
        v = P_list[-1]
        
        # Branch 1: Include v
        # New P: P \ {v} \ N_Gc(v)  (Neighbors in Gc are compatible, so they CANNOT be in the same clique? 
        # Wait. In Gc, an edge means "compatible". 
        # A Clique in Gc is a set of vertices where EVERY pair is connected by an edge in Gc.
        # So if we include v, we can ONLY include other vertices that are neighbors of v in Gc.
        # So new P = (P \ {v}) & adj_complement[v]
        
        new_P_include = (P_mask & ~(1 << v)) & adj_complement[v]
        new_Q_mask = Q_mask | (1 << v)
        new_Q_weight = Q_weight + weights[v]
        
        # Update best if this is a leaf or if we want? No, we update at leaf.
        # But we can update best with current clique + any single vertex? No.
        
        branch_and_bound(new_P_include, new_Q_mask, new_Q_weight)
        
        # Branch 2: Exclude v
        new_P_exclude = P_mask & ~(1 << v)
        branch_and_bound(new_P_exclude, Q_mask, Q_weight)

    # Initial Call
    # P = all vertices 1..n
    initial_P = (1 << (n + 1)) - 2 # Bits 1..n
    initial_Q = 0
    initial_W = 0.0
    
    branch_and_bound(initial_P, initial_Q, initial_W)
    
    # Convert best_clique_mask to list
    result = mask_to_list(best_clique_mask)
    
    # If B&B failed or returned empty, fallback to greedy
    if not result:
        S = set()
        for x in range(1, n + 1):
            # Check if x is compatible with all in S
            # x is compatible with y if not (adj_conflict[x] & (1<<y))
            # i.e., adj_conflict[x] has no bits set for y in S
            valid = True
            S_mask = 0
            for y in S:
                S_mask |= (1 << y)
            if adj_conflict[x] & S_mask:
                valid = False
            if valid:
                S.add(x)
        result = list(S)
        
    return result

if __name__ == "__main__":
    construct()
