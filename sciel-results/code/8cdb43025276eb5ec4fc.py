import time
from math import gcd

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # 1. Build the conflict graph
    # Vertices 0 to 127 represent numbers 1 to 128
    # adj[i] is a bitset (int) of neighbors of i
    
    adj = [0] * N
    
    # Precompute gcds for all pairs to speed up? 
    # Or just compute on the fly. N=128 is small.
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            g = gcd(u, v)
            u_prime = u // g
            v_prime = v // g
            max_k = N // g
            
            # Check if there exists k in [1, max_k] 
            # such that gcd(u_prime, k) == 1 and gcd(v_prime, k) == 1
            found = False
            for k in range(1, max_k + 1):
                if gcd(u_prime, k) == 1 and gcd(v_prime, k) == 1:
                    found = True
                    break
            
            if found:
                # Edge exists between u-1 and v-1
                adj[u - 1] |= (1 << (v - 1))
                adj[v - 1] |= (1 << (u - 1))

    # 2. Solve Maximum Independent Set
    # We use a Branch and Bound approach with a greedy coloring bound.
    
    best_set = []
    
    # Greedy initial solution
    # Sort by degree (descending)
    degrees = [bin(adj[i]).count('1') for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    current_set = set()
    for v in order:
        # Check if v is independent from current_set
        # v is connected to any in current_set?
        conflict = False
        for u in current_set:
            if (adj[v] >> u) & 1:
                conflict = True
                break
        if not conflict:
            current_set.add(v)
            
    best_size = len(current_set)
    best_set = list(current_set)
    
    # B&B Search
    # We represent the set of available vertices as a bitset `avail`
    # and the current independent set as a list or bitset.
    # For efficiency in Python, we might use lists for the set if it's small,
    # but bitsets are better for neighbor checks.
    
    # Let's implement a recursive B&B
    # To avoid recursion depth issues or overhead, we might need to be careful.
    # N=128 is small enough for recursion.
    
    # We need a function to compute the greedy coloring bound for a subgraph defined by a bitset.
    def get_color_bound(mask):
        # Returns the number of colors needed to color the subgraph induced by 'mask'
        # This is an upper bound on the independent set size within 'mask'
        colors = 0
        remaining = mask
        while remaining:
            colors += 1
            # Greedily pick a vertex, remove it and its neighbors
            # To make the bound tighter, we should pick vertices with high degree in 'remaining'
            # For simplicity, we pick the first available vertex, but ordering matters.
            # A better heuristic: pick the vertex with the highest degree in 'remaining'.
            # However, for speed, simple greedy might suffice if we iterate vertices in a fixed order.
            
            # Let's iterate through vertices in 'remaining'
            # We want to find a maximal independent set in the complement? No.
            # The bound is the chromatic number of the subgraph.
            # Greedy coloring:
            # 1. Start with all vertices in 'remaining' uncolored.
            # 2. Pick a vertex v.
            # 3. Assign current color to v.
            # 4. Remove v and all its neighbors from 'remaining'.
            # 5. Repeat until 'remaining' is empty.
            
            # To get a tight bound, the order of picking vertices matters.
            # We'll pick the vertex with the highest degree in the current 'remaining' subgraph.
            
            # Extract vertices
            verts = []
            temp = remaining
            while temp:
                # Find lowest set bit
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
                
            if not verts:
                break
                
            # Sort by degree in 'remaining'
            # Degree of v in 'remaining' is popcount(adj[v] & remaining)
            # This is expensive if done for all.
            # Let's just pick the first one for speed, or a simple heuristic.
            # Actually, for N=128, even O(N^2) per bound calculation is fine if the tree is small.
            
            # Let's try picking the vertex with the highest degree in 'remaining'
            # to make the color classes small? No, we want to cover as many as possible?
            # The bound is the number of colors.
            # Greedy coloring usually works better if high-degree vertices are colored first.
            
            # Find max degree vertex
            max_deg = -1
            best_v = -1
            for v in verts:
                deg = bin(adj[v] & remaining).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            # Color best_v
            # Remove best_v and its neighbors from remaining
            remaining &= ~(adj[best_v] | (1 << best_v))
            
        return colors

    # Global variables for B&B
    best_size_global = best_size
    best_set_global = best_set
    
    # We search for a set larger than best_size_global
    # avail: bitset of vertices that can still be added
    # current: bitset of vertices currently in the independent set
    
    def bb_search(avail, current, current_size):
        nonlocal best_size_global, best_set_global
        
        if time.time() > deadline:
            return
            
        if avail == 0:
            if current_size > best_size_global:
                best_size_global = current_size
                # Reconstruct set
                s = []
                temp = current
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    s.append(idx)
                    temp ^= lsb
                best_set_global = s
            return
        
        # Bound: current_size + max possible independent set in avail
        bound = get_color_bound(avail)
        if current_size + bound <= best_size_global:
            return
            
        # Select a vertex to branch on
        # Heuristic: pick a vertex with high degree in avail
        # Or simply the first one?
        # Let's pick the one with the highest degree in the subgraph induced by avail
        # to prune effectively.
        
        verts = []
        temp = avail
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        if not verts:
            return
            
        # Find max degree in avail
        max_deg = -1
        branch_v = -1
        for v in verts:
            deg = bin(adj[v] & avail).count('1')
            if deg > max_deg:
                max_deg = deg
                branch_v = v
                
        # Branch 1: Include branch_v
        # Check if branch_v is independent from current
        # (It is, because it's in avail, and avail only contains vertices independent from current? 
        # Wait, in standard MIS B&B, 'avail' is the set of vertices that are not yet decided 
        # AND are not adjacent to any vertex in 'current'.
        # So any v in avail is safe to add to current.)
        
        new_current = current | (1 << branch_v)
        # New avail: remove branch_v and its neighbors from avail
        new_avail = avail & ~(adj[branch_v] | (1 << branch_v))
        
        bb_search(new_avail, new_current, current_size + 1)
        
        # Branch 2: Exclude branch_v
        # Remove branch_v from avail
        new_avail_excl = avail & ~(1 << branch_v)
        
        bb_search(new_avail_excl, current, current_size)

    # Initial call
    # We start with empty set, all vertices available
    initial_avail = (1 << N) - 1
    bb_search(initial_avail, 0, 0)
    
    return [x + 1 for x in best_set_global]

# Test logic
if __name__ == "__main__":
    result = construct()
    print(len(result))
