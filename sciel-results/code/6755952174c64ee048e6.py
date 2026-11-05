import time
from math import gcd

N = 512
DEADLINE = 88.0  # Safety margin from 90s limit

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    # adj[i] is a bitmask of vertices j such that lcm(i+1, j+1) > N
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to get bit count
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    # 2. Greedy Coloring for Upper Bound
    # This function colors the subgraph induced by R_mask.
    # It returns a list of (vertex, color) pairs sorted by color descending, 
    # or just the max color for pruning.
    # To be efficient, we return the max color and the ordered list of vertices.
    def greedy_color_bound(R_mask):
        """
        Returns (max_color, ordered_vertices) where ordered_vertices is a list of 
        (vertex_index, color) sorted by color descending.
        """
        if R_mask == 0:
            return 0, []
        
        colors = {}
        uncolored = R_mask
        color = 0
        
        while uncolored:
            color += 1
            # Find maximal independent set in the remaining graph for this color
            # Greedily pick vertices that are not adjacent to any vertex already picked for this color
            # And are in uncolored
            available = uncolored
            picked = 0
            while available:
                # Pick a vertex with max degree in available? Or just the first one?
                # Standard Tomita: pick vertex with max degree in the current color class candidate set
                # Here we just pick the lowest index for simplicity, or highest degree.
                # Let's pick the vertex with the highest degree in 'available' to make the color class smaller?
                # Actually, for coloring, we want to assign as many vertices as possible to the current color.
                # So we want a maximal independent set.
                # Greedy MIS: pick vertex, remove it and its neighbors.
                
                # Find vertex with max degree in 'available'
                best_v = -1
                max_deg = -1
                temp = available
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = popcount(adj[v] & available)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    temp ^= lsb
                
                if best_v == -1:
                    break
                    
                colors[best_v] = color
                picked |= (1 << best_v)
                # Remove best_v and its neighbors from available
                available &= ~(adj[best_v] | (1 << best_v))
                
            # Remove all picked vertices from uncolored
            uncolored &= ~picked
            
        # Create ordered list
        ordered = sorted(colors.items(), key=lambda x: x[1], reverse=True)
        return color, ordered

    # 3. Branch and Bound
    best_size = 0
    best_set = []
    
    # Initial lower bound: Divisors of 360
    # 360 = 2^3 * 3^2 * 5
    # LCM of any two divisors of 360 divides 360, so <= 512.
    divisors_360 = [d for d in range(1, N + 1) if 360 % d == 0]
    best_size = len(divisors_360)
    best_set = divisors_360[:]
    
    # Also try a greedy heuristic to potentially get a better start
    # Greedy: Pick vertices with smallest index? Or largest?
    # Let's try to add vertices to the initial set if possible
    current_set = set(divisors_360)
    changed = True
    while changed:
        changed = False
        for v in range(1, N + 1):
            if v in current_set:
                continue
            # Check if v can be added
            can_add = True
            for u in current_set:
                g = gcd(u, v)
                if (u * v) // g > N:
                    can_add = False
                    break
            if can_add:
                current_set.add(v)
                changed = True
                if time.time() - start_time > DEADLINE - 10:
                    break
        if time.time() - start_time > DEADLINE - 10:
            break
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    def solve(R_mask, current_size, current_set_list):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if R_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set_list[:]
            return
        
        # Get color bound and ordered vertices
        max_color, ordered_vertices = greedy_color_bound(R_mask)
        
        # Prune if upper bound is not better than current best
        if current_size + max_color <= best_size:
            return
            
        # Iterate through vertices in order of decreasing color
        for v, c in ordered_vertices:
            # If even with this vertex's color we can't beat best, stop
            # Note: c is an upper bound on the size of the independent set containing v 
            # in the subgraph induced by R_mask at the time of coloring.
            # Actually, in Tomita's algorithm, the color of v is an upper bound on the 
            # size of the independent set in the subgraph induced by the vertices 
            # with color >= c that are still available.
            if current_size + c <= best_size:
                break
                
            # Branch 1: Include v
            # New R_mask: remove v and all its neighbors
            new_R_mask = R_mask & ~(adj[v] | (1 << v))
            current_set_list.append(v + 1)
            solve(new_R_mask, current_size + 1, current_set_list)
            current_set_list.pop()
            
            # Branch 2: Exclude v
            # Remove v from R_mask
            R_mask &= ~(1 << v)
            
            # Check deadline inside loop
            if time.time() - start_time > DEADLINE:
                return

    # Start B&B with all vertices
    # We can potentially restrict the search space by removing vertices 
    # that are "dominated" or obviously not in the optimal set, 
    # but for now, let's run on the full graph.
    # To speed up, we can order the initial R_mask.
    initial_R_mask = (1 << N) - 1
    
    solve(initial_R_mask, 0, [])
    
    if best_size == 0:
        return [1]
        
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
