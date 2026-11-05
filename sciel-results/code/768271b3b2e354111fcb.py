import time
from math import gcd
import numpy as np

N = 512
DEADLINE = 85.0  # Safety margin from 90s limit

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Strong initial lower bound: divisors of 360
    # All divisors of 360 have lcm <= 360 <= 512
    divisors_360 = [d for d in range(1, N + 1) if 360 % d == 0]
    # Convert to 0-indexed
    initial_set = [d - 1 for d in divisors_360]
    best_size = len(initial_set)
    best_set = list(divisors_360)  # Store 1-indexed values
    
    # 3. Greedy coloring bound function
    # Returns an upper bound on the size of an independent set in the subgraph induced by R_mask
    def greedy_color_bound(R_mask):
        """
        Greedy coloring of the conflict graph induced by R_mask.
        The number of colors used is an upper bound on the clique number,
        which is a lower bound on... wait, we need upper bound on independent set.
        
        Actually, for MIS, we use: the chromatic number of the subgraph gives
        a lower bound on clique cover, but for upper bound on independent set,
        we can use: color the vertices, each color class is an independent set in 
        the complement... 
        
        Let me reconsider. In the conflict graph, an independent set is what we want.
        The greedy coloring bound works as follows:
        - Color the vertices of the conflict graph induced by R
        - The maximum color used is an upper bound on the size of any clique in the 
          complement graph, which equals the size of any independent set in the conflict graph.
        
        Wait, that's not right either. Let me think again.
        
        Standard approach for MIS:
        - We have a graph G (conflict graph)
        - We want maximum independent set in G
        - Equivalent to maximum clique in complement of G
        - Greedy coloring bound: color G, the number of colors is an upper bound on 
          the clique number of G, which is a lower bound on... 
        
        Actually, the standard Tomita algorithm uses:
        - Color the vertices of the candidate set
        - The color of a vertex v is an upper bound on the size of an independent set 
          containing v (in the subgraph)
        
        Let me implement a simpler bound: the size of R is an upper bound, but we can 
        do better with coloring.
        
        For the conflict graph G, if we greedily color the vertices such that adjacent 
        vertices get different colors, then any independent set can contain at most 
        one vertex of each color? No, that's for cliques.
        
        Let me just use: upper bound = popcount(R_mask) as a simple bound, and rely 
        on good branching to prune.
        """
        # Simple bound: number of remaining vertices
        return R_mask.bit_count() if hasattr(int, 'bit_count') else bin(R_mask).count('1')
    
    # 4. Improved B&B with better heuristics
    best_size = len(initial_set)
    best_set = list(divisors_360)
    
    # Precompute degrees for ordering
    degrees = [adj[i].bit_count() if hasattr(int, 'bit_count') else bin(adj[i]).count('1') for i in range(N)]
    
    def solve(R_mask, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            return
        
        if R_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Bound: current_size + popcount(R_mask) <= best_size -> prune
        rem = R_mask.bit_count() if hasattr(int, 'bit_count') else bin(R_mask).count('1')
        if current_size + rem <= best_size:
            return
        
        # Choose a vertex to branch on: highest degree in R
        best_v = -1
        max_deg = -1
        
        temp_mask = R_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            deg = (adj[v] & R_mask).bit_count() if hasattr(int, 'bit_count') else bin(adj[v] & R_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_mask ^= lsb
        
        if best_v == -1:
            return
            
        # Branch 1: Include best_v (try this first as it may find larger solutions faster)
        R_incl = R_mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v + 1)
        solve(R_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        R_excl = R_mask & ~(1 << best_v)
        solve(R_excl, current_size, current_set)
    
    # 5. Run B&B starting from the initial set
    # We'll try to improve upon the initial set
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # 6. Local improvement: try to add elements to the best set found
    best_set_set = set(best_set)
    improved = True
    while improved:
        improved = False
        for v in range(1, N + 1):
            if v in best_set_set:
                continue
            # Check if v can be added
            can_add = True
            for u in best_set_set:
                g = gcd(u, v)
                if (u * v) // g > N:
                    can_add = False
                    break
            if can_add:
                best_set_set.add(v)
                best_set = list(best_set_set)
                best_size = len(best_set)
                improved = True
                if time.time() - start_time > DEADLINE - 2:
                    break
    
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
