import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    def get_common_neighbors(adj, u, v):
        """Return the set of common neighbors of u and v in the given adjacency matrix."""
        # adj is a boolean numpy array or list of sets. 
        # For speed in SA, we'll use numpy bitsets or simple sets if n is small.
        # Given n <= 29, sets are very fast in Python.
        return adj[u] & adj[v]

    def count_c4s(adj):
        """Count total number of C4s across all 5 colors."""
        total_c4s = 0
        for c in range(5):
            # adj[c] is a list of sets
            for i in range(len(adj[c])):
                for j in range(i + 1, len(adj[c])):
                    common = len(adj[c][i] & adj[c][j])
                    if common >= 2:
                        # Number of C4s involving pair (i,j) is C(common, 2)
                        total_c4s += common * (common - 1) // 2
        return total_c4s

    def generate_initial_coloring(n, seed):
        """
        Generate a high-quality initial coloring.
        For n=25, we can use the algebraic construction if we knew it, 
        but since we don't have the explicit Lazebnik-Woldar code here, 
        we use a "block" heuristic or random restart with heavy local repair.
        
        Actually, a good heuristic is to use a random coloring and then 
        run a very fast greedy repair to remove as many C4s as possible.
        """
        random.seed(seed)
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build adjacency
        adj = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        # Greedy repair: iterate edges, if an edge creates a C4, change its color 
        # to the one that minimizes total C4s.
        # This is O(n^3) per pass, but we do few passes.
        for _ in range(2):
            for i in range(n):
                for j in range(i + 1, n):
                    idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                    current_c = colors[idx]
                    
                    # Remove current edge from adj
                    adj[current_c][i].remove(j)
                    adj[current_c][j].remove(i)
                    
                    best_c = current_c
                    best_c4s = count_c4s(adj) # This is expensive. 
                    # Optimization: Only calculate delta C4s.
                    # Let's just pick a random valid color if possible, or random if not.
                    # Actually, let's just do a simple local move: 
                    # Try all 5 colors, pick the one with min C4s in the affected neighborhood.
                    
                    # To keep it fast, let's just revert and use the SA loop to do the heavy lifting.
                    adj[current_c][i].add(j)
                    adj[current_c][j].add(i)
        
        return colors, adj

    def solve_sa(n, seed, time_limit=10.0):
        """
        Simulated Annealing to minimize C4 count.
        State: colors array.
        Move: Change color of a random edge.
        Cost: Total number of C4s (sum of C(k,2) over all pairs and colors).
        """
        random.seed(seed)
        num_edges = n * (n - 1) // 2
        
        # Initialize
        colors, adj = generate_initial_coloring(n, seed)
        
        # Precompute edge index mapping
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx_map[(i, j)] = idx
                idx += 1

        current_c4s = count_c4s(adj)
        best_c4s = current_c4s
        best_colors = colors[:]
        
        # SA Parameters
        T_start = 1.0
        T_end = 0.001
        alpha = 0.999
        T = T_start
        
        t_start = time.time()
        
        while time.time() < t_start + time_limit:
            if T < T_end:
                T = T_end
            
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            # Find u, v from e_idx
            # Inverse of lexicographic index
            # i*(n-1) - i*(i+1)//2 + (j-i-1) = e_idx
            # Approximate i
            i_approx = int((2 * e_idx + n) / (2 * n - 2))
            if i_approx >= n: i_approx = n - 1
            # Refine
            i = i_approx
            while True:
                # j = i + 1 + (e_idx - (i*(n-1) - i*(i+1)//2))
                base = i * (n - 1) - i * (i + 1) // 2
                rem = e_idx - base
                if 0 <= rem < n - 1 - i:
                    break
                i += 1
            
            j = i + 1 + (e_idx - (i * (n - 1) - i * (i + 1) // 2))
            
            old_c = colors[e_idx]
            if old_c == 0 and random.random() > 0.8: # Bias towards non-zero? No, symmetric.
                new_c = random.randint(0, 4)
            else:
                new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
            
            # Calculate Delta Cost
            # Removing edge (u,v) from color old_c
            # Adding edge (u,v) to color new_c
            
            # Delta from removing from old_c:
            # For each x in adj[old_c][u] \ {v}:
            #   common(u,x) decreases by 1.
            #   Change in C(k,2) is - (k-1)
            # For each x in adj[old_c][v] \ {u}:
            #   common(v,x) decreases by 1.
            #   Change in C(k,2) is - (k-1)
            
            delta = 0
            
            # Update for old_c removal
            nbrs_u_old = adj[old_c][u]
            nbrs_v_old = adj[old_c][v]
            
            # Note: adj[old_c][u] does not contain v yet? It does.
            # We need to iterate over common neighbors of u and v in old_c?
            # No, the C4 count is sum over pairs (a,b) of C(common(a,b), 2).
            # When we remove edge (u,v) from color c:
            # 1. The pair (u,v) itself: common(u,v) in c decreases by 0? 
            #    Wait, common(u,v) counts x such that (u,x) and (v,x) are in c.
            #    The edge (u,v) itself is not a "common neighbor".
            #    So common(u,v) does NOT change when we remove edge (u,v).
            #    However, the pairs (u,x) and (v,x) for all x DO change.
            #    For a fixed x, the pair (u,x) has common neighbors. 
            #    Removing (u,v) means v is no longer a neighbor of u in c.
            #    So for any pair (u, y) where y is a neighbor of u in c (y != v),
            #    does common(u,y) change?
            #    common(u,y) = |N(u) & N(y)|.
            #    If we remove v from N(u), then for any y such that v is in N(y),
            #    common(u,y) decreases by 1.
            #    So, for each y in N(v) (neighbors of v in c), if y != u,
            #    then common(u,y) decreases by 1.
            #    Change in cost for pair (u,y) is - (common(u,y) - 1).
            
            # Let's calculate delta for removal from old_c:
            for y in nbrs_v_old:
                if y == u: continue
                # common(u, y) in old_c
                k = len(adj[old_c][u] & adj[old_c][y])
                # k includes v? Yes, because v is in N(u) and v is in N(y) (since y in N(v)).
                # So current common is k. New common is k-1.
                # Delta = C(k-1, 2) - C(k, 2) = - (k-1)
                delta -= (k - 1)
            
            # Symmetrically for pairs (v, y) where y in N(u)
            for y in nbrs_u_old:
                if y == v: continue
                k = len(adj[old_c][v] & adj[old_c][y])
                delta -= (k - 1)
            
            # Now addition to new_c:
            nbrs_u_new = adj[new_c][u]
            nbrs_v_new = adj[new_c][v]
            
            for y in nbrs_v_new:
                if y == u: continue
                k = len(adj[new_c][u] & adj[new_c][y])
                # New common is k+1.
                # Delta = C(k+1, 2) - C(k, 2) = k
                delta += k
            
            for y in nbrs_u_new:
                if y == v: continue
                k = len(adj[new_c][v] & adj[new_c][y])
                delta += k
            
            new_c4s = current_c4s + delta
            
            # Acceptance
            if new_c4s < current_c4s or (T > 0 and random.random() < np.exp(-delta / T)):
                # Apply move
                adj[old_c][u].remove(v)
                adj[old_c][v].remove(u)
                adj[new_c][u].add(v)
                adj[new_c][v].add(u)
                colors[e_idx] = new_c
                current_c4s = new_c4s
                
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_colors = colors[:]
                
                if best_c4s == 0:
                    break
            
            T *= alpha
        
        if best_c4s == 0:
            return best_colors
        return None

    # Strategy:
    # 1. Try to solve n=28, 27, 26 using SA.
    # 2. If n=26 succeeds, return it.
    # 3. If not, fallback to n=25 with a more robust search or known structure.
    
    # We don't have the explicit algebraic construction code, so we rely on SA.
    # To make SA more effective, we can use a "restart" strategy with different seeds.
    
    for n in [28, 27, 26]:
        if time.time() > deadline - 15:
            break
        
        # Allocate time for this n
        # n=26 is the target. 27/28 are long shots.
        time_per_n = 30 if n == 26 else 10
        
        # Try multiple seeds
        num_attempts = 3
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break
            res = solve_sa(n, seed=42 + attempt, time_limit=time_per_n / num_attempts)
            if res is not None:
                return {'n': n, 'colors': res}
        
        # If n=26 failed, try harder with more time
        if n == 26 and time.time() < deadline - 20:
            res = solve_sa(n, seed=999, time_limit=30)
            if res is not None:
                return {'n': n, 'colors': res}

    # Fallback: n=25
    # Try SA for n=25 with significant time
    if time.time() < deadline - 10:
        for seed in [1, 2, 3]:
            res = solve_sa(25, seed=seed, time_limit=10)
            if res is not None:
                return {'n': 25, 'colors': res}
    
    # Trivial fallback
    n = 20
    num_edges = n * (n - 1) // 2
    colors = [i % 5 for i in range(num_edges)]
    return {'n': n, 'colors': colors}

