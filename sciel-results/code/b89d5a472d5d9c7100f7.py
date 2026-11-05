import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # We aim for n=34. If we can't find it, we fallback to lower n.
    # But the score is n. So we should try 34 first.
    
    def get_cliques(n):
        """
        Returns:
        k4s: list of tuples of 4 edge indices
        k5s: list of tuples of 10 edge indices
        """
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            idxs = (
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            )
            k4s.append(idxs)
            
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        return k4s, k5s

    def solve_n(n):
        """
        Attempts to find a valid coloring for n vertices.
        Returns colors list if found, else None.
        """
        if n < 1:
            return None
        if n <= 3:
            # Trivial
            return [0] * (n * (n-1) * (n-2) // 6)
            
        k4s, k5s = get_cliques(n)
        num_edges = len(k4s[0]) * 0 # dummy, actually len of edge list
        # Calculate num_edges properly
        num_edges = n * (n - 1) * (n - 2) // 6
        
        # Precompute reverse index: edge_idx -> list of (clique_idx, clique_type)
        # clique_type 0 for K4, 1 for K5
        edge_to_k4s = [[] for _ in range(num_edges)]
        edge_to_k5s = [[] for _ in range(num_edges)]
        
        for i, idxs in enumerate(k4s):
            for e in idxs:
                edge_to_k4s[e].append(i)
                
        for i, idxs in enumerate(k5s):
            for e in idxs:
                edge_to_k5s[e].append(i)

        # Local Search / Simulated Annealing
        best_cost = float('inf')
        best_colors = None
        
        # Multiple restarts
        num_restarts = 50
        steps_per_restart = 20000
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize colors randomly
            colors = np.random.randint(0, 2, num_edges, dtype=np.int8)
            
            # Calculate initial cost
            # Red K4 count
            red_k4 = 0
            for idxs in k4s:
                if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                    red_k4 += 1
            # Blue K5 count
            blue_k5 = 0
            for idxs in k5s:
                if colors[idxs[0]] == 1 and colors[idxs[1]] == 1 and colors[idxs[2]] == 1 and colors[idxs[3]] == 1 and \
                   colors[idxs[4]] == 1 and colors[idxs[5]] == 1 and colors[idxs[6]] == 1 and colors[idxs[7]] == 1 and \
                   colors[idxs[8]] == 1 and colors[idxs[9]] == 1:
                    blue_k5 += 1
            current_cost = red_k4 + blue_k5
            
            if current_cost == 0:
                return colors.tolist()
                
            temp = 1.0
            cool_rate = 0.995
            
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_colors.tolist() if best_colors else None
                
                # Pick random edge
                e = np.random.randint(num_edges)
                old_c = colors[e]
                new_c = 1 - old_c
                
                delta = 0
                
                # Check K4s containing e
                for k4_idx in edge_to_k4s[e]:
                    idxs = k4s[k4_idx]
                    # Count red edges in this K4
                    # Since only e changes, we can check the other 3
                    # If old_c == 0 (Red):
                    #   If other 3 are Red, it was a Red K4. Flipping to Blue breaks it. delta -= 1.
                    #   If other 3 are Blue, it was not a Red K4. Flipping to Blue makes it mixed (not Red). No change to Red K4 count?
                    #   Wait. If other 3 are Blue, and e becomes Blue, it's a Blue K4. Blue K4 is not penalized.
                    #   If other 3 are mixed, no change.
                    # If old_c == 1 (Blue):
                    #   If other 3 are Red, it was not a Red K4. Flipping to Red makes it a Red K4. delta += 1.
                    #   If other 3 are Blue, it was not a Red K4. Flipping to Red makes it mixed. No change.
                    
                    # Optimization: Check the 3 other edges
                    other_idxs = [i for i in idxs if i != e]
                    
                    if old_c == 0:
                        # e was Red, becomes Blue
                        if colors[other_idxs[0]] == 0 and colors[other_idxs[1]] == 0 and colors[other_idxs[2]] == 0:
                            delta -= 1 # Broke a Red K4
                        # Does it create a Blue K4? No, Blue K4 is not penalized.
                    else:
                        # e was Blue, becomes Red
                        if colors[other_idxs[0]] == 0 and colors[other_idxs[1]] == 0 and colors[other_idxs[2]] == 0:
                            delta += 1 # Created a Red K4
                        
                # Check K5s containing e
                for k5_idx in edge_to_k5s[e]:
                    idxs = k5s[k5_idx]
                    other_idxs = [i for i in idxs if i != e]
                    
                    if old_c == 1:
                        # e was Blue, becomes Red
                        # If other 4 are Blue, it was a Blue K5. Flipping to Red breaks it. delta -= 1.
                        if all(colors[i] == 1 for i in other_idxs):
                            delta -= 1
                    else:
                        # e was Red, becomes Blue
                        # If other 4 are Blue, it was not a Blue K5. Flipping to Blue makes it a Blue K5. delta += 1.
                        if all(colors[i] == 1 for i in other_idxs):
                            delta += 1

                new_cost = current_cost + delta
                
                # Acceptance criterion
                if new_cost < current_cost or (new_cost == current_cost and np.random.rand() < 0.5) or (new_cost > current_cost and np.random.rand() < np.exp(-new_cost / temp)):
                    colors[e] = new_c
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors.tolist()
                
                temp *= cool_rate
                if temp < 0.01:
                    temp = 1.0
                    
        return best_colors.tolist() if best_colors and best_cost == 0 else None

    # Try n=34
    for n in [34, 33, 32, 30, 25, 20]:
        if time.time() > deadline:
            break
        try:
            res = solve_n(n)
            if res:
                return {'n': n, 'colors': res}
        except Exception as e:
            print(f"Error for n={n}: {e}")
            
    # Fallback to a small valid n if all else fails
    # n=9 is always valid? No, R(4,5;3) > 9 is known.
    # Let's just return n=1 to be safe, or a simple random n=5.
    # But we should try to return the best found.
    # If solve_n failed, it returned None.
    
    # Fallback: Return a trivial coloring for n=4?
    # R(4,5;3) > 4 is true.
    n = 4
    colors = [0, 0, 0, 0] # All red K4? No, 4 vertices, 1 K4. If all red, it's a violation.
    # We need a valid one.
    # For n=4, we just need to avoid all-red K4. So at least one edge must be blue.
    colors = [0, 0, 0, 1]
    return {'n': 4, 'colors': colors}
