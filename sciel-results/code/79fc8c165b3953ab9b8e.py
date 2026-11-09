import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    n = 16
    # Edge index mapping: (i, j) -> idx
    # i < j
    # idx = i * (2*n - i - 1) // 2 + (j - i - 1) ? 
    # Let's just build it simply.
    edges = []
    edge_idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_idx_map[(i, j)] = idx
            edge_idx_map[(j, i)] = idx
            idx += 1
            
    total_edges = len(edges) # 120

    # K4s
    k4s = []
    edge_k4s = [[] for _ in range(total_edges)]
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4_idx = len(k4s)
        k4s.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    total_k4s = len(k4s) # 1820

    # Helper to calculate penalty for a specific K4
    def get_k4_penalty(k4_idx, colors):
        e_indices = k4s[k4_idx]
        c0 = colors[e_indices[0]]
        c1 = colors[e_indices[1]]
        c2 = colors[e_indices[2]]
        c3 = colors[e_indices[3]]
        c4 = colors[e_indices[4]]
        c5 = colors[e_indices[5]]
        
        # Unroll set creation for speed? 
        # set is usually fast in CPython for small sets.
        distinct = len({c0, c1, c2, c3, c4, c5})
        return max(0, 5 - distinct)

    def calc_total_penalty(colors):
        total = 0
        for k in range(total_k4s):
            total += get_k4_penalty(k, colors)
        return total

    # Search for a valid coloring with K colors
    def search_for_K(K, time_budget):
        t_start = time.time()
        best_penalty = float('inf')
        best_colors = None
        
        # Multiple restarts
        while time.time() - t_start < time_budget and time.time() < deadline:
            # Initialize
            colors = [random.randint(0, K - 1) for _ in range(total_edges)]
            
            # Optional: Structured init
            # for idx, (i, j) in enumerate(edges):
            #     colors[idx] = (i + j) % K
            
            current_penalty = calc_total_penalty(colors)
            
            if current_penalty == 0:
                return colors
            
            T = 1.0
            T_min = 0.05
            cooling_rate = 0.9999
            stuck_count = 0
            
            # Simulated Annealing loop
            while time.time() - t_start < time_budget and time.time() < deadline:
                # Pick random edge
                e_idx = random.randint(0, total_edges - 1)
                old_color = colors[e_idx]
                
                # Pick random new color
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta penalty
                # Only affected K4s are those containing e_idx
                old_delta = 0
                new_delta = 0
                affected_k4s = edge_k4s[e_idx]
                
                for k4_idx in affected_k4s:
                    old_delta += get_k4_penalty(k4_idx, colors)
                
                colors[e_idx] = new_color
                for k4_idx in affected_k4s:
                    new_delta += get_k4_penalty(k4_idx, colors)
                
                delta = new_delta - old_delta
                new_penalty = current_penalty + delta
                
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors[:]
                    stuck_count = 0
                else:
                    # Accept with probability exp(-delta / T)
                    if delta > 0:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_penalty = new_penalty
                        else:
                            # Revert
                            colors[e_idx] = old_color
                    else:
                        # delta == 0
                        current_penalty = new_penalty
                        if random.random() < 0.5:
                             # Revert to explore? No, accept to move
                             pass
                        else:
                             colors[e_idx] = old_color
                             current_penalty -= delta # revert delta? 
                             # Actually if delta==0, penalty is same. 
                             # If we revert, penalty is same.
                             # Let's just accept delta==0 moves to shuffle.
                             colors[e_idx] = new_color # keep new
                             current_penalty = new_penalty
                             
                    if current_penalty > best_penalty:
                        stuck_count += 1
                    else:
                        stuck_count = 0

                if current_penalty == 0:
                    return colors

                # Cooling
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    
                # Restart condition: if stuck too long
                if stuck_count > 5000:
                    break

        return best_colors if best_colors is not None else None

    # Try K = 13, 14, 15
    # Prioritize 13. If we find it, great.
    # We allocate time based on difficulty.
    
    # First, ensure we have a fallback valid solution.
    # A 17-coloring is easy to construct and verify.
    # But we want to minimize.
    
    # Let's try 13 with a generous budget.
    colors_13 = search_for_K(13, 40.0)
    if colors_13 is not None and calc_total_penalty(colors_13) == 0:
        return colors_13

    # If 13 failed, try 14
    colors_14 = search_for_K(14, 30.0)
    if colors_14 is not None and calc_total_penalty(colors_14) == 0:
        return colors_14

    # If 14 failed, try 15
    colors_15 = search_for_K(15, 20.0)
    if colors_15 is not None and calc_total_penalty(colors_15) == 0:
        return colors_15

    # Fallback: Generate a valid 17-coloring
    # (i + j) % 17 is a known valid construction for K_n with f(n) <= n-1? 
    # Actually (i+j)%17 works for K16?
    # Let's verify.
    base_colors = [0] * 120
    for idx, (i, j) in enumerate(edges):
        base_colors[idx] = (i + j) % 17
        
    if calc_total_penalty(base_colors) == 0:
        return base_colors

    # If even that fails (it shouldn't), return random 17
    # But (i+j)%17 is standard.
    # Let's just return the best we found, even if invalid? 
    # NO, "invalid result ... scores 0".
    # So we MUST return a valid one.
    # If search failed, we must force a valid one.
    # We can use a greedy repair on the base 17 coloring if it was invalid?
    # But (i+j)%17 is valid.
    
    # Double check (i+j)%17 validity.
    # If it is valid, return it.
    # If not, we have a problem.
    # But it is a standard construction.
    
    return base_colors
