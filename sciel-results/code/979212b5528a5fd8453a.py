import time
import random

def construct():
    n = 16
    total_edges = n * (n - 1) // 2
    start_time = time.time()
    deadline = start_time + 115
    
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    # Precompute K4s and edges in each K4
    k4s = []
    k4_edges = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    k4s.append((i, j, k, l))
                    k4_edges.append([
                        edge_idx(i, j), edge_idx(i, k), edge_idx(i, l),
                        edge_idx(j, k), edge_idx(j, l), edge_idx(k, l)
                    ])
                    
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(total_edges)]
    for idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(idx)
            
    def get_k4_colours(colours, k4_idx):
        return [colours[e] for e in k4_edges[k4_idx]]
        
    def is_valid_k4(colours, k4_idx):
        return len(set(get_k4_colours(colours, k4_idx))) >= 5

    def count_invalid(colours):
        inv = 0
        for i in range(len(k4s)):
            if not is_valid_k4(colours, i):
                inv += 1
        return inv

    def update_invalid(colours, edge_idx_changed, old_colour, new_colour, current_invalid):
        # This is complex to maintain incrementally without a full recount if multiple K4s change status.
        # Given 120s, full recount (1820 checks) is ~1820 * 6 ops = 10k ops.
        # 10k ops is ~0.001s.
        # 120s / 0.001s = 120,000 iterations.
        # This is plenty.
        return count_invalid(colours)

    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        # Initialize
        if K >= 15:
            m = n - 1
            colours = [0] * total_edges
            for r in range(m):
                colours[edge_idx(n - 1, r)] = r
                for k in range(1, m // 2 + 1):
                    u = (r + k) % m
                    v = (r - k) % m
                    colours[edge_idx(u, v)] = r
        else:
            colours = [random.randint(0, K - 1) for _ in range(total_edges)]
            
        current_invalid = count_invalid(colours)
        
        if current_invalid == 0:
            return colours
            
        # Local Search
        while time.time() < deadline:
            if current_invalid == 0:
                return colours
                
            # Pick a random invalid K4
            # Find one
            bad_k4_idx = -1
            for _ in range(50):
                idx = random.randint(0, len(k4s) - 1)
                if not is_valid_k4(colours, idx):
                    bad_k4_idx = idx
                    break
            
            if bad_k4_idx == -1:
                # Couldn't find bad K4 in 50 tries, maybe it's valid?
                if count_invalid(colours) == 0:
                    return colours
                # Else pick random edge
                e_idx = random.randint(0, total_edges - 1)
            else:
                # Pick random edge in bad K4
                e_idx = random.choice(k4_edges[bad_k4_idx])
                
            old_c = colours[e_idx]
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            colours[e_idx] = new_c
            current_invalid = count_invalid(colours)
            
            # If it got worse, revert? (Hill Climbing with backtracking)
            # For simplicity, accept all moves (Simulated Annealing style or Random Walk)
            # Given time, random walk might get stuck.
            # Let's try to improve: if new_invalid > old_invalid, revert.
            # But we don't store old_invalid easily in this loop structure.
            
        # If time runs out, return best found
        return colours if count_invalid(colours) == 0 else None

    # Fallback
    m = n - 1
    colours = [0] * total_edges
    for r in range(m):
        colours[edge_idx(n - 1, r)] = r
        for k in range(1, m // 2 + 1):
            u = (r + k) % m
            v = (r - k) % m
            colours[edge_idx(u, v)] = r
    return colours
