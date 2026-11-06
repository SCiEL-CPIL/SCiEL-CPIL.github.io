import time
import random
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit for safety
    
    # Precompute edge index for fast lookup
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N-1)*i - i*(i-1)//2 + (j-i-1)
    
    def check_4set(a, b, c, d, colors):
        ab = colors[edge_idx(a,b)]
        ac = colors[edge_idx(a,c)]
        ad = colors[edge_idx(a,d)]
        bc = colors[edge_idx(b,c)]
        bd = colors[edge_idx(b,d)]
        cd = colors[edge_idx(c,d)]
        # Three 4-cycles on {a,b,c,d}
        if len({ab, bc, cd, ad}) == 4: return True
        if len({ab, bd, cd, ac}) == 4: return True
        if len({ac, bc, bd, ad}) == 4: return True
        return False
    
    def can_set_edge(colors, i, j, new_color):
        idx = edge_idx(i, j)
        old = colors[idx]
        colors[idx] = new_color
        
        others = [v for v in range(N) if v != i and v != j]
        for idx_k in range(len(others)):
            for idx_l in range(idx_k + 1, len(others)):
                k = others[idx_k]
                l = others[idx_l]
                if check_4set(i, j, k, l, colors):
                    colors[idx] = old
                    return False
        colors[idx] = old
        return True
    
    # Initial coloring: color(i,j) = i for i < j (Gallai-like)
    colors = [0] * (N*(N-1)//2)
    for i in range(N-1):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = i
    
    edge_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    current_colors = set(colors)
    next_color = max(current_colors) + 1 if current_colors else 0
    
    best_count = len(current_colors)
    best_result = colors[:]
    
    # Backtracking search with MRV heuristic
    def backtrack(colors, next_color, depth, max_depth):
        nonlocal best_count, best_result
        
        if time.time() > deadline - 2:
            return
        
        current_used = len(set(colors))
        if current_used > best_count:
            best_count = current_used
            best_result = colors[:]
        
        if depth >= max_depth:
            return
        
        # Find edge with most constrained choices (MRV)
        # We'll prioritize edges that can take new colors
        candidates = []
        for (i, j) in edge_list:
            idx = edge_idx(i, j)
            old_color = colors[idx]
            # Check if we can assign a new color
            if can_set_edge(colors, i, j, next_color):
                candidates.append((i, j, idx, old_color))
        
        if not candidates:
            return
        
        # Shuffle for randomness
        random.shuffle(candidates)
        
        # Try the first few candidates
        for (i, j, idx, old_color) in candidates[:3]:
            if time.time() > deadline - 2:
                break
            
            colors[idx] = next_color
            backtrack(colors, next_color + 1, depth + 1, max_depth)
            colors[idx] = old_color
    
    # Multiple restarts with backtracking
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Reset to initial state
        for i in range(N-1):
            for j in range(i+1, N):
                colors[edge_idx(i, j)] = i
        
        next_color = 15  # max initial color is 14
        
        random.shuffle(edge_list)
        
        # Use backtracking with limited depth
        backtrack(colors, next_color, 0, 20)
    
    # Final greedy pass to try to add more colors
    for (i, j) in edge_list:
        if time.time() > deadline - 1:
            break
        idx = edge_idx(i, j)
        new_color = 120 - best_count  # Try high colors first
        if can_set_edge(best_result, i, j, new_color):
            best_result[idx] = new_color
            best_count = len(set(best_result))
    
    return best_result

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
