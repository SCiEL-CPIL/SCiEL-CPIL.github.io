import time
import sys

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Returns a dict with 'n' and 'colors'.
    """
    # Deadline: 110 seconds to be safe against the 120s kill.
    deadline = time.time() + 110
    
    # --- Step 1: Generate the base K32 coloring (Greenwood-Gleason / Clebsch) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)
    
    n_current = 32
    colors = base_colors[:]
    
    # Precompute adjacency bitmasks for each color
    # adj_color[c][i] is a bitmask of vertices j such that color(i,j) == c
    # We use Python integers as bitsets.
    
    def build_adj_colors(n, cols):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate index in cols
                # Row i has edges (i, i+1) ... (i, n-1)
                # Index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
                # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
                idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                c = cols[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    adj = build_adj_colors(n_current, colors)
    
    max_n = n_current
    best_colors = colors[:]
    
    # Backtracking search
    # We will add vertices one by one.
    # State: current_n, adj (list of 4 lists of bitmasks), colors (flat list)
    
    def backtrack(current_n, adj, cols):
        nonlocal max_n, best_colors
        
        if time.time() > deadline:
            return False
        
        if current_n > max_n:
            max_n = current_n
            best_colors = cols[:]
            
        if current_n >= 62:
            return True
            
        # Try to add vertex current_n
        # Edges to be assigned: (0, current_n), ..., (current_n-1, current_n)
        
        # We maintain S_c: bitmask of vertices i < current_n assigned color c to (i, current_n)
        # Constraint: For each c, the set of vertices in S_c must be an independent set in the color-c subgraph.
        # i.e., for any u, v in S_c, color(u, v) != c.
        # Equivalently, adj[c][u] & S_c should not contain v.
        # Actually, we can check this incrementally.
        
        new_vertex_adj = [0, 0, 0, 0] # Bitmasks for the new vertex's connections in each color
        
        def assign(idx, S, new_adj):
            nonlocal max_n, best_colors
            
            if time.time() > deadline:
                return False
                
            if idx == current_n:
                # Successfully assigned all edges for new vertex
                # Update adj and cols
                
                # 1. Update cols
                # The new edges are (i, current_n) for i in 0..current_n-1.
                # In the flat list for K_{current_n+1}, where do they go?
                # It's easier to just reconstruct the flat list from the matrix or insert.
                # Given the size (<= 62), reconstruction is fast.
                new_cols = []
                for i in range(current_n):
                    # Append edges (i, i+1) ... (i, current_n-1) from old cols
                    start_idx = i * (current_n - 1) - i * (i - 1) // 2
                    end_idx = start_idx + (current_n - 1 - i)
                    new_cols.extend(cols[start_idx:end_idx])
                    # Append new edge (i, current_n)
                    # The color is the one assigned to idx=i in this step.
                    # We stored it in new_adj? No, new_adj is bitmasks.
                    # We need to retrieve the color for (i, current_n).
                    # We can check which S_c contains i.
                    c = 0
                    while c < 4:
                        if (S[c] >> i) & 1:
                            break
                        c += 1
                    new_cols.append(c)
                
                # 2. Update adj
                # We need to add a new row to each adj[c].
                # And set bits in existing rows.
                new_adj = []
                for c in range(4):
                    new_row = [0] * (current_n + 1)
                    # Copy old rows
                    for i in range(current_n):
                        new_row[i] = adj[c][i]
                        # If i is connected to current_n with color c, set bit current_n in adj[c][i]
                        if (S[c] >> i) & 1:
                            new_row[i] |= (1 << current_n)
                    # The new vertex's adjacency in color c is S[c]
                    new_row[current_n] = S[c]
                    new_adj.append(new_row)
                
                # Recurse
                return backtrack(current_n + 1, new_adj, new_cols)
            
            # Try colors 0, 1, 2, 3 for edge (idx, current_n)
            # Optimization: We can try to prune based on previous assignments.
            for c in range(4):
                # Check if adding idx to S[c] violates independence
                # If idx is in S[c], it's already there? No, idx is new.
                # Check if any existing vertex u in S[c] is connected to idx with color c.
                # i.e., is (adj[c][idx] & S[c]) non-zero?
                if adj[c][idx] & S[c]:
                    continue
                
                # If valid, add idx to S[c]
                S[c] |= (1 << idx)
                if assign(idx + 1, S, new_adj):
                    return True
                S[c] &= ~(1 << idx)
                
            return False
            
        # Initial S for new vertex
        S = [0, 0, 0, 0]
        return assign(0, S, new_vertex_adj)
        
    backtrack(n_current, adj, colors)
    
    # Safety check: if max_n is 32, ensure colors are valid K32
    if max_n == 32:
        best_colors = colors[:]

    return {'n': max_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
