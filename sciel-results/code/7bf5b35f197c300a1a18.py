import time
import random

def construct():
    deadline = time.time() + 110
    
    def has_k222(adj, n):
        """Check if adj (list of int bitmasks) contains K2,2,2.
        K2,2,2 exists iff there exist vertices u,v such that |N(u) ∩ N(v) \ {u,v}| >= 4
        and within that common neighborhood, there exist x,y with |N(x) ∩ N(y) ∩ common| >= 2."""
        for u in range(n):
            nu = adj[u]
            for v_idx in range(u + 1, n):
                nv = adj[v_idx]
                s_mask = nu & nv
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v_idx)
                if s_mask.bit_count() < 4:
                    continue
                # Need to find x, y in s_mask with |N(x) ∩ N(y) ∩ s_mask| >= 2
                x_mask = s_mask
                while x_mask:
                    x_bit = x_mask & -x_mask
                    x_idx = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    nx = adj[x_idx]
                    y_mask = s_mask & (~((1 << (x_idx + 1)) - 1))
                    while y_mask:
                        y_bit = y_mask & -y_mask
                        y_idx = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        ny = adj[y_idx]
                        if (nx & ny & s_mask).bit_count() >= 2:
                            return True
        return False
    
    def count_violations_fast(colors, n):
        """Count total K2,2,2 violations in both color classes. Returns 0 if valid."""
        adj0 = [0] * n
        adj1 = [0] * n
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        # Use early termination: just check if any violation exists
        if has_k222(adj0, n):
            return 1  # at least one violation
        if has_k222(adj1, n):
            return 1
        return 0
    
    def count_violations_detailed(colors, n):
        """Count the number of K2,2,2 subgraphs for gradient-like info."""
        adj0 = [0] * n
        adj1 = [0] * n
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        def count_k222(adj):
            v = 0
            for u in range(n):
                nu = adj[u]
                for v_idx in range(u + 1, n):
                    nv = adj[v_idx]
                    s_mask = nu & nv
                    s_mask &= ~(1 << u)
                    s_mask &= ~(1 << v_idx)
                    if s_mask.bit_count() < 4:
                        continue
                    x_mask = s_mask
                    while x_mask:
                        x_bit = x_mask & -x_mask
                        x_idx = x_bit.bit_length() - 1
                        x_mask ^= x_bit
                        nx = adj[x_idx]
                        y_mask = s_mask & (~((1 << (x_idx + 1)) - 1))
                        while y_mask:
                            y_bit = y_mask & -y_mask
                            y_idx = y_bit.bit_length() - 1
                            y_mask ^= y_bit
                            ny = adj[y_idx]
                            common = nx & ny & s_mask
                            if common.bit_count() >= 2:
                                v += 1
            return v
        
        return count_k222(adj0) + count_k222(adj1)

    # 1. Generate Paley(29)
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    best_n = 29
    best_colors = colors_29
    if count_violations_fast(colors_29, 29) != 0:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30
    n = 30
    n_edges = n * (n - 1) // 2
    base_colors = colors_29.copy()
    
    # Edges involving vertex 29 are the last 29 edges
    # Edge (i, 29) for i in 0..28: index = sum_{k=0}^{i} (29-k) + 0 ... 
    # Actually edge (i,j) has index = i*(n-1) - i*(i-1)//2 + (j-i-1) for i<j
    # For j=29, i from 0 to 28:
    # idx(i,29) = i*29 - i*(i-1)//2 + (29-i-1) = i*29 - i*(i-1)//2 + 28-i
    
    new_edge_indices = []
    for i in range(29):
        idx = i * (n - 1) - i * (i - 1) // 2 + (29 - i - 1)
        new_edge_indices.append(idx)
    
    best_viol_30 = float('inf')
    best_colors_30 = None
    
    # Strategy:
    # 1. Start with random coloring of new edges
    # 2. Local search focusing on new edges first, then all edges
    # 3. Use detailed count for gradient when close
    
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        current_colors = base_colors.copy()
        for idx in new_edge_indices:
            current_colors.append(random.randint(0, 1))
        
        # Use fast check first
        current_viol = count_violations_fast(current_colors, n)
        
        if current_viol == 0:
            return {'n': n, 'colors': current_colors}
        
        # Get detailed count for optimization
        current_detailed = count_violations_detailed(current_colors, n)
        
        if current_detailed < best_viol_30:
            best_viol_30 = current_detailed
            best_colors_30 = current_colors.copy()
        
        # Local Search
        steps = 0
        max_steps = 300
        stuck_count = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            
            # Prefer flipping new edges (indices 406 to 434)
            if random.random() < 0.7:
                edge_idx = random.choice(new_edge_indices)
            else:
                edge_idx = random.randint(0, n_edges - 1)
            
            old_val = current_colors[edge_idx]
            current_colors[edge_idx] = 1 - old_val
            
            new_viol = count_violations_fast(current_colors, n)
            
            if new_viol == 0:
                return {'n': n, 'colors': current_colors}
            
            if new_viol == 0:
                # Check with detailed
                new_detailed = count_violations_detailed(current_colors, n)
            else:
                new_detailed = current_detailed + 1  # approximate, will be corrected
            
            # Use detailed count for comparison
            if new_viol == 0:
                new_detailed = count_violations_detailed(current_colors, n)
            else:
                # For non-zero, we can use a proxy: just count violations
                new_detailed = count_violations_detailed(current_colors, n)
            
            if new_detailed < current_detailed:
                current_detailed = new_detailed
                stuck_count = 0
                if new_detailed < best_viol_30:
                    best_viol_30 = new_detailed
                    best_colors_30 = current_colors.copy()
            else:
                current_colors[edge_idx] = old_val
                stuck_count += 1
                
                if stuck_count > 30:
                    # Jump: flip 2-3 new edges
                    for _ in range(random.randint(2, 3)):
                        j = random.choice(new_edge_indices)
                        current_colors[j] = 1 - current_colors[j]
                    current_detailed = count_violations_detailed(current_colors, n)
                    if current_detailed < best_viol_30:
                        best_viol_30 = current_detailed
                        best_colors_30 = current_colors.copy()
                    stuck_count = 0

    return {'n': best_n, 'colors': best_colors}
