import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    
    # Initialize with (i+j) % 17
    colors = [(i + j) % 17 for i, j in edges]
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s and the edges in each
    k4_list = []
    for a in range(n-3):
        for b in range(a+1, n-2):
            for c in range(b+1, n-1):
                for d in range(c+1, n):
                    k4_edges = [
                        edge_idx[(a,b)],
                        edge_idx[(a,c)],
                        edge_idx[(a,d)],
                        edge_idx[(b,c)],
                        edge_idx[(b,d)],
                        edge_idx[(c,d)]
                    ]
                    k4_list.append(k4_edges)
    
    num_k4s = len(k4_list)
    
    # For each edge, precompute which K4s contain it
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_i, k4_edges in enumerate(k4_list):
        for e_idx in k4_edges:
            edge_k4s[e_idx].append(k4_i)
    
    # Count distinct colors in each K4
    k4_color_counts = [0] * num_k4s
    k4_color_sets = [None] * num_k4s
    
    def compute_k4_stats():
        for k4_i in range(num_k4s):
            k4c = [colors[e] for e in k4_list[k4_i]]
            s = set(k4c)
            k4_color_sets[k4_i] = s
            k4_color_counts[k4_i] = len(s)
    
    compute_k4_stats()
    
    # Count bad K4s
    bad_count = sum(1 for c in k4_color_counts if c < 5)
    
    def update_k4_after_change(edge_idx_changed, old_color, new_color):
        """Update K4 stats after changing one edge. Returns delta_bad."""
        nonlocal bad_count
        delta_bad = 0
        for k4_i in edge_k4s[edge_idx_changed]:
            old_set = k4_color_sets[k4_i]
            old_count = k4_color_counts[k4_i]
            
            # Remove old_color from set, add new_color
            new_set = set(old_set)
            new_set.discard(old_color)
            new_set.add(new_color)
            new_count = len(new_set)
            
            if old_count < 5 and new_count >= 5:
                delta_bad -= 1
            elif old_count >= 5 and new_count < 5:
                delta_bad += 1
            
            k4_color_sets[k4_i] = new_set
            k4_color_counts[k4_i] = new_count
            colors[edge_idx_changed] = new_color  # temporarily set
        
        bad_count += delta_bad
        return delta_bad
    
    def revert_k4_after_change(edge_idx_changed, old_color, new_color, delta_bad):
        """Revert K4 stats after a failed change."""
        nonlocal bad_count
        for k4_i in edge_k4s[edge_idx_changed]:
            new_set = k4_color_sets[k4_i]
            new_count = k4_color_counts[k4_i]
            
            old_set = set(new_set)
            old_set.discard(new_color)
            old_set.add(old_color)
            old_count = len(old_set)
            
            k4_color_sets[k4_i] = old_set
            k4_color_counts[k4_i] = old_count
        
        colors[edge_idx_changed] = old_color
        bad_count -= delta_bad
    
    current_max = max(colors)
    best_bad = bad_count
    
    random.seed(42)
    
    # Local search: try to reduce number of colors used while keeping bad_count == 0
    # Strategy: iterate through edges, try to assign lower colors
    while time.time() < deadline:
        improved = False
        # Try each edge in random order for diversity
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for idx in edge_order:
            if time.time() > deadline:
                break
            
            old_color = colors[idx]
            
            # Try to reduce color: try colors from 0 up to old_color-1
            # Also try to reduce global max
            target_max = current_max
            for new_color in range(0, min(old_color, target_max)):
                if new_color == old_color:
                    continue
                
                # Apply change
                delta_bad = update_k4_after_change(idx, old_color, new_color)
                
                if bad_count == 0:
                    # Valid! Check if we reduced max
                    new_max = max(colors)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    # Keep the change
                    break
                else:
                    # Revert
                    revert_k4_after_change(idx, old_color, new_color, delta_bad)
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations to escape local minima
            if bad_count == 0:
                # Try swapping colors between two edges
                for _ in range(1000):
                    if time.time() > deadline:
                        break
                    i1 = random.randint(0, num_edges - 1)
                    i2 = random.randint(0, num_edges - 1)
                    if i1 == i2:
                        continue
                    
                    c1 = colors[i1]
                    c2 = colors[i2]
                    if c1 == c2:
                        continue
                    
                    # Swap colors
                    colors[i1] = c2
                    colors[i2] = c1
                    
                    # Update K4 stats for both edges
                    # This is more complex, so let's just recompute
                    # Actually, let's do it properly
                    k4s_i1 = edge_k4s[i1]
                    k4s_i2 = edge_k4s[i2]
                    
                    # Apply swap to i1
                    delta1 = 0
                    for k4_i in k4s_i1:
                        old_set = k4_color_sets[k4_i]
                        old_count = k4_color_counts[k4_i]
                        new_set = set(old_set)
                        new_set.discard(c1)
                        new_set.add(c2)
                        new_count = len(new_set)
                        if old_count < 5 and new_count >= 5:
                            delta1 -= 1
                        elif old_count >= 5 and new_count < 5:
                            delta1 += 1
                        k4_color_sets[k4_i] = new_set
                        k4_color_counts[k4_i] = new_count
                    
                    # Apply swap to i2 (note: i1's change may have affected some of i2's K4s)
                    delta2 = 0
                    for k4_i in k4s_i2:
                        old_set = k4_color_sets[k4_i]
                        old_count = k4_color_counts[k4_i]
                        new_set = set(old_set)
                        new_set.discard(c2)
                        new_set.add(c1)
                        new_count = len(new_set)
                        if old_count < 5 and new_count >= 5:
                            delta2 -= 1
                        elif old_count >= 5 and new_count < 5:
                            delta2 += 1
                        k4_color_sets[k4_i] = new_set
                        k4_color_counts[k4_i] = new_count
                    
                    total_delta = delta1 + delta2
                    if bad_count + total_delta == 0:
                        bad_count += total_delta
                        improved = True
                        break
                    else:
                        # Revert both
                        # Revert i2
                        for k4_i in k4s_i2:
                            old_set = k4_color_sets[k4_i]
                            old_count = k4_color_counts[k4_i]
                            new_set = set(old_set)
                            new_set.discard(c1)
                            new_set.add(c2)
                            new_count = len(new_set)
                            k4_color_sets[k4_i] = new_set
                            k4_color_counts[k4_i] = new_count
                        
                        # Revert i1
                        for k4_i in k4s_i1:
                            old_set = k4_color_sets[k4_i]
                            old_count = k4_color_counts[k4_i]
                            new_set = set(old_set)
                            new_set.discard(c2)
                            new_set.add(c1)
                            new_count = len(new_set)
                            k4_color_sets[k4_i] = new_set
                            k4_color_counts[k4_i] = new_count
                        
                        colors[i1] = c1
                        colors[i2] = c2
            else:
                # Bad K4s exist, try to fix them
                # Find a bad K4 and try to change one of its edges
                bad_k4s = [i for i in range(num_k4s) if k4_color_counts[i] < 5]
                if bad_k4s:
                    k4_i = random.choice(bad_k4s)
                    for e_idx in k4_list[k4_i]:
                        old_color = colors[e_idx]
                        # Try random new colors
                        for _ in range(20):
                            new_color = random.randint(0, 119)
                            if new_color == old_color:
                                continue
                            delta_bad = update_k4_after_change(e_idx, old_color, new_color)
                            if bad_count == 0:
                                improved = True
                                break
                            else:
                                revert_k4_after_change(e_idx, old_color, new_color, delta_bad)
                        if improved:
                            break
                if not improved:
                    # Random restart with small perturbation
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        e_idx = random.randint(0, num_edges - 1)
                        old_color = colors[e_idx]
                        new_color = random.randint(0, 119)
                        delta_bad = update_k4_after_change(e_idx, old_color, new_color)
                        if bad_count <= best_bad:
                            best_bad = bad_count
                            improved = True
                        else:
                            revert_k4_after_change(e_idx, old_color, new_color, delta_bad)
                        if improved:
                            break
                if not improved:
                    break
    
    # Final check and cleanup
    # Ensure all colors are in range 0..119
    for i in range(num_edges):
        if colors[i] < 0 or colors[i] > 119:
            colors[i] = colors[i] % 120
    
    return colors
