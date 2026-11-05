import time
import random
import itertools
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
    
    num_k4s = len(k4s)
    
    # Precompute for fast violation counting
    # For each edge, store the K4 indices it belongs to
    # For each K4, store its 6 edge indices
    
    def count_violations(colors):
        """Count number of K4s with < 5 distinct colors. Fast using bit masks."""
        violations = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edges[k4_idx]
            mask = 0
            for e in e_indices:
                mask |= (1 << colors[e])
            if mask.bit_count() < 5:
                violations += 1
        return violations
    
    def check_k4_valid(colors, k4_idx):
        """Check if a specific K4 is valid."""
        e_indices = k4_edges[k4_idx]
        mask = 0
        for e in e_indices:
            mask |= (1 << colors[e])
        return mask.bit_count() >= 5
    
    def get_violated_k4s_for_edge(colors, edge_idx, new_color):
        """Get list of K4s that would be violated if we set edge_idx to new_color."""
        violated = []
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edges[k4_idx]
            mask = (1 << new_color)
            for e in e_indices:
                if e != edge_idx:
                    mask |= (1 << colors[e])
            if mask.bit_count() < 5:
                violated.append(k4_idx)
        return violated
    
    # GF(4) algebraic construction for initial coloring
    # GF(4) = {0, 1, w, w^2} where w^2 = w + 1
    # Represent elements as 2-bit: 0->00, 1->01, w->10, w^2->11
    # Addition in GF(4) is XOR
    # Multiplication: w^2 = w+1, w^3 = 1
    # 16 vertices = GF(4)^2, indexed 0..15
    # Vertex i = (i >> 2, i & 3) in GF(4)^2
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        # GF(4) multiplication
        # 0*x=0, 1*x=x, w*x: w*0=0, w*1=w, w*w=w^2, w*w^2=1
        # w^2*x: w^2*0=0, w^2*1=w^2, w^2*w=1, w^2*w^2=w
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        # a, b in {2, 3} (w, w^2)
        # w*w = w^2 (2*2=3), w*w^2 = 1 (2*3=1), w^2*w^2 = w (3*3=2)
        if a == 2 and b == 2:
            return 3
        if a == 2 and b == 3:
            return 1
        if a == 3 and b == 2:
            return 1
        if a == 3 and b == 3:
            return 2
        return 0  # shouldn't reach here
    
    def gf4_inv(a):
        if a == 0:
            return 0
        if a == 1:
            return 1
        if a == 2:  # w
            return 3  # w^2
        if a == 3:  # w^2
            return 2  # w
    
    # Slope-based coloring: for vertices u, v in GF(4)^2
    # slope = (v_y - u_y) / (v_x - u_x) if v_x != u_x, else infinity
    def slope_color(u, v):
        """Compute slope-based color for edge (u,v) in GF(4)^2."""
        ux, uy = u >> 2, u & 3
        vx, vy = v >> 2, v & 3
        dx = gf4_add(ux, vx)  # vx - ux = vx + ux in GF(4)
        dy = gf4_add(uy, vy)
        if dx == 0:
            return 4  # infinity slope
        else:
            return gf4_mul(dy, gf4_inv(dx))  # 0,1,2,3
    
    # Build initial coloring using slope
    slope_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            slope_colors.append(slope_color(i, j))
    
    # This gives 5 colors (0-4). Check validity.
    slope_valid = True
    for k4_idx in range(num_k4s):
        e_indices = k4_edges[k4_idx]
        mask = 0
        for e in e_indices:
            mask |= (1 << slope_colors[e])
        if mask.bit_count() < 5:
            slope_valid = False
            break
    
    # Also try a perturbed version: use slope + some function of vertices
    # to break ties and get more colors
    def make_perturbed_coloring(perturb_func, num_colors):
        """Create a coloring based on slope with perturbation."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                base = slope_color(i, j)
                c = (base + perturb_func(i, j)) % num_colors
                colors.append(c)
        return colors
    
    # Baseline: simple valid coloring
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Verify baseline
    valid = True
    for k4_idx in range(num_k4s):
        e_indices = k4_edges[k4_idx]
        k4_colors = [baseline[e] for e in e_indices]
        if len(set(k4_colors)) < 5:
            valid = False
            break
    
    best_result = None
    best_colors_used = 100
    
    if valid:
        best_result = baseline[:]
        best_colors_used = len(set(baseline))
    
    # Simulated annealing style local search
    def local_search_sa(colors, max_iters=5000):
        """Simulated annealing to minimize number of colors while maintaining validity."""
        current = colors[:]
        current_violations = count_violations(current)
        current_colors_used = len(set(current))
        
        if current_violations == 0:
            best = current[:]
            best_used = current_colors_used
        else:
            best = current[:]
            best_used = current_colors_used + 1000  # penalize violations
        
        T = 5.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = current[e_idx]
            
            # Pick a new color (biased toward reducing color count)
            # Try colors that are already used, or a new one
            used_colors = set(current)
            if random.random() < 0.7:
                # Pick from existing colors
                new_color = random.choice(list(used_colors))
            else:
                # Try a random color in range
                new_color = random.randint(0, max(used_colors) + 1)
            
            if new_color == old_color:
                continue
            
            # Apply change
            current[e_idx] = new_color
            
            # Count violations efficiently: only check K4s containing e_idx
            new_violations = 0
            for k4_idx in edge_to_k4s[e_idx]:
                e_indices = k4_edges[k4_idx]
                mask = 0
                for e in e_indices:
                    mask |= (1 << current[e])
                if mask.bit_count() < 5:
                    new_violations += 1
            
            new_colors_used = len(set(current))
            
            # Score: violations * 100 + colors_used
            old_score = current_violations * 100 + current_colors_used
            new_score = new_violations * 100 + new_colors_used
            
            delta = new_score - old_score
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_violations = new_violations
                current_colors_used = new_colors_used
                
                if new_violations == 0 and new_colors_used < best_used:
                    best = current[:]
                    best_used = new_colors_used
            else:
                # Reject
                current[e_idx] = old_color
            
            T *= cooling
        
        return best, best_used
    
    # Edge ordering for greedy
    edge_degree = [len(edge_to_k4s[e]) for e in range(num_edges)]
    base_order = sorted(range(num_edges), key=lambda e: -edge_degree[e])
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Slope-based with perturbation
    for perturb_seed in range(5):
        random.seed(perturb_seed)
        for num_colors in [13, 14, 15]:
            # Random perturbation function
            perturb_map = {}
            for i in range(n):
                for j in range(i + 1, n):
                    perturb_map[(i, j)] = random.randint(0, num_colors - 1)
            
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    base = slope_color(i, j)
                    c = (base + perturb_map[(i, j)]) % num_colors
                    colors.append(c)
            strategies.append(colors)
    
    # Strategy 2: Random with structure
    for seed in range(10):
        random.seed(seed)
        for num_colors in [13, 14, 15]:
            # Start with slope and add random noise
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    base = slope_color(i, j)
                    if random.random() < 0.5:
                        c = random.randint(0, num_colors - 1)
                    else:
                        c = (base + random.randint(0, num_colors - 1)) % num_colors
                    colors.append(c)
            strategies.append(colors)
    
    # Strategy 3: Pure random
    for seed in range(10):
        random.seed(seed)
        for num_colors in [14, 15]:
            colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
            strategies.append(colors)
    
    # Run local search on each strategy
    for strat_idx, colors in enumerate(strategies):
        if time.time() > deadline - 10:
            break
        
        refined, refined_used = local_search_sa(colors, max_iters=3000)
        
        # Verify
        if count_violations(refined) == 0:
            actual_used = len(set(refined))
            if actual_used < best_colors_used:
                best_colors_used = actual_used
                best_result = refined[:]
                if best_colors_used <= 13:
                    return best_result
    
    # Greedy construction with backtracking for tighter color bounds
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 15:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            colors = [-1] * num_edges
            success = True
            
            edge_order = base_order[:]
            random.shuffle(edge_order)
            
            for e_idx in edge_order:
                color_options = list(range(target_C))
                random.shuffle(color_options)
                
                assigned = False
                for c in color_options:
                    # Quick check: only K4s containing this edge
                    valid = True
                    for k4_idx in edge_to_k4s[e_idx]:
                        e_indices = k4_edges[k4_idx]
                        # Check if K4 is fully colored
                        all_colored = True
                        for e in e_indices:
                            if e == e_idx:
                                continue
                            if colors[e] < 0:
                                all_colored = False
                                break
                        if not all_colored:
                            continue
                        # Check distinct colors
                        mask = (1 << c)
                        for e in e_indices:
                            if e != e_idx:
                                mask |= (1 << colors[e])
                        if mask.bit_count() < 5:
                            valid = False
                            break
                    if valid:
                        colors[e_idx] = c
                        assigned = True
                        break
                
                if not assigned:
                    success = False
                    break
            
            if not success:
                continue
            
            # Verify
            if count_violations(colors) == 0:
                num_used = len(set(colors))
                if num_used < best_colors_used:
                    best_colors_used = num_used
                    best_result = colors[:]
                    if best_colors_used <= 13:
                        return best_result
                
                # Try to reduce further
                refined, refined_used = local_search_sa(colors, max_iters=2000)
                if count_violations(refined) == 0:
                    actual_used = len(set(refined))
                    if actual_used < best_colors_used:
                        best_colors_used = actual_used
                        best_result = refined[:]
                        if best_colors_used <= 13:
                            return best_result
    
    if best_result is not None:
        return best_result
    else:
        return baseline
