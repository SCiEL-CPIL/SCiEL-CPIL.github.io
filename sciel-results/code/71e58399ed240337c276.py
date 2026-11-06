import random
import itertools
from collections import defaultdict
import math

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    """
    random.seed(42)
    
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        a = dy
        b = -dx
        c = -(dy * p1[0] - dx * p1[1])
        g = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
        if g == 0:
            return None
        a //= g
        b //= g
        c //= g
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def get_lines_for_point(p, points, exclude_idx=-1):
        """Get lines passing through p and at least one other point in points."""
        lines = set()
        for i, q in enumerate(points):
            if i == exclude_idx:
                continue
            if p == q:
                continue
            k = line_key(p, q)
            if k:
                lines.add(k)
        return lines

    def count_triples(points):
        """Count collinear triples and return masks for B&B."""
        n = len(points)
        triples = []
        line_points = defaultdict(list)
        
        # Group points by line
        # To avoid O(N^3), we can use the fact that we only care about lines with >= 3 points.
        # But we need to find them.
        # For each pair (i, j), find line, add to dict.
        # Then for each line with k points, add C(k,3) triples.
        
        line_map = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                k = line_key(points[i], points[j])
                if k:
                    line_map[k].append(i)
                    line_map[k].append(j)
        
        # This is inefficient because we add i and j multiple times.
        # Better: 
        # For each i, for each j > i, get line. 
        # Actually, just iterate all pairs, store line -> set of indices.
        
        line_indices = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                k = line_key(points[i], points[j])
                if k:
                    line_indices[k].add(i)
                    line_indices[k].add(j)
        
        triple_masks = [[] for _ in range(n)]
        triple_count = 0
        for k, indices in line_indices.items():
            if len(indices) >= 3:
                # Generate all triples from this set
                idx_list = list(indices)
                for a in range(len(idx_list)):
                    for b in range(a+1, len(idx_list)):
                        for c in range(b+1, len(idx_list)):
                            i, j, l = idx_list[a], idx_list[b], idx_list[c]
                            # Add to masks
                            m_jl = (1 << j) | (1 << l)
                            m_il = (1 << i) | (1 << l)
                            m_ij = (1 << i) | (1 << j)
                            triple_masks[i].append(m_jl)
                            triple_masks[j].append(m_il)
                            triple_masks[l].append(m_ij)
                            triple_count += 1
                            
        return triple_masks, triple_count

    def exact_mis(triple_masks):
        n = len(triple_masks)
        best = 0
        
        # Sort vertices by degree (number of triples they are in) to help pruning?
        # Or just use bitmask order.
        
        def dfs(available, current, size):
            nonlocal best
            if size + bin(available).count('1') <= best:
                return
            
            # If no available vertices, update best
            if available == 0:
                if size > best:
                    best = size
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in the *remaining* graph?
            # For simplicity, pick lowest index.
            v_bit = available & (-available)
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            dfs(available & ~v_bit, current, size)
            
            # Option 2: Include v
            # Check if v can be added to current
            # v can be added if for all m in triple_masks[v], (current & m) != m
            can_add = True
            for m in triple_masks[v]:
                if (current & m) == m:
                    can_add = False
                    break
            
            if can_add:
                dfs(available & ~v_bit, current | v_bit, size + 1)
        
        dfs((1 << n) - 1, 0, 0)
        return best

    def is_valid_config(points):
        if len(set(points)) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        # Check no 4 collinear
        line_indices = defaultdict(set)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                k = line_key(points[i], points[j])
                if k:
                    line_indices[k].add(i)
                    line_indices[k].add(j)
        for k, indices in line_indices.items():
            if len(indices) >= 4:
                return False
        return True

    # Initial configuration: Random
    # We want to start with a "dense" configuration to have low MIS.
    # Try a grid-like structure.
    # 5x4 grid: x in 0..4, y in 0..3
    # Scale it to fit in 100?
    # Or just random and let SA find the structure.
    
    points = []
    seen = set()
    # Try to generate a grid-like set
    # 5 rows, 4 cols
    # x = i * 10, y = j * 10
    base_points = []
    for i in range(5):
        for j in range(4):
            base_points.append((i*10, j*10))
            
    # Check if valid
    if is_valid_config(base_points):
        points = base_points
    else:
        # Fallback to random
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p not in seen:
                seen.add(p)
                points.append(p)
        # Ensure validity
        while not is_valid_config(points):
            # Perturb a random point
            idx = random.randint(0, 19)
            points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
            seen = set(points)
            # Remove duplicates by re-generating
            # This is messy. Let's just regenerate if invalid.
            points = []
            seen = set()
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    points.append(p)

    current_masks, _ = count_triples(points)
    current_score = exact_mis(current_masks)
    
    best_points = list(points)
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    max_iterations = 50000
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iterations):
        if no_improve > max_no_improve:
            break
            
        idx = random.randint(0, 19)
        old_point = points[idx]
        new_point = (random.randint(-100, 100), random.randint(-100, 100))
        
        if new_point == old_point:
            continue
            
        # Check if new_point is already in points
        if new_point in points:
            continue
            
        new_points = list(points)
        new_points[idx] = new_point
        
        # Check validity (no 4 collinear)
        if not is_valid_config(new_points):
            continue
            
        new_masks, _ = count_triples(new_points)
        new_score = exact_mis(new_masks)
        
        if new_score < current_score:
            points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = list(points)
            no_improve = 0
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.5:
                    points = new_points
                    current_score = new_score
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
            
            no_improve += 1
            
        T *= alpha
        if T < T_min:
            T = T_min

    return [[p[0], p[1]] for p in best_points]
