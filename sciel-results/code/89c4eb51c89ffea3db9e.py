"""Rewritten program: minimize largest no-three-collinear subset of 20 points."""
import random
import math
import time
from collections import defaultdict

def construct():
    random.seed(42)
    
    def normalize_line(p1, p2):
        """Return normalized (a, b, c) for line through p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        g = math.gcd(a, math.gcd(b, c))
        if g != 0:
            a //= g
            b //= g
            c //= g
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def get_triples(points):
        """Return list of collinear triples (as tuples of indices)."""
        n = len(points)
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = normalize_line(points[i], points[j])
                lines[key].add(i)
                lines[key].add(j)
        
        triples = []
        for key, pts in lines.items():
            if len(pts) == 3:
                p_list = sorted(pts)
                triples.append((p_list[0], p_list[1], p_list[2]))
        return triples

    def is_valid(points):
        """Check distinct, bounds, and no 4 collinear."""
        if len(set(points)) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        
        # Check no 4 collinear
        n = len(points)
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = normalize_line(points[i], points[j])
                lines[key].add(i)
                lines[key].add(j)
        
        for pts in lines.values():
            if len(pts) >= 4:
                return False
        return True

    def calc_mis(points):
        """Calculate Maximum Independent Set size for the hypergraph of collinear triples."""
        n = len(points)
        triples = get_triples(points)
        
        # Precompute conflict pairs for each vertex
        # conflict_pairs[v] = list of (u, w) such that {v, u, w} is a triple
        conflict_pairs = [[] for _ in range(n)]
        for t in triples:
            i, j, k = t
            conflict_pairs[i].append((j, k))
            conflict_pairs[j].append((i, k))
            conflict_pairs[k].append((i, j))
            
        # Heuristic: order vertices by degree (number of conflict pairs)
        degrees = [len(cp) for cp in conflict_pairs]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        best = 0
        
        def dfs(available_mask, current_mask, current_size):
            nonlocal best
            
            # Pruning
            if current_size + bin(available_mask).count('1') <= best:
                return
            
            if available_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick next vertex from order that is in available_mask
            v = -1
            for idx in order:
                if available_mask & (1 << idx):
                    v = idx
                    break
            
            if v == -1:
                return
            
            v_bit = 1 << v
            
            # Option 1: Exclude v
            dfs(available_mask & ~v_bit, current_mask, current_size)
            
            # Option 2: Include v
            # Check if v conflicts with current_mask
            can_include = True
            for u, w in conflict_pairs[v]:
                if (current_mask & (1 << u)) and (current_mask & (1 << w)):
                    can_include = False
                    break
            
            if can_include:
                # Update available mask
                # Remove v
                new_available = available_mask & ~v_bit
                # Remove any u that would form a triple with v and some w in current_mask
                for u, w in conflict_pairs[v]:
                    if current_mask & (1 << w):
                        new_available &= ~(1 << u)
                    if current_mask & (1 << u):
                        new_available &= ~(1 << w)
                        
                dfs(new_available, current_mask | v_bit, current_size + 1)
                
        dfs((1 << n) - 1, 0, 0)
        return best

    # Generate initial valid configuration
    def generate_initial():
        while True:
            points = []
            seen = set()
            while len(points) < 20:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
            if is_valid(points):
                return points

    points = generate_initial()
    current_score = calc_mis(points)
    
    best_points = list(points)
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.99
    
    start_time = time.time()
    time_limit = 110.0
    
    no_improve = 0
    max_no_improve = 1000
    
    candidates = set()
    for _ in range(1000):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        candidates.add((x, y))
    candidates = list(candidates)
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
            
        idx = random.randint(0, 19)
        old_point = points[idx]
        new_point = random.choice(candidates)
        
        if new_point == old_point:
            continue
            
        new_points = list(points)
        new_points[idx] = new_point
        
        if not is_valid(new_points):
            continue
            
        new_score = calc_mis(new_points)
        
        accepted = False
        if new_score < current_score:
            points = new_points
            current_score = new_score
            accepted = True
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
                    accepted = True
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
                    accepted = True
            
            if not accepted:
                no_improve += 1
                
        T *= alpha
        if T < T_min:
            T = T_min
            if no_improve > 500:
                T = 0.5
                no_improve = 0

    return [[p[0], p[1]] for p in best_points]
