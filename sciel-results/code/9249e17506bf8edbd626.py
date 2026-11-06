import random
import itertools
import numpy as np
import time
import math

def _check_no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = yj - yi
                b = xi - xj
                c = xj*yi - xi*yj
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    xl, yl = points[l]
                    if a * xl + b * yl + c == 0:
                        return False
    return True

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    if n == 0:
        return 0
        
    line_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    mask |= (1 << k)
            line_mask[i][j] = mask
            line_mask[j][i] = mask

    best = [0]
    
    def dfs(idx, current_mask, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            best[0] = max(best[0], current_size)
            return
        
        dfs(idx + 1, current_mask, current_size)
        
        can_take = True
        m = current_mask
        while m:
            lsb = m & -m
            a_idx = lsb.bit_length() - 1
            if line_mask[a_idx][idx] & current_mask:
                can_take = False
                break
            m ^= lsb
            
        if can_take:
            dfs(idx + 1, current_mask | (1 << idx), current_size + 1)

    dfs(0, 0, 0)
    return best[0]

def construct():
    """
    Genetic algorithm to find 20 points with no four collinear
    that minimizes the largest no-three-collinear subset.
    """
    random.seed(42)
    
    def is_valid(points):
        if len(set(tuple(p) for p in points)) != 20:
            return False
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
            return False
        if not _check_no_four_collinear(points):
            return False
        return True
    
    def initial_population(size=20):
        """Generate initial population of valid configurations."""
        pop = []
        # Method 1: Systematically perturbed grid
        for trial in range(200):
            pts = []
            for i in range(5):
                for j in range(4):
                    x = i * 15 + (j * j + i) % 7 - 3
                    y = j * 15 + (i * i + j) % 7 - 3
                    x += random.randint(-1, 1)
                    y += random.randint(-1, 1)
                    pts.append([x, y])
            if is_valid(pts):
                pop.append(pts)
                if len(pop) >= size:
                    break
        # Method 2: Random valid configurations
        while len(pop) < size:
            pts = []
            for _ in range(20):
                x = random.randint(-60, 60)
                y = random.randint(-60, 60)
                pts.append([x, y])
            if is_valid(pts):
                pop.append(pts)
        return pop
    
    def crossover(p1, p2):
        """Uniform crossover."""
        return [p1[i][:] if random.random() < 0.5 else p2[i][:] for i in range(20)]
    
    def mutate(points, rate=0.35):
        """Mutation with variable step size."""
        new_pts = [p[:] for p in points]
        for i in range(20):
            if random.random() < rate:
                step = random.choice([3, 5, 8, 12])
                dx = random.randint(-step, step)
                dy = random.randint(-step, step)
                new_pts[i][0] = max(-100, min(100, new_pts[i][0] + dx))
                new_pts[i][1] = max(-100, min(100, new_pts[i][1] + dy))
        return new_pts
    
    def tournament_select(pop, scores, k=3):
        """Tournament selection."""
        idxs = random.sample(range(len(pop)), k)
        return pop[min(idxs, key=lambda i: scores[i])][:]
    
    # Initialize
    pop = initial_population(25)
    scores = [_score(p) for p in pop]
    
    best_idx = min(range(len(pop)), key=lambda i: scores[i])
    best_points = [p[:] for p in pop[best_idx]]
    best_score = scores[best_idx]
    
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        # Elitism: keep top 5
        sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i])
        new_pop = [pop[i][:] for i in sorted_idx[:5]]
        new_scores = [scores[i] for i in sorted_idx[:5]]
        
        # Generate offspring to fill population
        while len(new_pop) < len(pop):
            p1 = tournament_select(pop, scores)
            p2 = tournament_select(pop, scores)
            child = crossover(p1, p2)
            child = mutate(child)
            if is_valid(child):
                new_pop.append(child)
                new_scores.append(_score(child))
        
        pop = new_pop
        scores = new_scores
        
        # Update best
        best_idx = min(range(len(pop)), key=lambda i: scores[i])
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_points = [p[:] for p in pop[best_idx]]
    
    return best_points
