import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Grid dimensions for compact search space
    # A 31x31 grid provides enough variety for 20 points while keeping neighborhood search fast
    GRID = 31
    HALF = GRID // 2  # 15
    # Coordinates range from -15 to 15
    
    # Precompute all points in the grid
    all_pts = []
    for x in range(-HALF, HALF + 1):
        for y in range(-HALF, HALF + 1):
            all_pts.append((x, y))
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Combined line analysis: returns (has_4, heuristic_score)
    def analyze_lines(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                g = gcd(abs(A), abs(B))
                if g != 0:
                    g = gcd(g, abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        
        has_4 = False
        score = 0
        for pts_on_line in line_counts.values():
            cnt = len(pts_on_line)
            if cnt >= 4:
                has_4 = True
                break
            if cnt >= 3:
                score += cnt
        
        return has_4, score
    
    def has_4_collinear(points):
        h4, _ = analyze_lines(points)
        return h4

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    def max_no_3_collinear(points):
        n = len(points)
        
        # Precompute conflict pairs as tuples for faster lookup
        conflict = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        conflict[i].add((j, k))
                        conflict[j].add((i, k))
                        conflict[k].add((i, j))
        
        best = 0
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')
            
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip point start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include point start_idx
            valid = True
            for (j, k) in conflict[start_idx]:
                if (S_mask & (1 << j)) and (S_mask & (1 << k)):
                    valid = False
                    break
            
            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)
                
        backtrack(0, 0)
        return best

    def heuristic_score(points):
        _, score = analyze_lines(points)
        return score

    # Simulated Annealing with improved moves
    def simulated_annealing():
        # Find a valid initial configuration
        current_pts = None
        for _ in range(200):
            candidate = random.sample(all_pts, 20)
            h4, score = analyze_lines(candidate)
            if not h4:
                current_pts = candidate
                break
        
        if current_pts is None:
            current_pts = random.sample(all_pts, 20)
            while has_4_collinear(current_pts):
                idx = random.randint(0, 19)
                current_pts[idx] = random.choice(all_pts)
        
        current_score = heuristic_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        best_exact_score = max_no_3_collinear(best_pts)
        
        T = 10.0
        T_min = 0.01
        alpha = 0.9985
        
        steps = 0
        while T > T_min:
            if time.time() > deadline - 5:
                break
            
            # Choose move type: 70% single-point replacement, 30% swap
            if random.random() < 0.7:
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = random.choice(all_pts)
                
                if new_p == old_p or new_p in current_pts:
                    continue
                
                current_pts[idx] = new_p
                h4, new_score = analyze_lines(current_pts)
                
                if h4:
                    current_pts[idx] = old_p
                    continue
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = current_pts[:]
                        if steps % 20 == 0:
                            exact = max_no_3_collinear(current_pts)
                            if exact < best_exact_score:
                                best_exact_score = exact
                else:
                    current_pts[idx] = old_p
            else:
                i, j = random.sample(range(20), 2)
                pi, pj = current_pts[i], current_pts[j]
                new_i = random.choice(all_pts)
                new_j = random.choice(all_pts)
                
                if new_i in current_pts or new_j in current_pts or new_i == new_j:
                    continue
                
                current_pts[i] = new_i
                current_pts[j] = new_j
                h4, new_score = analyze_lines(current_pts)
                
                if h4:
                    current_pts[i] = pi
                    current_pts[j] = pj
                    continue
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = current_pts[:]
                        if steps % 20 == 0:
                            exact = max_no_3_collinear(current_pts)
                            if exact < best_exact_score:
                                best_exact_score = exact
                else:
                    current_pts[i] = pi
                    current_pts[j] = pj
            
            T *= alpha
            steps += 1
            
            if steps % 50 == 0:
                exact = max_no_3_collinear(current_pts)
                if exact < best_exact_score:
                    best_exact_score = exact
                    best_pts = current_pts[:]
        
        return best_pts, best_exact_score

    # Run SA multiple times with different random seeds/starts
    best_global_pts = None
    best_global_score = 21
    
    for run in range(8):
        if time.time() > deadline - 10:
            break
        
        pts, score = simulated_annealing()
        
        if score < best_global_score:
            best_global_score = score
            best_global_pts = pts
        
        if best_global_score <= 9:
            break

    if best_global_pts is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
        
    return [list(p) for p in best_global_pts]

# Test harness (optional, for local debugging)
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Valid (no 4 collinear):", not has_4_collinear(pts))
    print("Score:", max_no_3_collinear(pts))
