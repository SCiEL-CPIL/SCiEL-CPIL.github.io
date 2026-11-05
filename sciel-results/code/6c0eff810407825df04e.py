import time
import random
from collections import Counter

def construct():
    deadline = time.time() + 50.0
    random.seed(12345)
    
    N = 30
    
    def doubled_area(p1, p2, p3):
        d = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
        return d if d >= 0 else -d
    
    def full_count(points):
        areas = []
        for i in range(N):
            pi = points[i]
            for j in range(i+1, N):
                pj = points[j]
                for k in range(j+1, N):
                    pk = points[k]
                    a = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pk[0]-pi[0])*(pj[1]-pi[1])
                    if a < 0:
                        a = -a
                    if a > 0:
                        areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count
    
    # Initial grid: 6x5
    coords = [(x, y) for x in range(6) for y in range(5)]
    
    counter, best_area, best_count = full_count(coords)
    best_overall = best_count
    best_points = list(coords)
    best_target = best_area
    
    target = best_target
    current_count = best_count
    
    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx: continue
            for k in range(j+1, N):
                if k == idx: continue
                pairs.append((j, k))
        return pairs
    
    all_pairs = [get_pairs_excluding(i) for i in range(N)]
    
    no_improve = 0
    max_no_improve = 200
    iterations = 0
    
    restarts = 0
    max_restarts = 3
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 100
            
            restarts += 1
            if restarts > max_restarts:
                coords = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(N)]
                seen = set()
                for i in range(N):
                    while coords[i] in seen:
                        coords[i] = (random.randint(-10, 10), random.randint(-10, 10))
                    seen.add(coords[i])
                restarts = 0
                no_improve = 0
                continue
        
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        improved = False
        
        # Pre-extract coordinates of other points for this idx
        other_coords = []
        for j, k in pairs:
            other_coords.append((coords[j], coords[k]))
        
        for _ in range(10):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt == old_pt:
                continue
            if new_pt in coords:
                continue
            
            old_matches = 0
            new_matches = 0
            nx, ny = new_pt
            ox, oy = old_pt
            
            for pj, pk in other_coords:
                jx, jy = pj
                kx, ky = pk
                
                # Old area
                d_old = (jx - ox) * (ky - oy) - (kx - ox) * (jy - oy)
                if d_old < 0:
                    d_old = -d_old
                if d_old == target:
                    old_matches += 1
                
                # New area
                d_new = (jx - nx) * (ky - ny) - (kx - nx) * (jy - ny)
                if d_new < 0:
                    d_new = -d_new
                if d_new == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            
            if delta > 0:
                current_count += delta
                coords[idx] = new_pt
                improved = True
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
                break
            elif delta == 0 and random.random() < 0.03:
                coords[idx] = new_pt
                no_improve += 1
                break
        
        if not improved:
            no_improve += 1
        
        iterations += 1
    
    # Final polish: try to improve with deterministic local moves
    if time.time() < deadline - 2.0:
        for polish_iter in range(50):
            if time.time() > deadline - 2.0:
                break
            progress = False
            for idx in range(N):
                if time.time() > deadline - 2.0:
                    break
                old_pt = coords[idx]
                pairs = all_pairs[idx]
                best_delta = 0
                best_new_pt = None
                
                # Try all positions in a small neighborhood
                ox, oy = old_pt
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        new_pt = (nx, ny)
                        if new_pt in coords:
                            continue
                        
                        new_matches = 0
                        for j, k in pairs:
                            pj = coords[j]
                            pk = coords[k]
                            jx, jy = pj
                            kx, ky = pk
                            d = (jx - nx) * (ky - ny) - (kx - nx) * (jy - ny)
                            if d < 0:
                                d = -d
                            if d == target:
                                new_matches += 1
                        
                        old_matches = 0
                        for j, k in pairs:
                            pj = coords[j]
                            pk = coords[k]
                            jx, jy = pj
                            kx, ky = pk
                            d = (jx - ox) * (ky - oy) - (kx - ox) * (jy - oy)
                            if d < 0:
                                d = -d
                            if d == target:
                                old_matches += 1
                        
                        d_delta = new_matches - old_matches
                        if d_delta > best_delta:
                            best_delta = d_delta
                            best_new_pt = new_pt
                
                if best_new_pt is not None and best_delta > 0:
                    current_count += best_delta
                    coords[idx] = best_new_pt
                    progress = True
                    if current_count > best_overall:
                        best_overall = current_count
                        best_points = list(coords)
            
            if not progress:
                break
    
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x,y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x,y) for x in range(6) for y in range(5)]
            
    return best_points
