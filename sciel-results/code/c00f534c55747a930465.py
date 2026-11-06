import time
import random
import collections

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 30
    COORD_MIN = -30
    COORD_MAX = 30
    
    best_score = 0
    best_points = None
    
    def get_initial_points():
        # Grid 6x5
        pts = []
        for x in range(6):
            for y in range(5):
                pts.append((x, y))
        # Randomize order
        random.shuffle(pts)
        return pts

    def calculate_areas(points):
        counts = collections.Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    area = abs(x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2)
                    if area > 0:
                        counts[area] += 1
        return counts

    def get_score_from_counts(counts):
        if not counts:
            return 0
        return max(counts.values())

    def local_search(points):
        nonlocal best_score, best_points
        
        current_points = [p for p in points]
        current_counts = calculate_areas(current_points)
        current_score = get_score_from_counts(current_counts)
        
        # Precompute indices for triples containing each point i
        triples_with = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    triples_with[i].append((j, k))
                    triples_with[j].append((i, k))
                    triples_with[k].append((i, j))
                    
        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.1
        
        step = 0
        while time.time() < deadline:
            step += 1
            
            # Pick a random point
            idx = random.randint(0, N-1)
            old_x, old_y = current_points[idx]
            
            # Generate new position
            if random.random() < 0.8:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                new_x = old_x + dx
                new_y = old_y + dy
            else:
                new_x = random.randint(COORD_MIN, COORD_MAX)
                new_y = random.randint(COORD_MIN, COORD_MAX)
                
            # Bounds check
            if new_x < COORD_MIN or new_x > COORD_MAX or new_y < COORD_MIN or new_y > COORD_MAX:
                continue
                
            # Distinctness check
            is_valid = True
            for i, p in enumerate(current_points):
                if i != idx and p == (new_x, new_y):
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Store old areas for affected triples
            old_areas = []
            for j, k in triples_with[idx]:
                x1, y1 = current_points[idx]
                x2, y2 = current_points[j]
                x3, y3 = current_points[k]
                old_area = abs(x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2)
                old_areas.append(old_area)
                if old_area > 0:
                    current_counts[old_area] -= 1
                    if current_counts[old_area] == 0:
                        del current_counts[old_area]
                        
            # Apply move
            current_points[idx] = (new_x, new_y)
            
            # Add new areas
            new_areas = []
            for j, k in triples_with[idx]:
                x1, y1 = current_points[idx]
                x2, y2 = current_points[j]
                x3, y3 = current_points[k]
                new_area = abs(x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2)
                new_areas.append(new_area)
                if new_area > 0:
                    current_counts[new_area] += 1
                    
            # Calculate new score
            new_score = get_score_from_counts(current_counts)
            
            # Acceptance
            accepted = False
            if new_score > current_score:
                accepted = True
            else:
                if new_score < current_score:
                    delta = new_score - current_score
                    if random.random() < pow(2.71828, delta / T):
                        accepted = True
                else: # new_score == current_score
                    if random.random() < 0.5: # Accept equal with prob 0.5
                        accepted = True
            
            if accepted:
                current_score = new_score
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
                # Remove new areas
                for area in new_areas:
                    if area > 0:
                        current_counts[area] -= 1
                        if current_counts[area] == 0:
                            del current_counts[area]
                # Add back old areas
                for area in old_areas:
                    if area > 0:
                        current_counts[area] += 1

            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_points = [p for p in current_points]
                
            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T
                
            # Break if time is tight
            if time.time() > deadline - 2.0:
                break
                
        return best_score, best_points

    # Main loop
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        pts = get_initial_points()
        for i in range(N):
            if random.random() < 0.2:
                x, y = pts[i]
                pts[i] = (x + random.randint(-1, 1), y + random.randint(-1, 1))
        
        score, final_pts = local_search(pts)
        
        if time.time() > deadline - 2.0:
            break

    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
