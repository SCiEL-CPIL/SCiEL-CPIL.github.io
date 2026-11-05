import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270
    
    def dist2(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy
    
    def is_valid_addition(point, current_set, dist_matrix):
        n = len(current_set)
        for i in range(n):
            da = dist2(point, current_set[i])
            for j in range(i + 1, n):
                db = dist2(point, current_set[j])
                dab = dist_matrix[i][j]
                if da == db or da == dab or db == dab:
                    return False
        return True
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    rng = random.Random(999)
    
    best_set = []
    best_size = 0
    
    while time.time() < deadline:
        points = all_points[:]
        rng.shuffle(points)
        
        current_set = []
        dist_matrix = []  # dist_matrix[i][j] = dist2(current_set[i], current_set[j])
        
        for p in points:
            if is_valid_addition(p, current_set, dist_matrix):
                # Add to dist_matrix
                new_row = [0] * len(current_set)
                for i, a in enumerate(current_set):
                    d = dist2(p, a)
                    new_row[i] = d
                    dist_matrix[i].append(d)
                dist_matrix.append(new_row)
                current_set.append(p)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return best_set
# EVOLVE-BLOCK-END
