import time
import random

N = 64

def squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    best_set = []
    best_size = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def is_valid_addition(p, s_list, dist_sets):
        """Check if adding p to s_list creates any isosceles triangle."""
        seen_dists = set()
        sd = squared_dist
        for i in range(len(s_list)):
            a = s_list[i]
            d = sd(p, a)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            if d in dist_sets[i]:
                return False
        return True

    def add_point(p, s_list, dist_sets):
        """Add point p to s_list and update dist_sets."""
        new_dist_set = set()
        sd = squared_dist
        for i in range(len(s_list)):
            a = s_list[i]
            d = sd(p, a)
            new_dist_set.add(d)
            dist_sets[i].add(d)
        s_list.append(p)
        dist_sets.append(new_dist_set)

    def greedy_pass(s_list, dist_sets, order):
        """Try to add points from order to s_list/dist_sets."""
        for p in order:
            if time.time() > deadline:
                break
            if p not in s_list and is_valid_addition(p, s_list, dist_sets):
                add_point(p, s_list, dist_sets)

    num_restarts = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        s_list = []
        dist_sets = []
        
        # First pass: random order
        shuffled1 = all_points[:]
        random.shuffle(shuffled1)
        greedy_pass(s_list, dist_sets, shuffled1)
        
        # Second pass: different random order, try to add more
        shuffled2 = all_points[:]
        random.shuffle(shuffled2)
        greedy_pass(s_list, dist_sets, shuffled2)
        
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = s_list[:]
            
    return best_set
