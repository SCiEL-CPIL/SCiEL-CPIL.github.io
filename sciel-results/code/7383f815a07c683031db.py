"""No-isosceles set: maximize points in [0,64)^2 with no 3 forming isosceles triangle."""
import time
import random

N = 64

def construct():
    start = time.time()
    deadline = start + 290  # 40s safety margin for 330s limit
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    def check_add(p, current_set):
        """
        Check if p can be added to current_set (list of (x,y) tuples).
        Returns True if no isosceles triangle is formed.
        """
        px, py = p
        n = len(current_set)
        for i in range(n):
            ax, ay = current_set[i]
            d_pa = (px - ax) * (px - ax) + (py - ay) * (py - ay)
            
            for j in range(i + 1, n):
                bx, by = current_set[j]
                d_pb = (px - bx) * (px - bx) + (py - by) * (py - by)
                d_ab = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    return False
        return True

    # Run multiple trials
    for trial in range(500):
        if time.time() > deadline - 20:
            break
        
        random.seed(trial * 104729 + 1)
        order = all_points[:]
        random.shuffle(order)
        
        current_set = []
        
        for p in order:
            if check_add(p, current_set):
                current_set.append(p)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Local search: try to add more points
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            selected_set = set(best_set)
            for p in all_points:
                if p not in selected_set:
                    if check_add(p, best_set):
                        best_set.append(p)
                        selected_set.add(p)
                        improved = True
                        break
                        
    return best_set
