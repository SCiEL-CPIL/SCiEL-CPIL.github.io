import random
import time

N = 9

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    best_path = [0]
    
    def generate_snake():
        path = [0]
        used = {0}
        
        def is_valid(nb, path):
            if nb in used:
                return False
            # Check against all except last
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    return False
            return True

        def dfs():
            nonlocal best_path
            if time.time() > deadline:
                return
            
            cur = path[-1]
            valid_neighbors = []
            for nb in neighbors[cur]:
                if is_valid(nb, path):
                    valid_neighbors.append(nb)
            
            if not valid_neighbors:
                return
            
            random.shuffle(valid_neighbors)
            
            for nb in valid_neighbors:
                path.append(nb)
                used.add(nb)
                
                if len(path) > len(best_path):
                    best_path = path.copy()
                    
                dfs()
                
                path.pop()
                used.remove(nb)

        dfs()
        return path

    while time.time() < deadline:
        generate_snake()
        
    return best_path
