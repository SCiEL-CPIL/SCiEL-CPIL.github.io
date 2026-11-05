import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_path = []
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]
    
    # Precompute neighbor sets for faster lookup
    neighbor_sets = [set(neighbors[v]) for v in range(NUM_VERTS)]

    def greedy_extend(start, use_random_tie_breaker=False, heuristic='warnsdorff'):
        """
        Greedily extend a snake starting from `start`.
        Returns the path as a list of integers.
        """
        path = [start]
        used = {start}
        
        forbidden = set()
        
        while True:
            cur = path[-1]
            candidates = []
            
            for nxt in neighbors[cur]:
                if nxt in used:
                    continue
                if nxt in forbidden:
                    continue
                candidates.append(nxt)
                
            if not candidates:
                break
            
            best_score = float('inf') if heuristic == 'warnsdorff' else -1
            best_nxt = None
            best_rand = random.random() if use_random_tie_breaker else 0
            
            for nxt in candidates:
                # Count available neighbors for nxt
                # A neighbor m of nxt is available if:
                # 1. m not in used
                # 2. m not in forbidden
                # 3. m not in neighbors(cur) (will be added to forbidden)
                free_count = 0
                for m in neighbors[nxt]:
                    if m in used:
                        continue
                    if m in forbidden:
                        continue
                    if m in neighbor_sets[cur]:
                        continue
                    free_count += 1
                
                if heuristic == 'warnsdorff':
                    # Prefer fewer options (tighter)
                    if free_count < best_score:
                        best_score = free_count
                        best_nxt = nxt
                        if use_random_tie_breaker:
                            best_rand = random.random()
                    elif free_count == best_score:
                        if use_random_tie_breaker:
                            r = random.random()
                            if r > best_rand:
                                best_nxt = nxt
                                best_rand = r
                else:
                    # Prefer more options
                    if free_count > best_score:
                        best_score = free_count
                        best_nxt = nxt
                        if use_random_tie_breaker:
                            best_rand = random.random()
                    elif free_count == best_score:
                        if use_random_tie_breaker:
                            r = random.random()
                            if r > best_rand:
                                best_nxt = nxt
                                best_rand = r
            
            if best_nxt is None:
                break
                
            path.append(best_nxt)
            used.add(best_nxt)
            
            # Update forbidden
            # The previous `cur` (which is now path[-2]) becomes part of the "forbidden history"
            # Any neighbor of cur is now forbidden for the next step
            forbidden.update(neighbor_sets[cur])
                
            # Time check
            if time.time() > deadline:
                break
        
        return path

    # Strategy 1: Deterministic greedy from all starts (Warnsdorff)
    for start in range(NUM_VERTS):
        if time.time() > deadline - 5:
            break
        p = greedy_extend(start, use_random_tie_breaker=False, heuristic='warnsdorff')
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    # Strategy 2: Randomized greedy restarts (Warnsdorff)
    while time.time() < deadline - 2:
        start = random.randint(0, NUM_VERTS - 1)
        p = greedy_extend(start, use_random_tie_breaker=True, heuristic='warnsdorff')
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    return best_path

# Verification helper (not used in construct, but good to have for debugging if needed)
def verify_snake(path):
    if not path:
        return True
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() < 2:
                return False
    return True

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)}")
    print(f"Valid: {verify_snake(p)}")
