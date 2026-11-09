import time
import random

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit (evaluator kills at 150s, but keep margin)
    
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices for speed
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def run_greedy(rng):
        """
        Generate an initial snake using a greedy heuristic.
        """
        path = [rng.randint(0, MAX_VERTS - 1)]
        used = [False] * MAX_VERTS
        used[path[0]] = True
        adj_count = [0] * MAX_VERTS
        
        # Initialize adj_count for the first vertex
        for nb in neighbors[path[0]]:
            adj_count[nb] += 1
            
        # Greedy growth
        while True:
            if time.time() > deadline:
                break
                
            # Check tail
            tail_v = path[-1]
            candidates_tail = []
            for nb in neighbors[tail_v]:
                if used[nb]: continue
                if (nb ^ tail_v).bit_count() == 1 and adj_count[nb] == 1:
                    candidates_tail.append(nb)
            
            # Check head
            head_v = path[0]
            candidates_head = []
            for nb in neighbors[head_v]:
                if used[nb]: continue
                if (nb ^ head_v).bit_count() == 1 and adj_count[nb] == 1:
                    candidates_head.append(nb)
            
            if not candidates_tail and not candidates_head:
                break
                
            # Warnsdorff: pick the end with fewer options
            if len(candidates_head) < len(candidates_tail):
                v = rng.choice(candidates_head)
                path.insert(0, v)
            else:
                v = rng.choice(candidates_tail)
                path.append(v)
                
            used[v] = True
            for nb in neighbors[v]:
                adj_count[nb] += 1
                
        return path, used, adj_count

    def local_search(path, used, adj_count, rng):
        """
        Fast local search using randomized exhaustive insertion.
        """
        # We do multiple passes. In each pass, we shuffle the gap order.
        # After each insertion, we restart the pass to handle index shifts.
        
        max_passes = 50
        for _ in range(max_passes):
            if time.time() > deadline:
                break
                
            if len(path) < 2:
                break
                
            # Create a list of gap indices to try
            gap_indices = list(range(1, len(path)))
            rng.shuffle(gap_indices)
            
            inserted_any = False
            
            for pos in gap_indices:
                if time.time() > deadline:
                    break
                
                if pos < 1 or pos >= len(path):
                    continue
                    
                prev_v = path[pos - 1]
                curr_v = path[pos]
                
                # Find candidates for insertion at this gap
                candidates = []
                for u in neighbors[prev_v]:
                    if used[u]:
                        continue
                    if (u ^ curr_v).bit_count() != 1:
                        continue
                    if adj_count[u] == 2:
                        candidates.append(u)
                        
                if candidates:
                    v = rng.choice(candidates)
                    path.insert(pos, v)
                    used[v] = True
                    for nb in neighbors[v]:
                        adj_count[nb] += 1
                    inserted_any = True
                    break # Restart pass
                    
            if not inserted_any:
                break
                
        return path, used, adj_count

    def run_attempt(rng):
        path, used, adj_count = run_greedy(rng)
        path, used, adj_count = local_search(path, used, adj_count, rng)
        return path

    # Main Loop
    while time.time() < deadline:
        seed = int(time.time() * 1000) + random.randint(0, 10000)
        rng = random.Random(seed)
        
        path = run_attempt(rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
    # Final Polish: Try to extend the best path further
    if best_path is not None and time.time() < deadline - 2.0:
        used = [False] * MAX_VERTS
        for v in best_path:
            used[v] = True
        adj_count = [0] * MAX_VERTS
        for v in best_path:
            for nb in neighbors[v]:
                adj_count[nb] += 1
                
        rng = random.Random(12345)
        polished_path, _, _ = local_search(best_path, used, adj_count, rng)
        
        if len(polished_path) > best_len:
            best_path = polished_path
            best_len = len(best_path)

    if best_path is None:
        return [0]
        
    return best_path

# EVOLVE-BLOCK-END
