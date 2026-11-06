import time
import random
import sys

N = 9
MAX_TIME = 115.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Precompute neighbors for fast access
    # neighbors[v] is a list of neighbors of v
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def is_valid_extension(path, new_v):
        # path is a list of vertices
        # new_v is adjacent to path[-1]
        # Check if new_v is adjacent to any path[i] for i < len(path) - 2
        # We can optimize this by checking distance to all vertices in path except the last two
        # Since path is an induced path, dist(path[i], path[j]) >= 2 for |i-j|>1.
        # We just need to ensure new_v is not adjacent to path[i] for i <= len(path)-3.
        # Note: path[-1] is the current end. new_v is adjacent to path[-1].
        # path[-2] is adjacent to path[-1]. dist(new_v, path[-2]) must be >= 2?
        # Yes, because if dist(new_v, path[-2]) == 1, then new_v, path[-1], path[-2] form a triangle?
        # No, in hypercube, no triangles. 
        # If new_v ~ path[-1] and new_v ~ path[-2], then dist(path[-1], path[-2]) = 2.
        # But path[-1] and path[-2] are adjacent (dist 1). So this is impossible in a hypercube?
        # Wait. In Q_n, if u~v and u~w, then dist(v,w) is 0, 2, or 4...
        # If v~w, dist(v,w)=1.
        # Can u~v and u~w and v~w? No, hypercube is bipartite, no odd cycles.
        # So if new_v ~ path[-1] and new_v ~ path[-2], then path[-1] and path[-2] must be at distance 2.
        # But they are consecutive in the path, so dist is 1. Contradiction.
        # Therefore, new_v cannot be adjacent to path[-2].
        # So we only need to check i < len(path) - 2.
        
        # Optimization: Use bit_count
        for i in range(len(path) - 2):
            if (new_v ^ path[i]).bit_count() == 1:
                return False
        return True

    def greedy_extend(start_v, max_steps=200):
        path = [start_v]
        used = {start_v}
        for _ in range(max_steps):
            cur = path[-1]
            candidates = []
            for v in neighbors[cur]:
                if v in used:
                    continue
                if is_valid_extension(path, v):
                    # Heuristic: prefer vertices with fewer valid extensions (MRV)
                    # or more? Usually MRV (most constrained) is better for backtracking.
                    # For greedy, maybe prefer those with more options to avoid dead ends?
                    # Let's use a simple heuristic: count valid neighbors of v.
                    count = 0
                    for nv in neighbors[v]:
                        if nv in used:
                            continue
                        # Check if nv is valid extension from v
                        # path + [v] is the new path.
                        # nv is adjacent to v.
                        # Need to check nv against path[0:len(path)-1]
                        valid = True
                        for i in range(len(path) - 1):
                            if (nv ^ path[i]).bit_count() == 1:
                                valid = False
                                break
                        if valid:
                            count += 1
                    candidates.append((count, v))
            if not candidates:
                break
            # Sort by count ascending (MRV) to commit early to constrained choices?
            # Or descending to keep options open?
            # In snake-in-the-box, "dead ends" are common. MRV tends to find solutions faster in CSPs.
            candidates.sort(key=lambda x: x[0])
            # Randomize tie-breaking to allow different paths in beam search
            next_v = candidates[0][1]
            path.append(next_v)
            used.add(next_v)
        return path

    def beam_search(initial_path, beam_width=50):
        best_path = initial_path
        beams = [initial_path]
        
        while time.time() < deadline:
            new_beams = []
            for path in beams:
                cur = path[-1]
                used = set(path) # This is expensive, O(L). 
                # Optimization: We can maintain used sets in the beam items, 
                # but for simplicity and given N=9, L is small (~100-200), set creation is okay.
                # Actually, we can just check `v in path` which is O(L). 
                # Let's stick to list membership for now, or pass used set.
                
                candidates = []
                for v in neighbors[cur]:
                    if v in used:
                        continue
                    if is_valid_extension(path, v):
                        # Score the new path
                        # Heuristic: number of valid extensions from v
                        # We want to keep paths that are "promising"
                        # A common score is the number of available neighbors.
                        # Let's calculate a quick score: len(path) + 1
                        # To differentiate, we can look at the "future potential".
                        # For now, just keep the path.
                        new_path = path + [v]
                        # To avoid duplicate paths in beams, we can use a set of tuples,
                        # but that's heavy. 
                        # We'll just add to new_beams and deduplicate/prune later.
                        new_beams.append(new_path)
            
            if not new_beams:
                break
                
            # Prune new_beams to keep top `beam_width`
            # Score: primarily length, secondarily heuristic
            # Since all new_beams have length = len(initial_path) + 1, 
            # we need a tie-breaker.
            # Let's use the number of valid extensions from the end of the new path.
            def score_path(p):
                if not p:
                    return 0
                cur = p[-1]
                used = set(p)
                count = 0
                for v in neighbors[cur]:
                    if v in used:
                        continue
                    if is_valid_extension(p, v):
                        count += 1
                return count

            # Sort by score descending
            new_beams.sort(key=score_path, reverse=True)
            beams = new_beams[:beam_width]
            
            # Update best path
            if len(beams[0]) > len(best_path):
                best_path = beams[0]
                
        return best_path

    # 1. Generate initial diverse paths using greedy with randomization
    initial_paths = []
    for i in range(10):
        random.seed(i)
        # Add some randomness to greedy
        # We can modify greedy to pick randomly among top candidates
        # For now, just run greedy from 0 with different tie-breaking logic if possible
        # Or just run beam search from a few greedy starts.
        
        # Let's create a slightly randomized greedy
        def randomized_greedy(start_v, max_steps=200, rnd=None):
            if rnd is None:
                rnd = random.Random()
            path = [start_v]
            used = {start_v}
            for _ in range(max_steps):
                cur = path[-1]
                candidates = []
                for v in neighbors[cur]:
                    if v in used:
                        continue
                    if is_valid_extension(path, v):
                        count = 0
                        for nv in neighbors[v]:
                            if nv in used:
                                continue
                            valid = True
                            for i in range(len(path) - 1):
                                if (nv ^ path[i]).bit_count() == 1:
                                    valid = False
                                    break
                            if valid:
                                count += 1
                        candidates.append((count, v))
                if not candidates:
                    break
                # Pick randomly from top 3 candidates (by MRV score)
                candidates.sort(key=lambda x: x[0])
                top_k = min(3, len(candidates))
                chosen = rnd.choice(candidates[:top_k])
                next_v = chosen[1]
                path.append(next_v)
                used.add(next_v)
            return path

        p = randomized_greedy(0, 200)
        initial_paths.append(p)

    # 2. Run beam search on the best few initial paths
    best_overall = max(initial_paths, key=len)
    
    # Sort initial paths by length
    initial_paths.sort(key=len, reverse=True)
    
    # Run beam search on top 3 initial paths
    for init_p in initial_paths[:3]:
        if time.time() > deadline:
            break
        # Use a smaller beam width for multiple runs
        res = beam_search(init_p, beam_width=20)
        if len(res) > len(best_overall):
            best_overall = res

    return best_overall

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
