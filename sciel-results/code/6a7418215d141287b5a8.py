import random
import time

N = 9
V = 1 << N
MAX_TIME = 110

# Precompute neighbors as lists of integers for faster iteration
NEIGHBORS = [[] for _ in range(V)]
for v in range(V):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    best_path = []
    best_len = 0
    start_time = time.time()
    
    # We will run a heuristic search:
    # 1. Start with a random vertex (or 0).
    # 2. Greedily extend using a "forward degree" heuristic.
    # 3. If stuck, backtrack a few steps.
    # 4. If backtracking fails, restart.
    
    # To maximize diversity, we can also try different heuristics or random seeds.
    
    def score_candidate(v, used_mask, bad_mask, cur):
        # Count how many neighbors of v are available for the next step
        # Available means: not used, not in bad_mask, and not cur (since cur will be in bad_mask for next step? No, cur is in path)
        # Actually, when we move to v, cur becomes "bad" for the *next* step.
        # So the available neighbors for the step *after* v are:
        # Neighbors of v that are not in used_mask, not in bad_mask, and not cur.
        count = 0
        for u in NEIGHBORS[v]:
            if (used_mask >> u) & 1: continue
            if (bad_mask >> u) & 1: continue
            if u == cur: continue # cur is adjacent to v, but cur is in path. 
            # Wait, the "bad" mask for the next step will include neighbors of cur.
            # So if u is a neighbor of cur, it will be bad.
            # We check if u is in NEIGHBORS[cur]? No, that's expensive.
            # We can just check if u is in the set of neighbors of cur.
            # But we don't have a fast "is neighbor" check other than bit ops.
            # Let's refine: 
            # Next bad_mask = bad_mask | NEIGHBORS[cur] (as a mask).
            # So a vertex u is valid for the step after v if:
            # 1. u is not in used_mask
            # 2. u is not in bad_mask
            # 3. u is not in NEIGHBORS[cur] (because cur becomes "bad")
            # 4. u is not v (v is used)
            
            # We can precompute a bitmask for neighbors of cur? 
            # We already have NEIGHBOR_MASKS in the original code. Let's use that.
            pass
        return count

    # Let's stick to the bitmask approach for speed but improve the selection heuristic.
    NEIGHBOR_MASKS = [0] * V
    for v in range(V):
        mask = 0
        for b in range(N):
            mask |= 1 << (v ^ (1 << b))
        NEIGHBOR_MASKS[v] = mask

    def heuristic_greedy(start_vertex):
        path = [start_vertex]
        used_mask = 1 << start_vertex
        bad_mask = 0
        
        # Backtracking stack: (path_len, used_mask, bad_mask, candidates_list, next_index)
        # We will implement a simple limited backtracking.
        # If we get stuck, we pop the last vertex and try the next candidate.
        
        # To make it faster, we'll use a list of (vertex, score) for candidates.
        
        while len(path) < V:
            if time.time() - start_time > MAX_TIME:
                return path
            
            cur = path[-1]
            
            # Find candidates
            candidates_mask = NEIGHBOR_MASKS[cur] & ~used_mask & ~bad_mask
            
            if candidates_mask == 0:
                # Stuck. Backtrack.
                if len(path) <= 1:
                    break
                # Pop last vertex
                path.pop()
                # We need to restore state? 
                # This simple backtracking is tricky because we don't store the full state history.
                # Let's use a different strategy: 
                # If stuck, we restart the greedy process from a random prefix of the current path?
                # Or just restart from scratch?
                # Given the time limit, restarting from scratch is safe but might be slow.
                # Let's try to "kick" the search: remove the last K vertices and re-greedy.
                
                # For simplicity and speed, we will just restart the whole greedy process 
                # with a new random start if we get stuck early, 
                # or try to backtrack manually if we are deep.
                
                # Manual backtracking:
                # We need to know what the previous candidates were.
                # This requires storing history.
                
                # Alternative: Just restart.
                return path
            
            # Generate list of candidates
            candidates = []
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            # Heuristic: Score each candidate by its forward degree
            # Forward degree = number of neighbors of v that are not used, not bad, and not cur
            # Note: cur will be added to bad_mask for the next step.
            scores = []
            for v in candidates:
                # Next bad mask will be bad_mask | NEIGHBOR_MASKS[cur]
                next_bad = bad_mask | NEIGHBOR_MASKS[cur]
                # Available neighbors for v:
                # Neighbors of v that are not in used_mask, not in next_bad
                avail_mask = NEIGHBOR_MASKS[v] & ~used_mask & ~next_bad
                # Count bits
                count = bin(avail_mask).count('1')
                scores.append((count, v))
            
            # Sort by score descending, then randomize ties
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Pick a candidate. 
            # To add diversity, we can pick from the top 3 with some probability.
            # Or just pick the best one. Picking the best is faster but might get stuck.
            # Let's pick the best one with 70% probability, and a random one from top 3 with 30%.
            if len(scores) > 1 and random.random() < 0.3:
                top_k = min(3, len(scores))
                # Randomly select from top k
                idx = random.randint(0, top_k - 1)
                v = scores[idx][1]
            else:
                v = scores[0][1]
            
            # Update state
            path.append(v)
            used_mask |= (1 << v)
            bad_mask |= NEIGHBOR_MASKS[cur]
            
            if len(path) > best_len:
                # Update best
                pass # We return the path at the end
        
        return path

    # Main loop: Run many heuristic greedy attempts
    # We can also try to extend the best path found so far.
    
    # Strategy:
    # 1. Run a few greedy searches from 0.
    # 2. Run greedy searches from random vertices.
    # 3. If we find a good path, try to "extend" it by starting the greedy from the last vertex 
    #    with the used/bad masks from that path? 
    #    This is hard because we need to reconstruct the masks.
    
    # Let's just run many independent greedy searches.
    # To improve, we can also try "reverse greedy" or different tie-breaking.
    
    # Optimization: 
    # The heuristic calculation `bin(avail_mask).count('1')` is slow.
    # We can precompute popcount? No, masks are 512 bits. 
    # But `bin().count('1')` is implemented in C and is quite fast.
    
    # Let's run for MAX_TIME
    while time.time() - start_time < MAX_TIME:
        # Random start vertex
        start_v = random.randint(0, V - 1)
        p = heuristic_greedy(start_v)
        if len(p) > len(best_path):
            best_path = p
            best_len = len(p)
        
        # Also try starting from 0 with different random seeds (implicitly done by random choices in heuristic)
        # To get more diversity, we can vary the "randomness" in the heuristic.
        # The current heuristic uses random.random() < 0.3.
        
        # We can also try to "kick" the best path:
        # Take best_path, remove the last K vertices, and re-greedy from there?
        # This requires reconstructing used/bad masks.
        # Let's implement a "resume" function.
        
        if best_len > 10:
            # Try to extend best_path by restarting greedy from a random point in the middle?
            # Or just from the end?
            # If we start from the end, we need the used/bad masks.
            # We can compute them from best_path.
            
            # Compute masks for best_path
            used = 0
            bad = 0
            for i, v in enumerate(best_path):
                used |= (1 << v)
                if i > 0:
                    bad |= NEIGHBOR_MASKS[best_path[i-1]]
            
            # Now, try to extend from the last vertex of best_path
            # But we are already at the end. 
            # Instead, let's try to "repair" by removing the last 5 vertices and re-greedy.
            if len(best_path) > 5:
                cut_len = len(best_path) - random.randint(1, 5)
                prefix = best_path[:cut_len]
                
                # Reconstruct masks for prefix
                used = 0
                bad = 0
                for i, v in enumerate(prefix):
                    used |= (1 << v)
                    if i > 0:
                        bad |= NEIGHBOR_MASKS[prefix[i-1]]
                
                # Now run a greedy extension from prefix
                # We modify heuristic_greedy to accept initial state?
                # Or just call a helper.
                
                # Let's just create a new path starting from prefix
                path = prefix[:]
                cur = path[-1]
                
                # Extend
                while len(path) < V:
                    if time.time() - start_time > MAX_TIME:
                        break
                    
                    candidates_mask = NEIGHBOR_MASKS[cur] & ~used & ~bad
                    if candidates_mask == 0:
                        # Stuck. Break.
                        break
                    
                    candidates = []
                    m = candidates_mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        candidates.append(v)
                        m ^= lsb
                    
                    # Heuristic
                    scores = []
                    for v in candidates:
                        next_bad = bad | NEIGHBOR_MASKS[cur]
                        avail_mask = NEIGHBOR_MASKS[v] & ~used & ~next_bad
                        count = bin(avail_mask).count('1')
                        scores.append((count, v))
                    
                    scores.sort(key=lambda x: x[0], reverse=True)
                    
                    if len(scores) > 1 and random.random() < 0.3:
                        top_k = min(3, len(scores))
                        idx = random.randint(0, top_k - 1)
                        v = scores[idx][1]
                    else:
                        v = scores[0][1]
                    
                    path.append(v)
                    used |= (1 << v)
                    bad |= NEIGHBOR_MASKS[cur]
                    cur = v
                    
                if len(path) > len(best_path):
                    best_path = path
                    best_len = len(path)

    return best_path
