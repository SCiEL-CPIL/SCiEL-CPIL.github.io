import time
import random

N = 9
MAX_TIME = 115.0 # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s usually, let's be safe)

# Precompute neighbors
NEIGHBORS = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    def time_left():
        return MAX_TIME - (time.time() - start_time)

    def greedy_search(randomize=True):
        """
        Performs a greedy construction of a snake.
        Returns the path found.
        """
        path = [0]
        in_path = {0}
        # adj_count[u] = number of vertices in 'path' adjacent to u
        adj_count = [0] * (1 << N)
        
        # Initialize adj_count for vertex 0
        for u in NEIGHBORS[0]:
            adj_count[u] += 1
            
        while True:
            cur = path[-1]
            candidates = []
            
            for u in NEIGHBORS[cur]:
                if u in in_path:
                    continue
                # u is adjacent to cur.
                # For u to be a valid next step, it must not be adjacent to any OTHER vertex in path.
                # adj_count[u] counts all path vertices adjacent to u.
                # Since cur is in path and adjacent to u, adj_count[u] is at least 1.
                # If adj_count[u] == 1, then only cur is adjacent. Valid.
                # If adj_count[u] > 1, another vertex is adjacent. Invalid.
                if adj_count[u] == 1:
                    # Calculate heuristic score: how many options does u have next?
                    # This is the Warnsdorff rule.
                    # Count valid neighbors of u.
                    options = 0
                    for w in NEIGHBORS[u]:
                        if w in in_path:
                            continue
                        # w must not be adjacent to any vertex in path EXCEPT u (which is not in path yet)
                        # Wait, if we add u, the new path is path + [u].
                        # The check for w (next after u) is:
                        # w not in new_in_path (path + {u})
                        # w not adjacent to any p in new_path[:-1] (path)
                        # Note: w is adjacent to u.
                        # So we check if w is adjacent to any p in path.
                        # adj_count[w] currently counts neighbors in 'path'.
                        # If adj_count[w] == 0, then w is not adjacent to any p in path.
                        # If adj_count[w] > 0, w is adjacent to some p in path.
                        # So w is a valid candidate for the step after u ONLY IF adj_count[w] == 0.
                        if adj_count[w] == 0:
                            options += 1
                    
                    candidates.append((options, u))
            
            if not candidates:
                break
            
            # Sort by options (ascending) to prefer constrained moves
            candidates.sort(key=lambda x: x[0])
            
            # Randomization: Pick from the top K candidates with some probability, or shuffle ties
            # To explore more, we can pick randomly from the best few.
            if randomize and len(candidates) > 1:
                # Pick from the top 3 or 4 candidates to allow some diversity
                top_k = min(3, len(candidates))
                # Weighted random? Or just random choice from top_k?
                # Let's try random choice from the top 2 to balance greediness and exploration
                chosen_idx = random.randint(0, min(1, len(candidates) - 1))
                _, nxt = candidates[chosen_idx]
            else:
                _, nxt = candidates[0]
            
            # Add nxt to path
            path.append(nxt)
            in_path.add(nxt)
            
            # Update adj_count
            for w in NEIGHBORS[nxt]:
                adj_count[w] += 1
                
            # Early exit if time is low
            if time_left() < 1.0:
                break
                
        return path

    # Run multiple greedy restarts
    # The first run is deterministic (greedy). Subsequent runs are randomized.
    # We can also try different starting vertices, but symmetry makes 0 sufficient if we break symmetry properly.
    # However, randomizing the choice of the first few steps can help.
    
    # Initial deterministic run
    p = greedy_search(randomize=False)
    if len(p) > best_len:
        best_len = len(p)
        best_path = p[:]

    # Randomized restarts
    # We need to finish within MAX_TIME.
    # Each greedy run is fast (O(N * L^2) worst case, but with adj_count it's O(N*L)).
    # L is at most 512. N=9. 9*512 = 4608 ops per step. 512 steps -> ~2M ops.
    # Python can do ~10M simple ops/sec. So one run takes < 0.2s.
    # We can do ~500 runs in 100s.
    
    while time_left() > 2.0:
        p = greedy_search(randomize=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
        # Optional: Local improvement on the best path found so far?
        # Given the time, simple restarts are often effective for this problem.
        # We can add a small backtracking search on the last few steps of the best path.
        
    return best_path

# Fallback just in case
def _greedy_seed():
    path, used = [0], {0}
    adj_count = [0] * (1 << N)
    for u in NEIGHBORS[0]:
        adj_count[u] += 1
    while True:
        cur = path[-1]
        nxt = None
        for u in NEIGHBORS[cur]:
            if u not in used and adj_count[u] == 1:
                nxt = u
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        for w in NEIGHBORS[nxt]:
            adj_count[w] += 1
    return path

# Ensure construct is the entry point
# The code above defines construct().
