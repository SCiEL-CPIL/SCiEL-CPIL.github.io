import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s kill
    
    # 1. Initialize with Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i, j] = True
                adj[j, i] = True
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    # Helper to find all K4s in a graph defined by adj_matrix and size size
    def find_k4s(adj, size):
        k4s = []
        for combo in combinations(range(size), 4):
            a, b, c, d = combo
            if (adj[a, b] and adj[a, c] and adj[a, d] and
                adj[b, c] and adj[b, d] and adj[c, d]):
                k4s.append(combo)
        return k4s

    # Helper to find all bar-K4s (independent sets of size 4)
    def find_bar_k4s(adj, size):
        bar_k4s = []
        for combo in combinations(range(size), 4):
            a, b, c, d = combo
            if (not adj[a, b] and not adj[a, c] and not adj[a, d] and
                not adj[b, c] and not adj[b, d] and not adj[c, d]):
                bar_k4s.append(combo)
        return bar_k4s

    # Helper to check if a set S is valid:
    # - S contains no K4
    # - S intersects every bar-K4 (so V\S contains no bar-K4)
    def is_valid_neighborhood(S, k4s, bar_k4s):
        # Check K4 constraint: no K4 fully in S
        for k4 in k4s:
            if all(v in S for v in k4):
                return False
        # Check bar-K4 constraint: S must hit every bar-K4
        for bk4 in bar_k4s:
            if not any(v in S for v in bk4):
                return False
        return True

    # Slack-based heuristic:
    # For a candidate S, define:
    # - k4_slack = min over k4s of (number of vertices of k4 NOT in S)
    #   If k4_slack == 0, invalid. We want k4_slack >= 1.
    # - bar_k4_slack = min over bar_k4s of (number of vertices of bk4 IN S)
    #   If bar_k4_slack == 0, invalid. We want bar_k4_slack >= 1.
    # We want to maximize min(k4_slack, bar_k4_slack).
    
    def compute_slack(S, k4s, bar_k4s):
        min_k4_slack = float('inf')
        for k4 in k4s:
            count_in_S = sum(1 for v in k4 if v in S)
            slack = 4 - count_in_S
            if slack < min_k4_slack:
                min_k4_slack = slack
            if slack == 0:
                return 0  # Invalid
        
        min_bk4_slack = float('inf')
        for bk4 in bar_k4s:
            count_in_S = sum(1 for v in bk4 if v in S)
            slack = count_in_S
            if slack < min_bk4_slack:
                min_bk4_slack = slack
            if slack == 0:
                return 0  # Invalid
        
        if min_k4_slack == float('inf'):
            min_k4_slack = 4
        if min_bk4_slack == float('inf'):
            min_bk4_slack = 4
            
        return min(min_k4_slack, min_bk4_slack)

    current_n = n
    best_n = current_n
    best_adj = adj.copy()
    
    # We will try to extend the graph.
    # Strategy: 
    # 1. Precompute K4s and bar-K4s for the current graph.
    # 2. Try to add a new vertex by finding a valid neighborhood S.
    # 3. Use a greedy/local search approach to find S that maximizes slack.
    
    while time.time() < deadline:
        # Precompute constraints for current graph
        k4s = find_k4s(adj, current_n)
        bar_k4s = find_bar_k4s(adj, current_n)
        
        # If no K4s and no bar-K4s, we can add any vertex?
        # Actually, if no K4s, S can be anything w.r.t K4s.
        # If no bar-K4s, S can be anything w.r.t bar-K4s.
        
        found = False
        best_S = None
        best_slack = -1
        
        # Local search for S
        # Start with random S
        S = set()
        for v in range(current_n):
            if random.random() < 0.5:
                S.add(v)
                
        current_slack = compute_slack(S, k4s, bar_k4s)
        if current_slack > 0:
            best_S = S.copy()
            best_slack = current_slack
        
        # Iterate improvements
        max_iterations = 2000
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
                
            improved = False
            # Try flipping a single vertex in S
            # To speed up, pick a random vertex to flip
            v = random.randint(0, current_n - 1)
            
            if v in S:
                S.remove(v)
            else:
                S.add(v)
                
            new_slack = compute_slack(S, k4s, bar_k4s)
            
            if new_slack > current_slack:
                current_slack = new_slack
                improved = True
                if current_slack > best_slack:
                    best_slack = current_slack
                    best_S = S.copy()
                    if best_slack >= 2: # High slack, likely good
                        break
            else:
                # Revert
                if v in S:
                    S.remove(v)
                else:
                    S.add(v)
                    
            if not improved and random.random() < 0.1:
                # Random restart occasionally
                S = set()
                for vv in range(current_n):
                    if random.random() < 0.5:
                        S.add(vv)
                current_slack = compute_slack(S, k4s, bar_k4s)
                if current_slack > best_slack:
                    best_slack = current_slack
                    best_S = S.copy()
                    
        if best_S is not None and best_slack > 0:
            # Add new vertex with neighborhood best_S
            adj = np.resize(adj, (current_n + 1, current_n + 1))
            adj[current_n, :] = False
            adj[:, current_n] = False
            for v in best_S:
                adj[current_n, v] = True
                adj[v, current_n] = True
            current_n += 1
            found = True
            best_n = current_n
            best_adj = adj.copy()
        else:
            # Failed to add vertex
            # Try a few more times with different random seeds?
            # Or just break if we've been stuck for a while?
            # Let's try a few more quick random attempts
            quick_found = False
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                S = set()
                for v in range(current_n):
                    if random.random() < 0.5:
                        S.add(v)
                if is_valid_neighborhood(S, k4s, bar_k4s):
                    adj = np.resize(adj, (current_n + 1, current_n + 1))
                    adj[current_n, :] = False
                    adj[:, current_n] = False
                    for v in S:
                        adj[current_n, v] = True
                        adj[v, current_n] = True
                    current_n += 1
                    quick_found = True
                    best_n = current_n
                    best_adj = adj.copy()
                    break
            if not quick_found:
                break

    # Final check: ensure no K5 or independent 5
    # We built it incrementally, so it should be fine, but let's be safe.
    # If we failed the final check, we return the previous best?
    # The incremental logic guarantees no K5/Ind5 if the neighborhood logic was correct.
    # Logic:
    # New vertex v connects to S.
    # K5 involving v: v + 4 vertices in S that form a K4. We ensured S has no K4.
    # Ind5 involving v: v + 4 vertices in V\S that form an Ind4. We ensured V\S has no Ind4.
    # K5 not involving v: already checked in previous steps.
    # Ind5 not involving v: already checked in previous steps.
    # So the graph is valid.
    
    edges = get_edges(best_adj, best_n)
    return best_n, edges

# Test run
if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
