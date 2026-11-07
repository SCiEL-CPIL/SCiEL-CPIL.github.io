"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_START_TIME = time.time()
_TIME_LIMIT = 85.0  # Safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _bfs_distances(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution(n, base_edges, added_edges):
    edge_set = set()
    for u, v in base_edges:
        edge_set.add((min(u, v), max(u, v)))
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False, 0, 0
        edge_set.add(key)
    
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check
    # Since we only add E-O edges to a path (bipartite), it's always triangle-free.
    # But we verify just in case.
    for i in range(n):
        for j in adj[i]:
            for k in adj[i]:
                if j < k and k in adj[j]:
                    return False, 0, 0
    
    # Connected + diameter
    max_diam = 0
    for start in range(n):
        dist = _bfs_distances(adj, n, start)
        if -1 in dist:
            return False, 0, 0
        d = max(dist)
        if d > max_diam:
            max_diam = d
    
    if max_diam > 4:
        return False, max_diam, len(added_edges)
        
    return True, max_diam, len(added_edges)

def construct():
    n = P['n']
    base = P['edges']
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
    
    # Generate all valid candidate edges (bipartite: even-odd)
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in base_set:
                if i % 2 != j % 2:
                    candidates.append((i, j))
    
    # Initial seed: Try a few promising configurations
    # Configuration 1: (0, 11), (11, 22)
    seed1 = [[0, 11], [11, 22]]
    # Configuration 2: (0, 7), (7, 14), (14, 23)
    seed2 = [[0, 7], [7, 14], [14, 23]]
    # Configuration 3: (0, 9), (9, 18), (18, 23)
    seed3 = [[0, 9], [9, 18], [18, 23]]
    
    best_solution = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in [seed1, seed2, seed3]:
        valid, diam, count = _check_solution(n, base, seed)
        if valid:
            if count < best_score:
                best_score = count
                best_solution = [e[:] for e in seed]
        
    # Local Search
    # We will perform a random walk / hill climbing
    # State: list of added edges
    # Objective: Minimize count, subject to valid.
    # If invalid, penalty is huge.
    
    def get_score(added_edges):
        valid, diam, count = _check_solution(n, base, added_edges)
        if not valid:
            # Penalty for invalidity
            # If disconnected, penalty 1000
            # If diameter > 4, penalty 100 + (diam - 4)
            # We need to distinguish. _check_solution returns False for both.
            # Let's make a more detailed check for scoring in search.
            pass
        return count if valid else float('inf')

    # Better scoring function for search
    def search_score(added_edges):
        edge_set = set(base_set)
        for u, v in added_edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return float('inf')
            edge_set.add(key)
        
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check triangle-free (should be ok)
        for i in range(n):
            for j in adj[i]:
                for k in adj[i]:
                    if j < k and k in adj[j]:
                        return float('inf')
        
        # Check connectivity and diameter
        max_diam = 0
        connected = True
        for start in range(n):
            dist = _bfs_distances(adj, n, start)
            if -1 in dist:
                connected = False
                break
            d = max(dist)
            if d > max_diam:
                max_diam = d
        
        if not connected:
            return 1000 + len(added_edges)
        
        if max_diam > 4:
            return 100 + (max_diam - 4) * 10 + len(added_edges)
            
        return len(added_edges)

    # Start from best seed or empty
    if best_solution:
        current = [e[:] for e in best_solution]
    else:
        current = []
    
    current_score = search_score(current)
    
    # Candidate edges for moves
    # We only consider adding edges not in base and not already added
    # And removing existing added edges
    
    iterations = 0
    while _time_ok() and iterations < 20000:
        iterations += 1
        
        # Generate a random neighbor
        new_state = [e[:] for e in current]
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Add a random valid edge
            # Filter candidates not in current
            current_keys = set((min(u,v), max(u,v)) for u,v in current)
            available = [c for c in candidates if (min(c[0],c[1]), max(c[0],c[1])) not in current_keys]
            if not available:
                continue
            new_edge = random.choice(available)
            new_state.append(list(new_edge))
            
        elif move_type == 'remove':
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            new_state.pop(idx)
            
        elif move_type == 'swap':
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            current_keys = set((min(u,v), max(u,v)) for u,v in current)
            available = [c for c in candidates if (min(c[0],c[1]), max(c[0],c[1])) not in current_keys]
            if not available:
                continue
            new_edge = random.choice(available)
            new_state[idx] = list(new_edge)
            
        new_score = search_score(new_state)
        
        # Accept if better, or with probability if worse (simulated annealing style)
        # Simple hill climbing: accept if better
        if new_score < current_score:
            current = new_state
            current_score = new_score
            
            # If we found a valid solution with fewer edges, update best
            if current_score < 100: # Valid solution score is just count
                if current_score < best_score:
                    best_score = current_score
                    best_solution = [e[:] for e in current]
                    
            # If we reached 3 edges and it's valid, it's likely optimal? 
            # Minimum possible is 1? No, diameter 4 on 24 nodes needs more.
            # If best_score is 3, we can stop early? 
            # Let's keep searching to ensure no 2-edge solution exists (unlikely).
            if best_score <= 3:
                break

        # Random restarts if stuck?
        # If no improvement in 1000 steps, restart from a seed
        # For simplicity, we just keep going.

    # Final check
    if best_solution:
        # Ensure types are correct
        result = []
        for u, v in best_solution:
            result.append([int(u), int(v)])
        return result
    else:
        # Fallback: return empty list (invalid) or try a known valid one
        # The seeds should have provided a valid one.
        # If not, return the best valid one found during search?
        # search_score returns < 100 for valid.
        # If best_solution is None, it means no valid solution was found.
        # This shouldn't happen with the seeds.
        return []

# EVOLVE-BLOCK-END
