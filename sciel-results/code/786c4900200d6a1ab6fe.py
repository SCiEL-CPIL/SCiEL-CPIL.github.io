import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    n = 15
    all_edges = list(combinations(range(n), 2))
    
    def check_valid(edge_set):
        """Check if edge_set satisfies the constraint."""
        if time.time() > deadline:
            return False
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for u, v in edge_set:
            # For path s-u-v-t where s≠t, need deg(s) != deg(t)
            # s is in N(u)\{v}, t is in N(v)\{u}
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def count_violations(edge_set):
        """Count number of violating paths (for optimization)."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        violations = 0
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        violations += 1
        return violations
    
    # Greedy construction: add edges that don't create violations
    best_edges = set()
    best_count = 0
    
    for _ in range(50):  # Multiple random restarts
        if time.time() > deadline:
            break
        
        random.seed(_ * 12345 + 678)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        current_edges = set()
        
        for e in shuffled:
            if time.time() > deadline:
                break
            candidate = current_edges | {e}
            if check_valid(candidate):
                current_edges = candidate
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
        
        # Local search: try to add more edges
        if time.time() > deadline:
            break
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in all_edges:
                if e in best_edges:
                    continue
                candidate = best_edges | {e}
                if check_valid(candidate):
                    best_edges = candidate
                    best_count += 1
                    improved = True
                    break
        
        # Try removing one edge and adding two others (2-opt style)
        if time.time() > deadline:
            break
        
        for _ in range(200):
            if time.time() > deadline:
                break
            if len(best_edges) < 2:
                break
            e1, e2 = random.sample(list(best_edges), 2)
            candidate = (best_edges - {e1, e2})
            # Try to add two new edges
            added = 0
            for e in all_edges:
                if e in candidate or e == e1 or e == e2:
                    continue
                test = candidate | {e}
                if check_valid(test):
                    candidate = test
                    added += 1
                    if added == 2:
                        break
            if added == 2 and len(candidate) > len(best_edges):
                best_edges = candidate
                best_count = len(candidate)
    
    # Final local search polish
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for e in all_edges:
                if e in best_edges:
                    continue
                candidate = best_edges | {e}
                if check_valid(candidate):
                    best_edges = candidate
                    best_count += 1
                    improved = True
    
    # Convert to list of [u, v] with proper types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
