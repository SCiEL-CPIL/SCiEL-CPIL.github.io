import numpy as np
import time
import random
import sys
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: adj[i] = set of j where (i-j) mod m or (j-i) mod m is in forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            d = (i - j) % m
            if d in forbidden or (m - d) % m in forbidden:
                adj[i].add(j)
                adj[j].add(i)
    
    # Degree ordering for backtracking
    order = sorted(range(m), key=lambda v: -len(adj[v]))
    # Remap
    pos = {v: i for i, v in enumerate(order)}
    remap = [0] * m
    for v in range(m):
        remap[v] = pos[v]
    
    # Build remapped adjacency
    n = m
    adj2 = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[order[i]]:
            adj2[i].add(pos[j])
    
    # Bron-Kerbosch style maximum independent set via complement
    # Actually, let's use a branch-and-bound with good upper bounds
    
    best_set = []
    best_size = 0
    
    # Greedy initial solution
    def greedy():
        s = set()
        for v in range(n):
            ok = True
            for u in s:
                if u in adj2[v]:
                    ok = False
                    break
            if ok:
                s.add(v)
        return s
    
    g = greedy()
    best_size = len(g)
    best_set = list(g)
    
    # Simulated annealing / local search on the 205-vertex graph
    # State: a set S (independent set). Moves: add a vertex, remove a vertex, swap.
    
    def is_valid_set(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if s[j] in adj2[s[i]]:
                    return False
        return True
    
    def can_add(s, v):
        for u in s:
            if u in adj2[v]:
                return False
        return True
    
    # Randomized local search with multiple restarts
    random.seed(12345)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Start from a random greedy or perturbed greedy
        if restart == 0:
            current = set(g)
        else:
            # Random greedy with random ordering
            verts = list(range(n))
            random.shuffle(verts)
            current = set()
            for v in verts:
                if can_add(current, v):
                    current.add(v)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        # Local search: try to improve by swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any vertex
            for v in range(n):
                if v not in current and can_add(current, v):
                    current.add(v)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = list(current)
                    break
            if improved:
                continue
            
            # Try 1-for-1 swaps: remove u, add v where v conflicts only with u
            found_swap = False
            for u in list(current):
                for v in range(n):
                    if v in current:
                        continue
                    # Check if v can be added after removing u
                    conflicts_with_others = False
                    for w in current:
                        if w != u and w in adj2[v]:
                            conflicts_with_others = True
                            break
                    if not conflicts_with_others:
                        current.discard(u)
                        current.add(v)
                        found_swap = True
                        improved = True
                        # After swap, try to add more
                        for v2 in range(n):
                            if v2 not in current and can_add(current, v2):
                                current.add(v2)
                                if len(current) > best_size:
                                    best_size = len(current)
                                    best_set = list(current)
                                break
                        break
                if found_swap:
                    break
            
            if not found_swap:
                # Try 2-for-1: remove two, add one
                if len(current) >= 2:
                    cl = list(current)
                    random.shuffle(cl)
                    for i in range(min(20, len(cl))):
                        for j in range(i+1, min(len(cl), i+20)):
                            u1, u2 = cl[i], cl[j]
                            for v in range(n):
                                if v in current:
                                    continue
                                # v must not conflict with current \ {u1, u2}
                                ok = True
                                for w in current:
                                    if w != u1 and w != u2 and w in adj2[v]:
                                        ok = False
                                        break
                                if ok:
                                    current.discard(u1)
                                    current.discard(u2)
                                    current.add(v)
                                    improved = True
                                    # Try to add more
                                    for v2 in range(n):
                                        if v2 not in current and can_add(current, v2):
                                            current.add(v2)
                                            if len(current) > best_size:
                                                best_size = len(current)
                                                best_set = list(current)
                                            break
                                    break
                            if improved:
                                break
                        if improved:
                            break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        # Perturbation: remove random vertices and re-add greedily
        if time.time() > deadline:
            break
        if restart > 0 and restart % 50 == 0:
            # Aggressive restart
            verts = list(range(n))
            random.shuffle(verts)
            current = set()
            for v in verts:
                if can_add(current, v):
                    current.add(v)
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
    
    # Final verification
    result = sorted(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d1 = (result[i] - result[j]) % m
            d2 = (result[j] - result[i]) % m
            if d1 in forbidden or d2 in forbidden:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: return a known valid small set
        # Single element is always valid
        return np.array([0], dtype=np.int64)
    
    print(f"Best size: {best_size}, time: {time.time()-start_time:.1f}s", file=sys.stderr)
    return np.array(result, dtype=np.int64)
