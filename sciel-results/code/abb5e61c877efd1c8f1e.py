import time
import random
import math

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    """
    Check if no simple path s-u-v-t (s != t) has deg(s) == deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # Collect s_list and t_list
            s_list = [s for s in adj[u] if s != v]
            t_list = [t for t in adj[v] if t != u]
            
            # Build degree-to-vertex mapping for each side
            deg_s = {}
            for s in s_list:
                d = deg[s]
                if d not in deg_s:
                    deg_s[d] = []
                deg_s[d].append(s)
            
            deg_t = {}
            for t in t_list:
                d = deg[t]
                if d not in deg_t:
                    deg_t[d] = []
                deg_t[d].append(t)
            
            # Check common degrees
            for d in deg_s:
                if d in deg_t:
                    # Check if there exist s in deg_s[d], t in deg_t[d] with s != t
                    found = False
                    for s in deg_s[d]:
                        for t in deg_t[d]:
                            if s != t:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def _make_star(n):
    adj = [set() for _ in range(n)]
    for v in range(1, n):
        adj[0].add(v)
        adj[v].add(0)
    return adj

def _local_search(adj, n, seed, max_iter, deadline):
    """Local search: add/remove edges, accept if valid and improves."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    current_valid = _is_valid(adj, n)
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges if current_valid else 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        if j in adj[i]:
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            adj[i].add(j)
            adj[j].add(i)
            
        if _is_valid(adj, n):
            new_edges = _edge_count(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
        else:
            # Revert
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
                
    return best_adj, best_edges

def construct():
    start = time.time()
    
    # Start with a star graph (14 edges)
    best_adj = _make_star(N)
    best_count = _edge_count(best_adj, N)
    
    # Verify
    if not _is_valid(best_adj, N):
        # Fallback to empty or simple valid graph
        best_adj = [set() for _ in range(N)]
        best_count = 0

    # Local search from multiple starts
    deadline = start + DEADLINE - 5
    
    # Start from best_adj
    for seed in range(100):
        if time.time() > deadline:
            break
            
        perturbed = [set(s) for s in best_adj]
        rng = random.Random(seed)
        # Remove 1-3 random edges to explore
        for _ in range(rng.randint(1, 4)):
            all_edges = []
            for i in range(N):
                for j in perturbed[i]:
                    if i < j:
                        all_edges.append((i, j))
            if all_edges:
                a, b = rng.choice(all_edges)
                perturbed[a].discard(b)
                perturbed[b].discard(a)
        
        result, cnt = _local_search(perturbed, N, 1000 + seed, 5000, deadline)
        
        if cnt > best_count and _is_valid(result, N):
            best_count = cnt
            best_adj = result
    
    # Final check
    if not _is_valid(best_adj, N):
        best_adj = _make_star(N)
        best_count = 14

    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
