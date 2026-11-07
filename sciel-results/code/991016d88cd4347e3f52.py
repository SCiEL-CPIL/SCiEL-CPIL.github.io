"""Constraint-satisfaction approach: assign vertices to degree groups, then build edges between groups that guarantee validity."""
import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 15}


def _check_valid(adj, n):
    """Check if graph is valid: no 3-edge path has equal-degree endpoints."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True


def _try_add_edge(adj, n, i, j):
    """Try adding edge (i,j), return True if valid."""
    adj[i].add(j)
    adj[j].add(i)
    ok = _check_valid(adj, n)
    if not ok:
        adj[i].discard(j)
        adj[j].discard(i)
    return ok


def _try_remove_edge(adj, n, i, j):
    """Remove edge (i,j) and return new edge count."""
    adj[i].discard(j)
    adj[j].discard(i)
    return sum(len(s) for s in adj) // 2


def _greedy_with_constraints(adj, n, rng, deadline):
    """Greedy edge addition with constraint checking."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    count = 0
    for (i, j) in edges:
        if time.time() > deadline:
            break
        if _try_add_edge(adj, n, i, j):
            count += 1
    return adj, count


def _local_search(adj, n, rng, deadline, max_iters=500):
    """Local search with add and swap moves."""
    cur_adj = [set(s) for s in adj]
    cur_count = sum(len(s) for s in cur_adj) // 2
    best_adj = [set(s) for s in cur_adj]
    best_count = cur_count
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding missing edges
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if j not in cur_adj[i]]
        rng.shuffle(missing)
        improved = False
        
        for (i, j) in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(cur_adj, n, i, j):
                cur_count += 1
                improved = True
                if cur_count > best_count:
                    best_count = cur_count
                    best_adj = [set(s) for s in cur_adj]
                    no_improve = 0
                break
        
        if not improved:
            # Try swaps: remove one edge, add another
            existing = [(i, j) for i in range(n) for j in cur_adj[i] if i < j]
            rng.shuffle(existing)
            for (a, b) in existing:
                if time.time() > deadline:
                    break
                # Try removing (a,b) and adding a different edge
                cur_adj[a].discard(b)
                cur_adj[b].discard(a)
                temp_count = cur_count - 1
                
                # Find a missing edge to add
                for (c, d) in missing:
                    if (c == a and d == b) or (c == b and d == a):
                        continue
                    if _try_add_edge(cur_adj, n, c, d):
                        if temp_count + 1 >= cur_count:
                            cur_count = temp_count + 1
                            if cur_count > best_count:
                                best_count = cur_count
                                best_adj = [set(s) for s in cur_adj]
                                no_improve = 0
                                improved = True
                        break
                    else:
                        cur_adj[c].discard(d)
                        cur_adj[d].discard(c)
                
                if not improved:
                    # Restore original edge
                    cur_adj[a].add(b)
                    cur_adj[b].add(a)
                    cur_count = temp_count + 1
                
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
    
    return best_adj, best_count


def construct():
    n = P['n']
    start = time.time()
    deadline = start + 45.0  # safety margin
    rng = random.Random(42)
    
    best_adj = None
    best_count = -1
    
    # Multiple random restarts with greedy + local search
    for k in range(20):
        if time.time() > deadline - 10.0:
            break
        
        adj = [set() for _ in range(n)]
        adj, cnt = _greedy_with_constraints(adj, n, rng, deadline - 5.0)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
        
        # Refine with local search
        adj2, cnt2 = _local_search(adj, n, rng, deadline - 2.0, max_iters=300)
        if cnt2 > best_count:
            best_count = cnt2
            best_adj = [set(s) for s in adj2]
    
    if best_adj is None:
        best_adj = [set() for _ in range(n)]
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges
# EVOLVE-BLOCK-END
