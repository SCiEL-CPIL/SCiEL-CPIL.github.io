import random
import time
import itertools

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_EDGE_SET = set()
for _a, _b in _Q3_EDGES:
    _Q3_EDGE_SET.add((_a, _b))
    _Q3_EDGE_SET.add((_b, _a))

# Precompute Q3 adjacency as bitmasks
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    # Start with K3,13
    adj = [0] * n  # bitmask adjacency
    for u in range(3):
        for v in range(3, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph using bitmask operations."""
        # Temporarily add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # For a Q3 containing edge (u,v), u and v must map to adjacent Q3 vertices
        result = False
        for qa, qb in _Q3_EDGES:
            # Try mapping u->qa, v->qb and u->qb, v->qa
            for (gu, gv) in [(u, v), (v, u)]:
                # gu maps to qa, gv maps to qb
                unassigned = [i for i in range(8) if i != qa and i != qb]
                
                # Order unassigned Q3 vertices by constraint count (most constrained first)
                def constraint_count(x):
                    count = 0
                    for y in [qa, qb]:
                        if _Q3_ADJ[x] & (1 << y):
                            count += 1
                    return count
                
                unassigned_sorted = sorted(unassigned, key=constraint_count, reverse=True)
                
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used = set([gu, gv])
                
                def backtrack(idx):
                    if time.time() > deadline - 10:
                        return False
                    if idx == len(unassigned_sorted):
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True
                    
                    x = unassigned_sorted[idx]
                    for g in range(n):
                        if g in used:
                            continue
                        feasible = True
                        for y in range(8):
                            if assignment[y] is not None:
                                if _Q3_ADJ[x] & (1 << y):
                                    hy = assignment[y]
                                    if not (adj[g] & (1 << hy)):
                                        feasible = False
                                        break
                        if not feasible:
                            continue
                        
                        assignment[x] = g
                        used.add(g)
                        if backtrack(idx + 1):
                            return True
                        assignment[x] = None
                        used.remove(g)
                    return False
                
                if backtrack(0):
                    result = True
                    break
            if result:
                break
        
        # Remove the edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        return result
    
    # Try to add missing edges
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 15:
            break
        if not creates_q3(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
