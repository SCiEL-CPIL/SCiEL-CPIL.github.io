"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random

def construct():
    n = 40
    random.seed(99)
    
    adj = [set() for _ in range(n)]
    
    def has_c6():
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                for a in adj[u]:
                    if a == v:
                        continue
                    for d in adj[v]:
                        if d == u or d == a:
                            continue
                        # Need b in adj[a], c in adj[d], b-c edge, all distinct
                        common = adj[a] & adj[d]
                        for b in common:
                            if b in (u, v, a, d):
                                continue
                            # c in adj[b] & adj[d] \ {u,v,a,d}
                            # Wait, path is u-a-b-c-d-v
                            # So b in N(a), c in N(b) ∩ N(d), c not in {u,v,a,d}
                            for c in adj[b]:
                                if c in (u, v, a, d):
                                    continue
                                if c in adj[d]:
                                    return True
        return False
    
    # Greedy: add edges in random order
    edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_list)
    
    for i, j in edges_list:
        adj[i].add(j)
        adj[j].add(i)
        if has_c6():
            adj[i].discard(j)
            adj[j].discard(i)
    
    # Second pass
    for i, j in edges_list:
        if j not in adj[i]:
            adj[i].add(j)
            adj[j].add(i)
            if not has_c6():
                pass
            else:
                adj[i].discard(j)
                adj[j].discard(i)
    
    result = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-START
# construct defined above
# EVOLVE-BLOCK-END
