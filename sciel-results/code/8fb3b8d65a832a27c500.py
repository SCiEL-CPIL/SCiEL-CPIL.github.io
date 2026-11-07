import time
import random
import sys

def _is_valid_fast(n, edges, adj, deg):
    """
    Check if the graph defined by edges, adj, deg is valid.
    Valid means: no 3-edge path s-u-v-t has deg[s] == deg[t].
    """
    # Check all 3-paths
    # s-u-v-t
    # u, v are adjacent
    # s is adjacent to u, s != v
    # t is adjacent to v, t != u
    # s != t
    
    for v in range(n):
        for u in adj[v]:
            # u-v is an edge
            # s in adj[u], s != v
            # t in adj[v], t != u
            # if s == t, skip
            # if deg[s] == deg[t], invalid
            
            # Optimization: Group neighbors by degree?
            # For a fixed u, v:
            # We need to check if there exists s in N(u)\{v} and t in N(v)\{u}
            # such that s != t and deg[s] == deg[t].
            
            # Let S = {deg[s] for s in adj[u] if s != v}
            # Let T = {deg[t] for t in adj[v] if t != u}
            # If S and T share a value d, we need to check if there are s,t with that degree and s != t.
            
            # This is still O(deg(u)*deg(v)) if done naively.
            # But we can use sets.
            
            degs_u = set()
            for s in adj[u]:
                if s != v:
                    degs_u.add(deg[s])
            
            degs_v = set()
            for t in adj[v]:
                if t != u:
                    degs_v.add(deg[t])
            
            common_degs = degs_u.intersection(degs_v)
            if common_degs:
                # We have a common degree value d.
                # We need to verify if there exist s in N(u)\{v}, t in N(v)\{u} with deg[s]=deg[t]=d and s != t.
                # If for some d, the only s with deg d is the same as the only t with deg d, and s==t, then it's ok?
                # No, if s==t, the path is not simple (s!=t required).
                # So if s==t, that specific pair is fine.
                # But if there are multiple, or if the unique ones are different, it's bad.
                
                for d in common_degs:
                    # Find all s in N(u)\{v} with deg[s] == d
                    candidates_s = [s for s in adj[u] if s != v and deg[s] == d]
                    # Find all t in N(v)\{u} with deg[t] == d
                    candidates_t = [t for t in adj[v] if t != u and deg[t] == d]
                    
                    # Check if there is a pair s, t with s != t
                    # If len(candidates_s) > 1 or len(candidates_t) > 1, likely yes.
                    # Specifically:
                    # If there exists s in candidates_s and t in candidates_t such that s != t.
                    # This fails only if candidates_s == candidates_t == [x] (single element x).
                    if len(candidates_s) == 1 and len(candidates_t) == 1:
                        if candidates_s[0] != candidates_t[0]:
                            return False
                    else:
                        # If one set has >1, or both have 1 but different, or one >1...
                        # Actually, if len(candidates_s) >= 2, we can pick s1, s2.
                        # If len(candidates_t) >= 1, pick t.
                        # If s1 != t, good. If s1 == t, pick s2. If s2 != t, good.
                        # So if len(candidates_s) >= 2, it's always bad?
                        # Only if all candidates_s are equal to all candidates_t?
                        # No, if candidates_s = {1, 2} and candidates_t = {1}, then pair (2,1) has 2!=1. Bad.
                        # Pair (1,1) has 1==1, ok. But we just need ONE bad pair to invalidate.
                        # So if there exists ANY pair with s!=t, it's invalid.
                        
                        # The only case where NO pair has s!=t is if candidates_s == candidates_t == {x}.
                        # We already handled that.
                        return False
    return True

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Initial seed: K7,8
    # Partitions: 0-6 (L), 7-14 (R)
    edges = []
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    # Add bipartite edges
    for i in range(7):
        for j in range(7, 15):
            edges.append((i, j))
            adj[i].add(j)
            adj[j].add(i)
            deg[i] += 1
            deg[j] += 1
    
    # Now add intra-partition edges greedily
    # Candidates: all edges within 0-6 and within 7-14
    candidates = []
    for i in range(7):
        for j in range(i+1, 7):
            candidates.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidates.append((i, j))
    
    # Randomize order for diversity? Or try all?
    # We want to maximize edges.
    # Greedy: try to add each candidate if valid.
    
    # To improve, we can do multiple passes or random restarts.
    # But K7,8 is a strong start.
    
    # Shuffle candidates to avoid bias
    random.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline - 5:
            break
            
        # Try adding (u, v)
        # Update
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append((u, v))
        
        if not _is_valid_fast(n, edges, adj, deg):
            # Rollback
            edges.pop()
            deg[u] -= 1
            deg[v] -= 1
            adj[u].remove(v)
            adj[v].remove(u)
    
    # Local search: try to add back rejected edges in different order?
    # Or try swaps?
    # Given the time, the greedy on K7,8 is likely sufficient to beat 3.14.
    # 3.14 * 105 (max edges) approx 330? No, score is value/seed.
    # If seed is ~10, value 31 -> 3.1.
    # K7,8 has 56 edges.
    # If we add ~10 intra edges, we get 66.
    # 66 / 10 = 6.6.
    
    result = [[u, v] for u, v in edges]
    return result
