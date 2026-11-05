import time
import random
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = []
    best_count = 0
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def creates_c6(adj, u, v):
        """
        Checks if adding edge (u,v) to graph adj creates a C6.
        Assumes (u,v) is NOT in adj.
        Returns a list of edges forming the C6 if found, else None.
        """
        nu = adj[u]
        nv = adj[v]
        
        # Path u-a-b-c-d-v
        for a in nu:
            for d in nv:
                if a == d:
                    continue
                
                # b is neighbor of a
                # c is neighbor of d
                # (b,c) is edge
                # b, c distinct from u, v, a, d and each other
                
                # Optimize: 
                # Candidates for b: adj[a] \ {u, v, d}
                # Candidates for c: adj[d] \ {u, v, a}
                
                # Iterate b
                for b in adj[a]:
                    if b == u or b == v or b == d:
                        continue
                    
                    # Check if any neighbor c of b is in adj[d] and valid
                    # c must be in adj[b] and adj[d]
                    # c != u, v, a, b, d
                    for c in adj[b]:
                        if c == u or c == v or c == a or c == b or c == d:
                            continue
                        if c in adj[d]:
                            # Found C6
                            # Edges: (u,a), (a,b), (b,c), (c,d), (d,v)
                            return [(u, a), (a, b), (b, c), (c, d), (d, v)]
        return None

    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Greedy Init
        adj = [set() for _ in range(N)]
        edges = []
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = [list(e) for e in edges]
            
        # SA Phase
        T = 1.0
        cooling = 0.995
        min_T = 0.05
        
        # Precompute non-edges? 
        # Just random pick.
        
        while T > min_T and time.time() < deadline - 10:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v:
                continue
            if u in adj[v]:
                continue
            
            c6_edges = creates_c6(adj, u, v)
            
            if c6_edges is None:
                # Valid addition
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [list(e) for e in edges]
            else:
                # Invalid addition. Try swap.
                # Pick random edge from C6
                e_to_remove = random.choice(c6_edges)
                
                x, y = e_to_remove
                adj[x].remove(y)
                adj[y].remove(x)
                edges.remove(e_to_remove)
                
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
                # Verify validity of new state
                # Check if (u,v) is in any C6
                if creates_c6(adj, u, v) is not None:
                    # Revert
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges.remove((u, v))
                    
                    adj[x].add(y)
                    adj[y].add(x)
                    edges.append(e_to_remove)
                else:
                    # Valid swap, count unchanged
                    pass
            
            T *= cooling

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
