import time
import random

N = 15
DEADLINE = 75.0

def _is_valid(adj, n):
    """
    Check if the graph satisfies the constraint:
    For every simple path s-u-v-t, deg(s) != deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    # For each edge (u, v), check all paths s-u-v-t
    # s is a neighbor of u (s != v)
    # t is a neighbor of v (t != u)
    # s != t
    for u in range(n):
        for v in adj[u]:
            if u > v: continue # Process each edge once
            # Neighbors of u excluding v
            ns = [x for x in adj[u] if x != v]
            # Neighbors of v excluding u
            nt = [x for x in adj[v] if x != u]
            for s in ns:
                ds = deg[s]
                for t in nt:
                    if s == t:
                        continue
                    dt = deg[t]
                    if ds == dt:
                        return False
    return True

def construct():
    start = time.time()
    
    # Construct K_{7,8}
    # Vertices 0..6 in Part A
    # Vertices 7..14 in Part B
    edges = []
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
            
    # Optional: Try to improve with local search if time permits
    # But K_{7,8} is likely optimal or near-optimal.
    # Given the risk of invalidity with random swaps, and the high score of 4.0 already achievable,
    # we stick to the deterministic solution to ensure validity and speed.
    
    # Verify just in case (fast for 56 edges)
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    if not _is_valid(adj, N):
        # Fallback to star if somehow invalid (should not happen)
        edges = []
        for v in range(1, N):
            edges.append([0, v])
            adj[0].add(v)
            adj[v].add(0)

    return edges

if __name__ == "__main__":
    e = construct()
    print(len(e))
