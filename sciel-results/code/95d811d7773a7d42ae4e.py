import time
import random

def construct(k, g):
    # Petersen Graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Build adjacency list for Petersen graph
    adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Precompute all closed walks of length 5 to 15 in Petersen graph
    # Each walk is represented as a list of (edge_index, sign)
    # sign is +1 if traversed u->v, -1 if v->u (relative to petersen_edges definition)
    closed_walks = []
    
    def find_closed_walks(start, current, depth, max_depth, path):
        if depth > 4: # min girth 5, so depth in DFS from start is >= 4 to close
            if current == start:
                closed_walks.append(path[:])
        if depth >= max_depth:
            return
        for neighbor, edge_idx in adj[current]:
            # Determine sign
            u, v = petersen_edges[edge_idx]
            if current == u:
                sign = 1
            else:
                sign = -1
            find_closed_walks(start, neighbor, depth + 1, max_depth, path + [(edge_idx, sign)])

    # We need walks of length L where 5 <= L <= 15
    # A walk of length L starting at start and ending at start has L edges.
    # In the recursive function, depth is the number of edges taken.
    # So we call with depth=0 and check when depth is in [5, 15] and current==start.
    # However, the standard DFS above checks `if current == start` at the beginning of the call?
    # No, let's adjust.
    
    closed_walks = []
    
    def dfs_walks(start, current, depth, path):
        if depth >= 5 and depth <= 15:
            if current == start:
                closed_walks.append(path[:])
        if depth >= 15:
            return
        for neighbor, edge_idx in adj[current]:
            u, v = petersen_edges[edge_idx]
            if current == u:
                sign = 1
            else:
                sign = -1
            dfs_walks(start, neighbor, depth + 1, path + [(edge_idx, sign)])
            
    for s in range(10):
        dfs_walks(s, s, 0, [])
        
    # Remove duplicates? Not strictly necessary for checking, but good for speed.
    # The number of walks is small enough.
    # Let's estimate: Branching factor 3, depth 15. 3^15 is ~14M.
    # But we only keep closed walks.
    # For Petersen graph, number of closed walks of length L is small.
    # Length 5: 120 (24 cycles * 5 start points * 2 directions? No, 12 5-cycles, each has 10 closed walks (5 starts, 2 dirs). Total 120.)
    # Length 6: ...
    # It should be manageable.

    start_time = time.time()
    
    # Moore bound for k=3, g=16 is 510. n=10m >= 512 => m >= 52.
    for m in range(52, 500):
        if time.time() - start_time > 100:
            break
            
        # Generate Thue-Morse derived voltage patterns
        # Thue-Morse sequence
        def get_thue_morse(n):
            return [bin(i).count('1') % 2 for i in range(n)]
        
        tm = get_thue_morse(20)
        
        patterns = []
        
        # Pattern 1: Linear with Thue-Morse offset
        # v_i = (i * A + tm[i] * B) % m
        for A in [1, 2, 3, 5, 7, 11, 13]:
            for B in [1, 2, 3, 5, 7]:
                patterns.append([(i * A + tm[i] * B) % m for i in range(15)])
                
        # Pattern 2: Thue-Morse scaling
        patterns.append([(tm[i] * (m // 2 + i)) % m for i in range(15)])
        patterns.append([(tm[i] * (m // 3 + i * 2)) % m for i in range(15)])
        
        # Pattern 3: Random seeded by m
        for seed in range(5):
            random.seed(m * 100 + seed)
            patterns.append([random.randint(1, m - 1) for _ in range(15)])
            
        for vol in patterns:
            # Check if any closed walk has sum 0 mod m
            valid = True
            for walk in closed_walks:
                s = 0
                for e_idx, sign in walk:
                    s += sign * vol[e_idx]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct edges
                edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    w = vol[i] % m
                    for s in range(m):
                        u_id = u * m + s
                        v_id = v * m + (s + w) % m
                        if u_id < v_id:
                            edges.append((u_id, v_id))
                        else:
                            edges.append((v_id, u_id))
                return 10 * m, edges
                
    # Fallback
    m = 500
    random.seed(42)
    vol = [random.randint(1, m - 1) for _ in range(15)]
    edges = []
    for i, (u, v) in enumerate(petersen_edges):
        w = vol[i] % m
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            if u_id < v_id:
                edges.append((u_id, v_id))
            else:
                edges.append((v_id, u_id))
    return 10 * m, edges
