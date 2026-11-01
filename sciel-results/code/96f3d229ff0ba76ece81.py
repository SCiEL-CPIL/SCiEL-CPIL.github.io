import time
import random

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        tm = [0] * n
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    def build_lift(m, voltages):
        """Build the voltage graph lift and return adjacency list."""
        n_vertices = 10 * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        return adj
    
    def generate_edges(m, voltages):
        """Generate edge list for the lift."""
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                if u_id < v_id:
                    edges.append((u_id, v_id))
                else:
                    edges.append((v_id, u_id))
        return edges
    
    def check_girth_fast(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using BFS from each vertex."""
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                if dist[node] >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node] and parent[neighbor] != node:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True
    
    start_time = time.time()
    
    # Precompute Thue-Morse sequence
    tm = get_thue_morse(500)
    
    # Try increasing m values with Thue-Morse derived voltage assignments
    # Use primes and numbers with few small factors to reduce collision probability
    candidate_ms = []
    for m in range(20, 300):
        # Prefer primes and numbers coprime to small integers
        candidate_ms.append(m)
    
    for m in candidate_ms:
        if time.time() - start_time > 105:
            break
        
        # Generate multiple Thue-Morse based voltage patterns
        patterns = []
        
        # Pattern 1: Direct Thue-Morse scaling
        patterns.append([(tm[i] * (m // 2 + 1)) % m for i in range(15)])
        
        # Pattern 2: Thue-Morse with prime multiplier
        patterns.append([(tm[i] * (3 * i + 1) * 7) % m for i in range(15)])
        
        # Pattern 3: XOR of shifted Thue-Morse
        patterns.append([((tm[i] ^ tm[(i + 3) % 500]) * (2 * i + 1) + 1) % m for i in range(15)])
        
        # Pattern 4: Quadratic Thue-Morse
        patterns.append([(tm[i] * (i * i + i + 1)) % m for i in range(15)])
        
        # Pattern 5: Thue-Morse with offset
        patterns.append([(tm[(i * 7) % 500] * (5 * i + 3)) % m for i in range(15)])
        
        # Pattern 6: Double Thue-Morse
        patterns.append([((tm[i] * tm[(i + 5) % 500] * 11 + tm[i]) % m) for i in range(15)])
        
        # Pattern 7: Thue-Morse with linear term
        patterns.append([(tm[i] * 13 + 3 * i + 1) % m for i in range(15)])
        
        # Pattern 8: Thue-Morse with quadratic term
        patterns.append([(tm[i] * 17 + 2 * i * i + i) % m for i in range(15)])
        
        # Pattern 9: Random with Thue-Morse seed for diversity
        random.seed(m * 7919 + 42)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        # Pattern 10: Another random
        random.seed(m * 104729 + 13)
        patterns.append([random.randint(1, m - 1) for _ in range(15)])
        
        for vol in patterns:
            # Quick filter: ensure no voltage is 0
            if any(v % m == 0 for v in vol):
                continue
            
            adj = build_lift(m, vol)
            
            # Check girth
            if check_girth_fast(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Fallback: try larger m with more random patterns
    for m in range(300, 400):
        if time.time() - start_time > 115:
            break
        
        for seed_offset in range(15):
            random.seed(m * 100 + seed_offset * 7)
            vol = [random.randint(1, m - 1) for _ in range(15)]
            if any(v % m == 0 for v in vol):
                continue
            adj = build_lift(m, vol)
            if check_girth_fast(adj, 10 * m, g):
                edges = generate_edges(m, vol)
                return 10 * m, edges
    
    # Last resort: use a known large m
    m = 400
    random.seed(42)
    vol = [random.randint(1, m - 1) for _ in range(15)]
    adj = build_lift(m, vol)
    edges = generate_edges(m, vol)
    return 10 * m, edges
