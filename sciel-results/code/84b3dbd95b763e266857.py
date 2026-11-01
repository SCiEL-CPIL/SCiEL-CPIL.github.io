import itertools
import random
import time
from collections import deque

def construct(k, g):
    # Petersen Graph Definition
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        tm = [0]
        for i in range(1, n):
            tm.append(bin(i).count('1') % 2)
        return tm
    
    tm_sequence = get_thue_morse(200)
    
    start_time = time.time()
    time_limit = 110
    
    # Generate voltage patterns from Thue-Morse with various transformations
    def generate_voltage_patterns(m):
        """Generate multiple voltage patterns for a given m."""
        patterns = []
        
        # Pattern 1: Direct Thue-Morse values scaled
        patterns.append([tm_sequence[i] * (m // 2) % m for i in range(15)])
        
        # Pattern 2: Thue-Morse with linear offset
        patterns.append([(tm_sequence[i] * (i + 1) + i) % m for i in range(15)])
        
        # Pattern 3: Thue-Morse with quadratic term
        patterns.append([(tm_sequence[i] * i * i + i) % m for i in range(15)])
        
        # Pattern 4: Thue-Morse bits shifted by different amounts
        for shift in range(min(5, m)):
            patterns.append([((tm_sequence[i + shift] * (i + 1)) % m) for i in range(15)])
        
        # Pattern 5: Use first 15 values of Thue-Morse as binary to form numbers
        # Split into groups and use modulo
        for base in range(max(1, m // 4)):
            patterns.append([(tm_sequence[i] * base + (i * base) // 3) % m for i in range(15)])
        
        # Pattern 6: Alternating Thue-Morse with inverse
        patterns.append([(tm_sequence[i] * (m - i - 1)) % m for i in range(15)])
        
        # Pattern 7: Thue-Morse with prime-like weighting
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        patterns.append([(tm_sequence[i] * primes[i]) % m for i in range(15)])
        
        # Pattern 8: Randomized Thue-Morse (deterministic seed)
        random.seed(m * 7919)  # Prime multiplier for better distribution
        patterns.append([random.choice([0, m//3, 2*m//3]) for _ in range(15)])
        
        # Pattern 9: Thue-Morse with Fibonacci weights
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        patterns.append([(tm_sequence[i] * fib[i]) % m for i in range(15)])
        
        # Pattern 10: Thue-Morse with exponential decay
        patterns.append([(tm_sequence[i] * (m - 1) // (i + 1)) % m for i in range(15)])
        
        return patterns
    
    def build_lift(voltages, m):
        """Build the lift graph adjacency list."""
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
    
    def check_girth_bfs(adj, n_vertices, max_cycle_len):
        """Check if graph has any cycle shorter than max_cycle_len using BFS."""
        for start in range(n_vertices):
            # BFS to find shortest cycle through start
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                if dist[node] >= max_cycle_len - 1:
                    continue
                
                for neighbor in adj[node]:
                    if neighbor == parent[node]:
                        continue
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    else:
                        # Found a cycle
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < max_cycle_len:
                            return False
        
        return True
    
    def verify_girth_exact(adj, n_vertices, g):
        """Verify girth is at least g using non-backtracking walk enumeration."""
        # For small graphs, we can enumerate non-backtracking walks up to length g-1
        # If no walk of length g-1 returns to start, girth >= g
        
        # Actually, let's use a more efficient approach:
        # Count non-backtracking walks of length L from each vertex
        # If for all L < g, no vertex has a non-backtracking walk of length L back to itself,
        # then girth >= g
        
        # This is complex. Let's use BFS-based cycle detection which is sufficient.
        return check_girth_bfs(adj, n_vertices, g)
    
    # Try m values starting from smaller ones
    for m in range(50, 200):
        if time.time() - start_time > time_limit:
            break
        
        patterns = generate_voltage_patterns(m)
        
        for vol in patterns:
            if time.time() - start_time > time_limit:
                break
            
            adj = build_lift(vol, m)
            
            # Quick check: verify 3-regularity
            if any(len(adj[v]) != 3 for v in range(10 * m)):
                continue
            
            # Check girth
            if verify_girth_exact(adj, 10 * m, g):
                # Construct edge list
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
    
    # Fallback: L-U-W from seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
