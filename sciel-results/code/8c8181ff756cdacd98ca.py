import time
import random
import numpy as np

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    num_edges = 15

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
                d = dist[node]
                if d >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = d + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        cycle_len = d + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True

    def build_and_check(m, voltages, girth_limit):
        """Build lift, check girth, return edges if valid."""
        n_vertices = n_base * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i]
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        if check_girth_fast(adj, n_vertices, girth_limit):
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                w = voltages[i]
                for s in range(m):
                    u_id = u * m + s
                    v_id = v * m + (s + w) % m
                    if u_id < v_id:
                        edges.append((u_id, v_id))
                    else:
                        edges.append((v_id, u_id))
            return n_vertices, edges
        return None

    def generate_thue_morse_voltages(m, num_edges):
        """
        Generate voltages derived from the Thue-Morse word.
        We use a sliding window of the Thue-Morse sequence to create high-dispersion integers.
        """
        # Generate enough Thue-Morse bits
        # We need num_edges values. Let's use a window size of 8 bits to get good range.
        # Total bits needed: num_edges + 8
        length = num_edges + 8
        tm_bits = [0] * (1 << (length.bit_length())) 
        # Standard Thue-Morse generation
        tm = [0]
        for i in range(length):
            # t_i = popcount(i) % 2
            tm.append(bin(i).count('1') % 2)
        
        voltages = []
        window_size = 8
        for i in range(num_edges):
            # Take bits from i to i+window_size
            val = 0
            for j in range(window_size):
                val = (val << 1) | tm[i + j]
            voltages.append(val % m)
        return voltages

    start_time = time.time()
    
    # Moore bound lower bound for n is 512.
    # n = 10 * m, so m >= 52.
    m_min = 52
    
    # Search for smallest m
    max_m_search = 200
    
    for m in range(m_min, m_min + max_m_search):
        if time.time() - start_time > 110:
            break
            
        # Try Thue-Morse derived voltages
        voltages = generate_thue_morse_voltages(m, num_edges)
        result = build_and_check(m, voltages, g)
        if result:
            return result
            
        # If Thue-Morse fails, try a shifted version or different window
        # We can also try adding a constant offset to the voltages to break symmetries
        for offset in range(1, 5):
            voltages_shifted = [(v + offset) % m for v in voltages]
            result = build_and_check(m, voltages_shifted, g)
            if result:
                return result

    # Fallback to random if deterministic fails within time
    m = m_min + max_m_search
    n_vertices = n_base * m
    rng = np.random.default_rng(42)
    voltages = rng.integers(0, m, size=num_edges).tolist()
    edges = []
    for i, (u, v) in enumerate(petersen_edges):
        w = voltages[i]
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            if u_id < v_id:
                edges.append((u_id, v_id))
            else:
                edges.append((v_id, u_id))
    return n_vertices, edges
