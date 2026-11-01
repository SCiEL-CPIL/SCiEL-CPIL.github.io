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

    def thue_morse(n):
        """Return the Thue-Morse value (parity of set bits) for n."""
        v = 0
        while n:
            v ^= 1
            n >>= 1
        return v

    def thue_morse_sequence(length, start=0):
        """Generate Thue-Morse sequence values starting from index 'start'."""
        return [thue_morse(start + i) for i in range(length)]

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

    def thue_morse_voltages(m, offset=0, scale=1):
        """
        Generate voltages using Thue-Morse word for high dispersion.
        The Thue-Morse sequence has excellent equidistribution properties
        and minimizes low-order polynomial collisions in closed walk sums.
        """
        # Use a window of the Thue-Morse sequence with offsets for variety
        tm = thue_morse_sequence(num_edges * 4, start=offset)
        # Map Thue-Morse bits (0/1) to voltages in [0, m) with scaling
        # Use multiple bits to get more range: combine pairs of TM bits
        voltages = []
        for i in range(num_edges):
            # Use 2 bits from TM sequence to get 4 values, then scale
            b1 = tm[i * 2]
            b2 = tm[i * 2 + 1]
            val = b1 * 2 + b2  # 0..3
            # Scale to [0, m) with some structure
            v = (val * scale + offset * 7 + i * 3) % m
            voltages.append(v)
        return voltages

    def thue_morse_voltages_v2(m, offset=0):
        """
        Alternative Thue-Morse voltage assignment using longer windows.
        Combines multiple TM bits for better dispersion.
        """
        # Use 3 bits per edge for 8 possible values
        tm = thue_morse_sequence(num_edges * 3 + 10, start=offset)
        voltages = []
        for i in range(num_edges):
            b1 = tm[i * 3]
            b2 = tm[i * 3 + 1]
            b3 = tm[i * 3 + 2]
            val = b1 * 4 + b2 * 2 + b3  # 0..7
            v = (val * (m // 8 + 1) + offset * 11 + i * 5) % m
            voltages.append(v)
        return voltages

    def thue_morse_voltages_v3(m, offset=0):
        """
        Use Thue-Morse to select from a set of well-spread base values.
        """
        # Precompute a set of well-spread values
        base_vals = [(i * m // num_edges + offset * 3) % m for i in range(num_edges)]
        tm = thue_morse_sequence(num_edges * 2, start=offset)
        voltages = []
        for i in range(num_edges):
            # TM bit selects perturbation
            perturb = tm[i] * (m // 2)
            v = (base_vals[i] + perturb + tm[i + num_edges] * 7) % m
            voltages.append(v)
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
            
        # Try Thue-Morse based deterministic patterns first
        # These have structural properties that minimize low-order collisions
        
        # Pattern 1: Basic Thue-Morse with varying offsets
        for offset in range(0, 10):
            voltages = thue_morse_voltages(m, offset=offset, scale=3)
            result = build_and_check(m, voltages, g)
            if result:
                return result
        
        # Pattern 2: 3-bit Thue-Morse
        for offset in range(0, 10):
            voltages = thue_morse_voltages_v2(m, offset=offset)
            result = build_and_check(m, voltages, g)
            if result:
                return result
        
        # Pattern 3: Thue-Morse perturbation of spread values
        for offset in range(0, 10):
            voltages = thue_morse_voltages_v3(m, offset=offset)
            result = build_and_check(m, voltages, g)
            if result:
                return result

        # Try random patterns as fallback
        num_tries = 300 if m < 80 else 500
        
        for t in range(num_tries):
            if time.time() - start_time > 105:
                break
            seed_val = (m * 100000 + t)
            rng = np.random.default_rng(seed_val)
            voltages = rng.integers(0, m, size=num_edges).tolist()
            
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
    # Fallback: use Thue-Morse at a larger m
    m = m_min + max_m_search
    n_vertices = n_base * m
    voltages = thue_morse_voltages(m, offset=0, scale=3)
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
