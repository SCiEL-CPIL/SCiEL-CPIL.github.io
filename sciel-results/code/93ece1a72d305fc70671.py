import time

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10

    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        # Efficient generation using recurrence or bit counting
        # bin(i).count('1') % 2 is simple and fast enough for small n
        return [bin(i).count('1') % 2 for i in range(n)]

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
                    elif neighbor != parent[node]:
                        cycle_len = dist[node] + dist[neighbor] + 1
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

    start_time = time.time()
    
    # Calculate lower bound for m
    if g % 2 == 0:
        r = g // 2
        moore_bound = k * (2 ** (r - 1) - 1) + 1
    else:
        r = (g - 1) // 2
        moore_bound = int(k / 2 * (2 ** r - 1)) + 1
        
    m_min = max(2, (moore_bound + n_base - 1) // n_base)

    # Precompute a long Thue-Morse sequence
    # We need enough bits to generate 15 voltages for each m we try.
    # Max m we might try: let's say up to 200.
    # We need 15 * log2(200) ~ 15 * 8 = 120 bits per candidate.
    # We will try multiple shifts. Let's generate 1000 bits.
    tm_bits = get_thue_morse(2000)

    # Search for smallest m
    # We try increasing m
    max_m_search = m_min + 100 # Search window
    
    for m in range(m_min, max_m_search):
        if time.time() - start_time > 110:
            break
            
        # Determine bits needed per voltage to cover range [0, m-1]
        L = (m - 1).bit_length()
        
        # Try several deterministic shifts/phases of the Thue-Morse sequence
        # We use 10 different shifts to increase chance of finding a valid graph
        num_shifts = 10
        
        for shift in range(num_shifts):
            voltages = []
            valid = True
            for i in range(15):
                # Take L bits starting at index (i * L + shift)
                # If we run out of bits, wrap around or extend
                start_idx = i * L + shift
                val = 0
                for j in range(L):
                    bit = tm_bits[(start_idx + j) % len(tm_bits)]
                    val = (val << 1) | bit
                voltages.append(val % m)
            
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
    # Fallback: return a large graph if not found
    # Use a random seed fallback if deterministic fails
    import random
    m = max_m_search
    n_vertices = n_base * m
    random.seed(42)
    voltages = [random.randint(0, m - 1) for _ in range(15)]
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
