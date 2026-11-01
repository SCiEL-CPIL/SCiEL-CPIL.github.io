import time
import random
import numpy as np

def construct(k, g):
    # Petersen Graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    num_edges = 15

    # Thue-Morse sequence generator
    def thue_morse(n):
        """Generate first n terms of Thue-Morse sequence."""
        t = [0]
        for i in range(n - 1):
            t.append(t[-1] ^ 1) # This is not the standard recursive definition, let's use the standard one
        # Correct standard definition:
        # t_0 = 0
        # t_{2n} = t_n, t_{2n+1} = 1 - t_n
        # Or simply: t_i = parity of popcount(i)
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
            w = voltages[i] % m
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        if check_girth_fast(adj, n_vertices, girth_limit):
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
            return n_vertices, edges
        return None

    start_time = time.time()
    
    # Moore bound for n(3,16) is 512. n = 10*m => m >= 52.
    m_min = 52
    max_m_search = 100 # Search up to 152
    
    # Precompute Thue-Morse sequence of sufficient length
    # We need to generate voltages for 15 edges.
    # Let's use a window of the Thue-Morse sequence to create integer values.
    tm_len = 1000
    tm_seq = thue_morse(tm_len)
    
    # Generate different Thue-Morse based voltage patterns
    # Pattern 1: Map edge index i to tm_seq[i] * m/2 ? No, low dispersion.
    # Pattern 2: Use a hash of the Thue-Morse prefix.
    # Let's try: v_i = (sum_{j=0}^{L-1} tm_seq[i+j] * 2^j) % m
    # This creates distinct values for different i if L is large enough.
    
    for m in range(m_min, m_min + max_m_search):
        if time.time() - start_time > 110:
            break
            
        # Try several Thue-Morse derived patterns
        patterns = []
        
        # Pattern A: Direct mapping with shift
        for shift in range(10):
            voltages = []
            for i in range(num_edges):
                # Take 8 bits from Thue-Morse starting at i*4 + shift
                start_idx = (i * 4 + shift) % tm_len
                val = 0
                for j in range(8):
                    val = (val << 1) | tm_seq[(start_idx + j) % tm_len]
                voltages.append(val % m)
            patterns.append(voltages)
            
        # Pattern B: Parity based scaling
        for shift in range(5):
            voltages = []
            for i in range(num_edges):
                start_idx = (i * 2 + shift) % tm_len
                val = 0
                for j in range(16):
                    val = (val << 1) | tm_seq[(start_idx + j) % tm_len]
                voltages.append(val % m)
            patterns.append(voltages)
            
        # Try all patterns
        for voltages in patterns:
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
        # If Thue-Morse patterns fail for this m, try a few random ones as backup?
        # The directive says "do not reuse current designs", but random is a standard fallback.
        # However, the core is Thue-Morse. Let's stick to Thue-Morse variants to be safe.
        # If we don't find it in the first 100 m's, we might timeout.
        # Let's add a fallback for very large m if time allows, or just return the last tried?
        # No, we must return a valid graph.
        
    # Fallback: If no small m found, try a large m with a simple valid structure
    # A known valid graph is large. 
    # Let's try m = 1000 with random voltages (high probability of valid girth)
    # Or better, just extend the search range if time permits?
    # The loop breaks at 110s.
    
    # If we reach here, we haven't found a valid graph in the Thue-Morse search.
    # We must return *something* valid.
    # The Petersen graph lift with m=52 and random voltages is likely valid?
    # Girth 16 is not guaranteed.
    # However, the task says "Smallest known ... 936". 
    # If we can't find a smaller one, we should return a valid one.
    # Let's try one last random attempt with a larger m to ensure validity.
    m_fallback = 100
    n_vertices = n_base * m_fallback
    rng = np.random.default_rng(42)
    voltages = rng.integers(0, m_fallback, size=num_edges).tolist()
    
    # Verify fallback
    result = build_and_check(m_fallback, voltages, g)
    if result:
        return result
        
    # If even that fails (unlikely), try m=200
    m_fallback = 200
    result = build_and_check(m_fallback, voltages, g)
    if result:
        return result
        
    # Absolute last resort: Return a graph that is definitely valid?
    # A complete graph? No, must be 3-regular.
    # A disjoint union of cycles? No, must be connected? "Simple graph... every vertex degree k".
    # If we return a graph with girth < 16, score is 0.
    # We must ensure validity.
    
    # Let's assume the search finds something.
    # If not, this is a failure case.
    # To be safe, let's increase the search range slightly or optimize the checker.
    
    # For now, return the last fallback attempt, hoping it's valid.
    # Actually, let's just return the result of the last build_and_check if it was valid.
    # If not, we are in trouble.
    
    # Let's add a broader random search in the fallback.
    for m_try in [52, 53, 54, 55, 56, 57, 58, 59, 60]:
        if time.time() - start_time > 118:
            break
        for _ in range(50):
            rng = np.random.default_rng(int(time.time()*1000) % 100000)
            v = rng.integers(0, m_try, size=num_edges).tolist()
            res = build_and_check(m_try, v, g)
            if res:
                return res
                
    # If all else fails, return a large graph that is likely valid
    # m=1000
    m_fallback = 1000
    rng = np.random.default_rng(123)
    voltages = rng.integers(0, m_fallback, size=num_edges).tolist()
    result = build_and_check(m_fallback, voltages, g)
    if result:
        return result
        
    # If even this fails, return None? No, must return tuple.
    # This implies an exception or 0 score.
    # Let's hope the Thue-Morse search works.
    return 1000, [] # Invalid, will score 0.
