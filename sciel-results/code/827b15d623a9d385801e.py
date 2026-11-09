import random
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fast popcount
    if hasattr(int, 'bit_count'):
        def popcount(x):
            return x.bit_count()
    else:
        def popcount(x):
            count = 0
            while x:
                x &= x - 1
                count += 1
            return count

    def is_regular_subgraph(vertices, adj_bits):
        """
        vertices: tuple of vertex indices
        adj_bits: list of integers
        """
        # Create mask
        mask = 0
        for v in vertices:
            mask |= (1 << v)
            
        # Get degrees
        deg0 = popcount(adj_bits[vertices[0]] & mask)
        for v in vertices[1:]:
            deg = popcount(adj_bits[v] & mask)
            if deg != deg0:
                return False
        return True

    def validate_graph(adj_bits, n):
        """
        Returns True if NO induced subgraph of size >= 6 is regular.
        """
        # Check k from 6 to n
        # We can break early if we find one
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_regular_subgraph(comb, adj_bits):
                    return False
        return True

    target_n = 20
    
    # We will try to find a valid graph of size 20.
    # If we fail, we return the seed result.
    
    # To increase chances, we can try different p values or random regular graphs.
    # Let's stick to p=0.5 first.
    
    while time.time() < deadline:
        adj_bits = [0] * target_n
        edges = []
        
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
                    edges.append([i, j])
        
        if validate_graph(adj_bits, target_n):
            return {'n': target_n, 'edges': edges}
            
    # Fallback
    return {'n': 5, 'edges': []}
