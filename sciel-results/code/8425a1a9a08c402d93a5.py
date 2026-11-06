import random
import math
import time

def construct():
    n = 16
    target_edges = 65
    
    # Bitset representation
    # adj[i] is an integer with bits set for neighbors of i
    
    def calc_score(adj):
        # Calculate degrees
        deg = [0] * n
        for i in range(n):
            deg[i] = bin(adj[i]).count('1')
        
        max_sum = 0
        for i in range(n):
            # Iterate neighbors j of i where i < j
            # We can iterate set bits
            j = i + 1
            while j < n:
                if (adj[i] >> j) & 1:
                    # Check common neighbors k > j
                    common = adj[i] & adj[j]
                    # Mask out bits <= j
                    common &= ~((1 << (j + 1)) - 1)
                    # Iterate set bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                        c ^= lsb
                j += 1
        return max_sum

    # Initialize random graph
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    current_edges = set(random.sample(all_edges, target_edges))
    
    # Convert to bitsets
    current_adj = [0] * n
    for u, v in current_edges:
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
    current_score = calc_score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    min_T = 0.001
    start_time = time.time()
    
    while T > min_T and (time.time() - start_time) < 115:
        # Generate move: remove edge (u, v), add edge (x, y)
        # Pick random edge from current
        # We need a list of edges for random selection
        # Maintaining a list of edges is expensive to update.
        # Instead, pick random u, v, check if edge exists.
        # But we need to remove an existing edge.
        # Let's pick a random pair (u, v) until it's an edge.
        
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u < v and ((current_adj[u] >> v) & 1):
                break
        
        # Remove (u, v)
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        # Pick random non-edge (x, y)
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x < y and not ((current_adj[x] >> y) & 1):
                break
        
        # Add (x, y)
        current_adj[x] |= (1 << y)
        current_adj[y] |= (1 << x)
        
        new_score = calc_score(current_adj)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() > prob:
                    # Reject move: revert
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_adj[x] &= ~(1 << y)
                    current_adj[y] &= ~(1 << x)
                    # current_score remains old
                else:
                    current_score = new_score
            else:
                # delta == 0, accept
                current_score = new_score
                
        T *= cooling
        
    # Convert best_adj back to edge list
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                best_edges.append([i, j])
                
    return {'n': n, 'edges': best_edges}
