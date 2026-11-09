import time
import random

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Bitmask helpers
    # adj[i] is an integer with bits set for neighbors of i
    
    def build_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_degrees(adj):
        degs = [0] * n
        for i in range(n):
            degs[i] = bin(adj[i]).count('1')
        return degs

    def calc_score(adj, degs):
        max_sum = 0
        # Iterate over all pairs u < v
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Find common neighbors w
                    # w must be connected to u and v
                    common = adj[u] & adj[v]
                    # We can iterate through set bits of common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        c ^= lsb
        return max_sum

    # Initial Graph: K_{8,8} + edge (0,1)
    # Part 0: 0-7, Part 1: 8-15
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append((u, v))
    edges.append((0, 1))
    
    # Verify count
    assert len(edges) == 65, f"Edge count {len(edges)} != 65"
    
    adj = build_adj_from_edges(edges)
    degs = get_degrees(adj)
    current_score = calc_score(adj, degs)
    
    best_score = current_score
    best_adj = adj[:]
    best_edges = edges[:]

    def apply_swap(adj, degs, u, v, x, y):
        """
        Remove edge (u,v), add edge (x,y).
        Returns new adj, degs, score.
        Assumes (u,v) is edge and (x,y) is non-edge.
        """
        # Copy adj
        new_adj = adj[:]
        
        # Remove (u,v)
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        degs[u] -= 1
        degs[v] -= 1
        
        # Add (x,y)
        new_adj[x] |= (1 << y)
        new_adj[y] |= (1 << x)
        degs[x] += 1
        degs[y] += 1
        
        return new_adj, degs, calc_score(new_adj, degs)

    def generate_moves(adj, degs):
        """
        Generate list of (u, v, x, y, new_score) for all valid swaps.
        """
        moves = []
        # List of current edges
        current_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    current_edges.append((u, v))
        
        # List of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
                    
        # This is O(E * NE * ScoreCalc). 
        # E=65, NE=55. 3575 moves.
        # ScoreCalc is fast with bitmasks.
        
        for u, v in current_edges:
            for x, y in non_edges:
                # Skip if swapping creates a multi-edge or self-loop? 
                # (u,v) and (x,y) are distinct simple edges.
                # If (u,v) == (x,y), we wouldn't swap, but they are from disjoint sets.
                # However, if x==u and y==v, it's the same edge, but (u,v) is in edges, (x,y) in non-edges, so impossible.
                # What if the swap results in the same graph? No, we remove one and add another.
                
                new_adj, new_degs, new_score = apply_swap(adj, degs, u, v, x, y)
                moves.append((new_score, u, v, x, y, new_adj, new_degs))
                
        return moves

    # Local Search Loop
    iterations = 0
    while time.time() < deadline:
        # Find best move
        moves = generate_moves(adj, degs)
        
        if not moves:
            break
            
        # Sort by score
        moves.sort(key=lambda x: x[0])
        
        best_move = moves[0]
        new_score, u, v, x, y, new_adj, new_degs = best_move
        
        if new_score < current_score:
            # Accept
            current_score = new_score
            adj = new_adj
            degs = new_degs
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                # Reconstruct edges from best_adj for final return
                temp_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if best_adj[i] & (1 << j):
                            temp_edges.append([i, j])
                best_edges = temp_edges
                
            iterations += 1
        else:
            # Local minimum reached. Perturb.
            # Perform 3 random swaps to escape
            for _ in range(3):
                # Pick random edge to remove
                rand_u = random.randint(0, n-1)
                rand_v = random.randint(0, n-1)
                if rand_u == rand_v:
                    continue
                if not (adj[rand_u] & (1 << rand_v)):
                    continue
                
                # Pick random non-edge to add
                rand_x = random.randint(0, n-1)
                rand_y = random.randint(0, n-1)
                if rand_x == rand_y:
                    continue
                if adj[rand_x] & (1 << rand_y):
                    continue
                
                # Apply swap
                adj, degs, current_score = apply_swap(adj, degs, rand_u, rand_v, rand_x, rand_y)
                
            # If we hit the deadline during perturbation, break
            if time.time() > deadline:
                break

    # Final check: ensure best_edges corresponds to best_adj
    # Reconstruct edges from best_adj just in case
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                final_edges.append([i, j])
                
    # Verify constraints
    if len(final_edges) != 65:
        # Fallback to original if something went wrong
        return {'n': n, 'edges': edges}

    return {'n': n, 'edges': final_edges}
