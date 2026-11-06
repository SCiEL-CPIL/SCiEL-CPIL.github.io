import random
import numpy as np

def construct():
    n = 40
    m = 401
    
    # 1. Greedy Construction
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    # common[u][v] stores the number of common neighbors
    common = np.zeros((n, n), dtype=int)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Greedy: add edge with min common neighbors
    for _ in range(m):
        best_edge = None
        min_common = 100
        # To add some randomness to break ties and avoid deterministic traps
        candidates = []
        for u, v in all_possible:
            if not adj[u, v]:
                c = common[u, v]
                if c < min_common:
                    min_common = c
                    candidates = [(u, v)]
                elif c == min_common:
                    candidates.append((u, v))
        
        if not candidates:
            break
            
        # Pick randomly from candidates with min common
        u, v = random.choice(candidates)
        
        # Add edge
        adj[u, v] = True
        adj[v, u] = True
        edges.add((u, v))
        
        # Update common neighbors for other edges
        # For any w connected to u, common[w][v] increases by 1
        # For any w connected to v, common[u][w] increases by 1
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u, w]:
                common[w, v] += 1
                common[v, w] += 1
            if adj[v, w]:
                common[u, w] += 1
                common[w, u] += 1

    # 2. Local Search with Frequency Tracking
    # Initialize frequency array
    # Max possible book size is n-2 = 38
    freq = np.zeros(n, dtype=int)
    for u, v in edges:
        b = common[u, v]
        freq[b] += 1
    
    def get_max_book(freq):
        for b in range(n-1, -1, -1):
            if freq[b] > 0:
                return b
        return 0
        
    current_max = get_max_book(freq)
    
    # Local Search Loop
    # We want to reduce current_max.
    # If we can't reduce max, we try to reduce freq[max] to make it easier to drop max later.
    
    num_iterations = 50000
    # Use a random seed for reproducibility
    random.seed(42)
    
    for _ in range(num_iterations):
        # Find an edge with the current max book size
        # To be efficient, we can just pick a random edge and check if it's max?
        # Or maintain a list of max edges?
        # Given n=40, scanning 401 edges is fast.
        
        max_edges = []
        for u, v in edges:
            if common[u, v] == current_max:
                max_edges.append((u, v))
                
        if not max_edges:
            # This shouldn't happen if current_max is correct, but just in case
            current_max = get_max_book(freq)
            continue

        # Pick a random max edge to remove
        u, v = random.choice(max_edges)
        
        # Try to find a non-edge to add that reduces the max
        # We need to pick a non-edge (x, y)
        # We want B_new (common[x][y]) to be < current_max
        # Ideally, we want B_new to be as small as possible.
        
        # Instead of random, let's pick a random non-edge.
        # If the graph is dense, rejection sampling is okay.
        # 401 edges, 780 total. ~379 non-edges.
        
        found = False
        x, y = -1, -1
        # Try a few random non-edges
        for _ in range(10):
            rx, ry = random.sample(range(n), 2)
            if rx > ry: rx, ry = ry, rx
            if not adj[rx, ry]:
                x, y = rx, ry
                found = True
                break
                
        if not found:
            continue
            
        b_old = common[u, v] # This is current_max
        b_new = common[x, y]
        
        # Simulate swap
        # Remove (u,v): freq[b_old] -= 1
        # Add (x,y): freq[b_new] += 1
        
        # Check if max decreases
        # New max is max( max_b_in_freq_after_swap, b_new )
        # Note: b_old is current_max.
        # If freq[b_old] > 1, then b_old is still present, so new max is at least b_old.
        # If freq[b_old] == 1, then b_old is gone. New max is max(second_max, b_new).
        
        if freq[b_old] > 1:
            # Max stays b_old unless b_new > b_old (impossible since b_old is max)
            # So max doesn't decrease.
            # But we might reduce freq[b_old] by 1.
            # This is a "sideways" move. Accept with low probability?
            # For now, only accept if max decreases.
            # To escape local minima, we might need to accept sideways moves.
            # Let's try accepting if b_new < b_old.
            # Actually, if freq[b_old] > 1, swapping one max edge for a lower one
            # reduces the count of max edges. This is good.
            # Let's accept if b_new < b_old.
            if b_new < b_old:
                # Accept
                adj[u, v] = False
                adj[v, u] = False
                edges.discard((u, v))
                
                adj[x, y] = True
                adj[y, x] = True
                edges.add((x, y))
                
                # Update common counts
                # Remove (u,v) effects
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u, w]:
                        common[w, v] -= 1
                        common[v, w] -= 1
                    if adj[v, w]:
                        common[u, w] -= 1
                        common[w, u] -= 1
                        
                # Add (x,y) effects
                for w in range(n):
                    if w == x or w == y:
                        continue
                    if adj[x, w]:
                        common[w, y] += 1
                        common[y, w] += 1
                    if adj[y, w]:
                        common[x, w] += 1
                        common[w, x] += 1
                        
                # Update freq
                freq[b_old] -= 1
                freq[b_new] += 1
                # current_max might decrease if freq[b_old] becomes 0?
                # No, we assumed freq[b_old] > 1, so it's still > 0.
                # So current_max remains the same.
            else:
                # Reject
                pass
        else:
            # freq[b_old] == 1
            # If we swap, b_old is removed.
            # New max is max( max_b < b_old with freq > 0, b_new )
            
            # Find second max
            second_max = 0
            for b in range(b_old - 1, -1, -1):
                if freq[b] > 0:
                    second_max = b
                    break
                    
            new_max = max(second_max, b_new)
            
            if new_max < current_max:
                # Accept
                adj[u, v] = False
                adj[v, u] = False
                edges.discard((u, v))
                
                adj[x, y] = True
                adj[y, x] = True
                edges.add((x, y))
                
                # Update common counts
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u, w]:
                            common[w, v] -= 1
                            common[v, w] -= 1
                    if adj[v, w]:
                        common[u, w] -= 1
                        common[w, u] -= 1
                        
                for w in range(n):
                    if w == x or w == y:
                        continue
                    if adj[x, w]:
                        common[w, y] += 1
                        common[y, w] += 1
                    if adj[y, w]:
                        common[x, w] += 1
                        common[w, x] += 1
                        
                freq[b_old] -= 1
                freq[b_new] += 1
                current_max = new_max
            else:
                # Reject
                pass
                
    result = [list(e) for e in edges]
    return result
