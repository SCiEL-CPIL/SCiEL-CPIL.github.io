import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m_target = 401
    
    # Bitmask representation for adjacency
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    
    def compute_max_book(adj):
        """
        Computes the maximum book size.
        Also returns the edge achieving this max to guide search.
        Returns (max_book, edge_u, edge_v)
        """
        max_book = 0
        best_u, best_v = -1, -1
        for i in range(n):
            mask_i = adj[i]
            # Iterate j > i
            # We can optimize by only checking if potential book > max_book
            # But for n=40, full check is fast enough if we do it smartly.
            for j in range(i + 1, n):
                if (mask_i >> j) & 1:
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book:
                        max_book = common
                        best_u, best_v = i, j
        return max_book, best_u, best_v

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def edge_count(adj):
        count = 0
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            count += mask.bit_count()
        return count

    best_adj = None
    best_score = float('inf')
    
    # Initial construction:
    # Start with a random regular graph (approx 20-regular) to have a good base.
    # Then add 1 edge.
    # Instead of pure random, use a "greedy random" initialization:
    # Sort all potential edges by some heuristic? No, just random regular is good.
    
    for seed in range(200): # More seeds, faster per seed
        if time.time() > deadline - 5:
            break
            
        random.seed(seed + 500)
        
        # 1. Generate a random 20-regular graph (400 edges)
        # Method: Randomly pair up stubs.
        # Each vertex has 20 stubs. Total 800 stubs.
        # This is complex to implement quickly without errors.
        # Alternative: Start with a circulant graph C_40(S) with |S|=10.
        # This is deterministic and fast.
        # Then randomize it via swaps to break symmetry.
        
        shifts = random.sample(range(1, 20), 10)
        adj = [0] * n
        for s in shifts:
            for i in range(n):
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        
        # Add 1 random edge to reach 401
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not ((adj[u] >> v) & 1):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                break
        
        current_score, cu, cv = compute_max_book(adj)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
        
        # 2. Targeted Local Search
        # Strategy:
        # While time allows:
        #   a. Find max book edge (u, v) and its size B.
        #   b. If B is small (e.g., <= 5), stop or randomize.
        #   c. Identify neighbors w such that (u,v,w) is a triangle.
        #   d. Pick a random w from these neighbors.
        #   e. Try to remove edge (u,w) or (v,w) and add a new random edge (x,y)
        #      that does not create a larger book.
        #   f. Accept if max_book decreases or stays same (with some probability for exploration).
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            score, u, v = compute_max_book(adj)
            if score <= 5:
                break # Theoretical lower bound is 6 (40/6), so 5 is impossible. 
                       # Wait, 40/6 = 6.66, so min integer is 7.
                       # If score <= 6, we are at the theoretical limit (or close).
                       # Actually, if score is 7, we can't do better than 7 if the theorem holds tightly?
                       # Theorem says >= 40/6 = 6.66 -> 7. So 7 is the target.
                       # If we hit 7, we should try to keep it but maybe explore for stability?
                       # No, 7 is the global min if achievable.
                       # Let's break if score == 7.
            if score == 7:
                if score < best_score:
                    best_score = score
                    best_adj = adj.copy()
                break
            
            # Get neighbors of u and v that form triangles with (u,v)
            # common neighbors = adj[u] & adj[v]
            common_mask = adj[u] & adj[v]
            # Exclude u and v themselves (though they shouldn't be in their own adj in simple graph)
            # common_mask has bits for w where (u,w) and (v,w) exist.
            
            if common_mask == 0:
                # Should not happen if score > 0
                break
            
            # Pick a random w from common neighbors
            # Convert mask to list of indices
            ws = []
            mask = common_mask
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                ws.append(w)
                mask ^= lsb
            
            if not ws:
                break
                
            w = random.choice(ws)
            
            # We want to remove one of the edges (u,w) or (v,w) to break the triangle (u,v,w).
            # Let's randomly choose which one to remove.
            if random.random() < 0.5:
                # Remove (u, w)
                rem_u, rem_v = u, w
            else:
                # Remove (v, w)
                rem_u, rem_v = v, w
            
            # Ensure we don't remove the edge (u,v) itself (we didn't, w is distinct)
            # Perform removal
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            
            # Now we have 400 edges. We need to add one back.
            # We want to add an edge (x,y) that minimizes the increase in max_book.
            # Strategy: Try a few random non-edges and pick the one with best resulting score.
            # Or just pick one random non-edge and check.
            
            best_new_adj = None
            best_new_score = float('inf')
            
            # Try 5 random non-edges
            for _ in range(5):
                cand_adj = adj.copy() # Copy is expensive? n=40, 40 ints. Should be ok.
                # Find a random non-edge in cand_adj
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x != y and not ((cand_adj[x] >> y) & 1):
                        break
                
                cand_adj[x] |= (1 << y)
                cand_adj[y] |= (1 << x)
                
                new_score, _, _ = compute_max_book(cand_adj)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_adj = cand_adj
            
            # Accept if it improves or is equal (with high prob) or slightly worse (with low prob)
            if best_new_adj is not None:
                if best_new_score < score:
                    adj = best_new_adj
                    current_score = best_new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                elif best_new_score == score:
                    # Accept with some probability to move around the plateau
                    if random.random() < 0.5:
                        adj = best_new_adj
                        current_score = best_new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = adj.copy()
                else:
                    # Worse, reject (or accept with very low prob for SA-like behavior)
                    # For now, strict improvement or plateau move only.
                    pass
        
        # After local search, check best
        final_score, _, _ = compute_max_book(adj)
        if final_score < best_score:
            best_score = final_score
            best_adj = adj.copy()
            
        if best_score == 7:
            break # Found optimal (or likely optimal)
            
    if best_adj is None:
        # Fallback
        adj = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    return get_edge_list(best_adj)
