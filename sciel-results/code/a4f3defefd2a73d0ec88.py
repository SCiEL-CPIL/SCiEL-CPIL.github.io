import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Helper to check validity using bitmasks for speed
    def is_valid_bitmask(n, adj_bitmasks):
        # adj_bitmasks[c][i] is an integer bitmask of neighbors of i in color c
        # Check C4: for each color c, for each pair (i,j), common neighbors <= 1
        # common = adj_bitmasks[c][i] & adj_bitmasks[c][j]
        # popcount(common) <= 1
        for c in range(5):
            adj = adj_bitmasks[c]
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    # Check if common has more than 1 bit set
                    # common & (common - 1) == 0 means <= 1 bit set
                    if common & (common - 1):
                        return False
        return True

    # Helper to build bitmasks from color list
    def build_bitmasks(n, colors):
        adj_bitmasks = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
                idx += 1
        return adj_bitmasks

    # Helper to get index of edge (i,j) in lexicographic order
    def get_index(i, j, n):
        # i < j
        # Number of edges in rows 0 to i-1
        # Row k has n-k-1 edges.
        # Sum_{k=0}^{i-1} (n-k-1) = i*(n-1) - (i-1)*i/2
        return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

    # Helper to count conflicts (number of C4s, roughly)
    # For speed, we can just check validity or count violations
    def count_conflicts_bitmask(n, adj_bitmasks):
        conflicts = 0
        for c in range(5):
            adj = adj_bitmasks[c]
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    # Count bits in common
                    # If > 1, it's a conflict
                    # popcount
                    cnt = 0
                    x = common
                    while x:
                        x &= x - 1
                        cnt += 1
                    if cnt > 1:
                        conflicts += cnt - 1
        return conflicts

    # Local search for fixed n
    def local_search(n, max_steps=100000):
        # Initialize randomly
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        adj_bitmasks = build_bitmasks(n, colors)
        
        # Precompute list of edges for random selection
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        current_conf = count_conflicts_bitmask(n, adj_bitmasks)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_conf == 0:
                return colors
            
            # Pick random edge
            e = random.randint(0, len(edges) - 1)
            i, j = edges[e]
            idx = get_index(i, j, n)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update bitmasks
            adj_bitmasks[old_c][i] &= ~(1 << j)
            adj_bitmasks[old_c][j] &= ~(1 << i)
            adj_bitmasks[new_c][i] |= (1 << j)
            adj_bitmasks[new_c][j] |= (1 << i)
            colors[idx] = new_c
            
            new_conf = count_conflicts_bitmask(n, adj_bitmasks)
            
            # Accept if better or random walk
            if new_conf < current_conf or (new_conf == current_conf and random.random() < 0.3):
                current_conf = new_conf
            else:
                # Revert
                adj_bitmasks[new_c][i] &= ~(1 << j)
                adj_bitmasks[new_c][j] &= ~(1 << i)
                adj_bitmasks[old_c][i] |= (1 << j)
                adj_bitmasks[old_c][j] |= (1 << i)
                colors[idx] = old_c
                
        return colors if is_valid_bitmask(n, build_bitmasks(n, colors)) else None

    best_n = 11
    best_colors = None
    
    # Start with a known valid small coloring? Or just random search for n=12, 13...
    # Since n=26 is the target, let's try to jump to larger n directly.
    # Random search for n=26 is hard. Let's try n=12, 13... up to 26.
    
    for target_n in range(12, 27):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple restarts
        for restart in range(200):
            if time.time() > deadline:
                break
            
            res = local_search(target_n)
            if res is not None:
                best_n = target_n
                best_colors = res
                found = True
                break
        
        if found:
            # Continue to next n
            continue
        else:
            # If we failed for this n, we might still try higher n?
            # Usually if we can't find n, we can't find n+1 easily from scratch.
            # But maybe a different random seed works?
            # Let's break if we fail for a few consecutive n's?
            # Or just keep trying until time runs out.
            # Given time constraints, if n=12 fails, n=13 is likely harder.
            # But let's give it a few tries.
            pass

    if best_colors is None:
        # Fallback to a simple valid coloring for small n (e.g., n=5)
        # Or just return n=1 with no edges?
        # The task says 1 <= n <= 29.
        # A trivial coloring for n=1 is valid (no edges).
        # But we want to maximize n.
        # If we found nothing, we should return a valid small one.
        # Let's generate a random valid one for n=5?
        # Or just use the seed from the original program for n=11?
        # Let's implement a quick greedy or random search for n=11 to be safe.
        res = local_search(11, max_steps=1000)
        if res:
            best_n = 11
            best_colors = res
        else:
            # Absolute fallback: n=1
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
