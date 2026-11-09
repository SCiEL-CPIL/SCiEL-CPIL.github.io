import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s wall-clock, keep 5s safety margin
    
    # The task: find smallest n such that there exists a tournament on n vertices
    # where every 4-vertex subset has a common dominator (a vertex with arcs to all 4).
    # 
    # Known: Paley tournament on p=67 (where 67 ≡ 3 mod 4) is valid, giving f(4) ≤ 67.
    # Erdős lower bound: f(4) ≥ 31.
    #
    # Strategy:
    # 1. Try to construct a valid tournament on n < 67 via local search.
    # 2. If we find one, return it (smaller n = better score).
    # 3. If not, fall back to Paley 67 (known valid).
    # 4. If even that fails somehow, return a larger valid construction.
    
    def paley_tournament(p):
        """Build Paley tournament on p vertices (p ≡ 3 mod 4 prime)."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # i -> j iff (j - i) mod p is a quadratic residue
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation
    
    def count_bad_4sets(orientation, n):
        """Count the number of 4-vertex subsets without a common dominator.
        
        A vertex v dominates a set S if v has arcs to all vertices in S.
        For a 4-set {a,b,c,d}, it's bad if no vertex dominates all 4.
        
        We use bitmask approach for speed.
        """
        # Build out-neighborhood bitmasks
        # out_mask[v] has bit u set if v -> u
        out_mask = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    out_mask[i] |= (1 << j)
                else:
                    # j -> i
                    out_mask[j] |= (1 << i)
        
        # For each 4-set {a,b,c,d}, check if there exists v such that
        # v -> a, v -> b, v -> c, v -> d
        # i.e., out_mask[v] has all of a,b,c,d set.
        
        bad_count = 0
        # Iterate over 4-sets
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if any vertex dominates all 4
                        # A vertex v dominates {a,b,c,d} iff 
                        # (out_mask[v] >> a) & 1 and (out_mask[v] >> b) & 1 and 
                        # (out_mask[v] >> c) & 1 and (out_mask[v] >> d) & 1
                        # Equivalently, out_mask[v] & ((1<<a)|(1<<b)|(1<<c)|(1<<d)) == (1<<a)|(1<<b)|(1<<c)|(1<<d)
                        mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        found = False
                        for v in range(n):
                            if (out_mask[v] & mask4) == mask4:
                                found = True
                                break
                        if not found:
                            bad_count += 1
        return bad_count
    
    def local_search(n, max_iter=50000, time_limit=10):
        """Local search to find a tournament on n vertices with 0 bad 4-sets."""
        start = time.time()
        
        # Random initial tournament
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
        # Build out_mask
        def build_out_mask(orientation, n):
            out_mask = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if orientation[idx] == 0:
                        out_mask[i] |= (1 << j)
                    else:
                        out_mask[j] |= (1 << i)
                    idx += 1
            return out_mask
        
        out_mask = build_out_mask(orientation, n)
        
        def count_bad_fast(out_mask, n):
            """Count bad 4-sets using precomputed out_masks."""
            bad = 0
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        for d in range(c + 1, n):
                            mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            for v in range(n):
                                if (out_mask[v] & mask4) == mask4:
                                    break
                            else:
                                bad += 1
            return bad
        
        current_bad = count_bad_fast(out_mask, n)
        best_bad = current_bad
        best_orientation = orientation[:]
        
        for iteration in range(max_iter):
            if time.time() > start + time_limit:
                break
            if current_bad == 0:
                break
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, n * (n - 1) // 2 - 1)
            
            # Find i, j from edge_idx
            # edge_idx corresponds to pair (i, j) with i < j
            # Number of edges before i: i*(n-1) - i*(i-1)//2 ... let's compute properly
            # Pairs in lex order: (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...,(1,n-1),...
            # For vertex i, the edges starting at i are (i, i+1), ..., (i, n-1): n-1-i edges
            # Cumulative: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
            i = 0
            remaining = edge_idx
            while remaining >= n - 1 - i:
                remaining -= n - 1 - i
                i += 1
            j = i + 1 + remaining
            
            # Flip the edge
            orientation[edge_idx] = 1 - orientation[edge_idx]
            
            # Update out_mask
            if orientation[edge_idx] == 0:
                # i -> j
                out_mask[i] |= (1 << j)
                out_mask[j] &= ~(1 << i)
            else:
                # j -> i
                out_mask[j] |= (1 << i)
                out_mask[i] &= ~(1 << j)
            
            new_bad = count_bad_fast(out_mask, n)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_orientation = orientation[:]
            else:
                # Revert
                orientation[edge_idx] = 1 - orientation[edge_idx]
                if orientation[edge_idx] == 0:
                    out_mask[i] |= (1 << j)
                    out_mask[j] &= ~(1 << i)
                else:
                    out_mask[j] |= (1 << i)
                    out_mask[i] &= ~(1 << j)
        
        return best_bad, best_orientation
    
    # First, try Paley 67 (known valid, fast to construct)
    p = 67
    if p % 4 == 3:
        orientation_67 = paley_tournament(p)
        # We trust Paley 67 is valid. Let's do a quick sanity check on a subset.
        # Actually, let's just return it if we can't find something smaller.
    
    # Try local search for n from 31 upward, but limit total time
    # Start from smaller n and go up, but we need to be careful about time.
    # Let's try a few promising values.
    
    # Actually, the known result is that f(4) is between 31 and 67.
    # Let's try to find a valid tournament for n < 67.
    
    # Try n values in decreasing order of likelihood, or just try a range
    # Given time constraints, let's try n = 50, 55, 60, 65 first
    # Actually, let's just do a quick search for a few n values
    
    search_time_budget = 120  # seconds
    search_start = time.time()
    
    found = False
    best_n = 67
    best_orientation = orientation_67
    
    # Try to find smaller n
    for n_try in range(31, 67):
        if time.time() > search_start + search_time_budget:
            break
        
        # Give each n_try a time budget
        time_per_n = min(5, max(1, (search_time_budget - (time.time() - search_start)) / (67 - n_try)))
        
        bad, orient = local_search(n_try, max_iter=20000, time_limit=time_per_n)
        
        if bad == 0:
            best_n = n_try
            best_orientation = orient
            found = True
            break
    
    if not found:
        # Fall back to Paley 67
        best_n = 67
        best_orientation = orientation_67
    
    return {'n': best_n, 'orientation': best_orientation}
