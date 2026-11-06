import time
import itertools
import random

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    n = 16
    
    # Precompute all disjoint pairs of 4-sets
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # Map 4-set to index for fast lookup? 
    # Actually, we just need to iterate.
    # Store 4-sets as frozensets for fast intersection?
    # But we need disjointness.
    
    disjoint_pairs = []
    # To speed up, we can index 4-sets by their elements?
    # For now, brute force disjoint check is O(1) with bitmasks.
    
    # Represent 4-sets as bitmasks
    set_masks = [0] * num_4sets
    for i, s in enumerate(all_4sets):
        mask = 0
        for v in s:
            mask |= (1 << v)
        set_masks[i] = mask
            
    for i in range(num_4sets):
        for j in range(i+1, num_4sets):
            if (set_masks[i] & set_masks[j]) == 0:
                disjoint_pairs.append((i, j))
                
    # For fast checking, for each vertex v, list of 4-set indices containing v
    sets_containing = [[] for _ in range(n)]
    for i, s in enumerate(all_4sets):
        for v in s:
            sets_containing[v].append(i)

    def is_matching(h_mask, A_idx, B_idx):
        """
        Check if the edges of H between A and B form a matching.
        h_mask is the global bitmask of H edges? 
        Better to pass the local bitmask or check adj.
        """
        pass

    # We will maintain adj_G (bitmask per vertex)
    # H edges are the complement.
    
    def check_pair(adj, A_idx, B_idx):
        """
        Returns True if H[A,B] is NOT a matching (i.e., G is safe for this pair).
        Returns False if H[A,B] IS a matching (i.e., G contains Q3 for this pair).
        """
        A = all_4sets[A_idx]
        B = all_4sets[B_idx]
        
        # Collect non-edges in G (which are edges in H) between A and B
        # We need to check if these non-edges form a matching.
        # A matching means: no two non-edges share a vertex.
        
        # Count non-edges
        # If count >= 5, it's not a matching (max matching size in 4x4 is 4).
        # If count <= 4, check if they are disjoint.
        
        non_edges = []
        for u in A:
            for v in B:
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
                    
        if len(non_edges) >= 5:
            return True # Safe
            
        # Check if non_edges form a matching
        # Use a bitmask for vertices used
        used_A = 0
        used_B = 0
        for u, v in non_edges:
            if (used_A & (1 << u)) or (used_B & (1 << v)):
                return True # Not a matching (conflict), so Safe
            used_A |= (1 << u)
            used_B |= (1 << v)
            
        # If we get here, they are all disjoint.
        # So it IS a matching.
        return False # Unsafe

    def is_q3_free(adj):
        for i, j in disjoint_pairs:
            if not check_pair(adj, i, j):
                return False
        return True

    # Initialize G
    # Start with a random valid graph or the seed
    # Let's use the seed from the current program as a strong initial point
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    best_adj = adj[:]
    best_count = sum(bin(a).count('1') for a in adj) // 2
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local Search / Hill Climbing
    # We try to add edges to G (remove from H)
    # If we get stuck, we can try to swap
    
    # To speed up, we track which pairs are "critical" (currently matching or close)
    # But for simplicity, we just check affected pairs.
    
    def get_affected_pairs(u, v):
        """
        Returns list of (A_idx, B_idx) pairs where u in A, v in B or vice versa.
        """
        affected = []
        # u in A, v in B
        for i in sets_containing[u]:
            for j in sets_containing[v]:
                if i == j: continue # Not disjoint? No, if u!=v, sets can be disjoint.
                # Check disjointness
                if (set_masks[i] & set_masks[j]) == 0:
                    affected.append((i, j))
        # u in B, v in A
        for i in sets_containing[v]:
            for j in sets_containing[u]:
                if i == j: continue
                if (set_masks[i] & set_masks[j]) == 0:
                    affected.append((i, j))
        return affected

    random.seed(42)
    missing_edges = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
    random.shuffle(missing_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add each missing edge
        # We need to re-calculate missing edges or maintain a list
        # Since we add edges, missing edges decrease.
        # Let's just iterate through all possible edges and check if they are missing.
        
        # For efficiency, maintain a list of candidates
        candidates = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Try adding (u, v) to G
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check only affected pairs
            valid = True
            affected = get_affected_pairs(u, v)
            for i, j in affected:
                if not check_pair(adj, i, j):
                    valid = False
                    break
            
            if valid:
                best_count += 1
                best_adj = adj[:]
                improved = True
                # Remove (u,v) from candidates if we were maintaining it, 
                # but here we just iterate.
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        # If no improvement, do a random restart or shake?
        # Given time, simple hill climbing might suffice if seed is good.
        # But to beat 1.30, we might need to escape local minima.
        # Let's add a "shake" step if no improvement for a while.
        
    # Final check
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
