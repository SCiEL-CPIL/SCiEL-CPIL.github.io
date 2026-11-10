"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_k4_free_masks(adj_masks):
    """Check if graph (adjacency masks) is K4-free."""
    n = len(adj_masks)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_masks[a] >> b) & 1:
                continue
            # common neighbors of a and b
            common = adj_masks[a] & adj_masks[b]
            # remove a and b from common
            common &= ~(1 << a)
            common &= ~(1 << b)
            # for each c in common, check if there is d in common \ {c} adjacent to c
            # actually, K4 is a,b,c,d all connected.
            # if a,b,c form a triangle, we need d connected to a,b,c.
            # so d must be in common(a,b) & common(a,c) & common(b,c)
            # simpler: iterate c in common
            c_mask = common
            while c_mask:
                lsb = c_mask & -c_mask
                c = lsb.bit_length() - 1
                c_mask ^= lsb
                # d must be neighbor of a, b, c
                d_mask = adj_masks[a] & adj_masks[b] & adj_masks[c]
                d_mask &= ~(1 << a)
                d_mask &= ~(1 << b)
                d_mask &= ~(1 << c)
                if d_mask:
                    return False
    return True

def _compute_score_masks(adj_masks):
    """Compute the largest triangle-free induced subgraph size using bitmask B&B."""
    n = len(adj_masks)
    # Precompute common neighbors
    # common[i][j] = adj[i] & adj[j] & ~(1<<i) & ~(1<<j)
    # We need to know if adding v to current set S creates a triangle.
    # This happens if there exist u, w in S such that v-u, v-w, u-w are edges.
    # i.e., u is in adj[v] & adj[w] for some w in S?
    # Actually, if we add v, the new "conflict" vertices are those u such that
    # u is adjacent to v AND u is adjacent to some w in S which is adjacent to v.
    # Let N(v) be neighbors of v.
    # Let S_v = { w in S | w is adjacent to v }.
    # Then u conflicts if u is in N(v) AND u is in N(w) for some w in S_v.
    # So conflict_mask(v, S) = N(v) & ( union of N(w) for w in S_v ).
    
    best_size = 0
    full_mask = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count("1")
        
    def backtrack(available, current_mask, current_size):
        nonlocal best_size
        if current_size > best_size:
            best_size = current_size
            
        # Upper bound: current_size + popcount(available)
        if current_size + popcount(available) <= best_size:
            return
            
        if available == 0:
            return
            
        # Heuristic: pick vertex with highest degree in available?
        # Or just pick the first one.
        # Let's pick a vertex that is "hard" to place, i.e., high conflict potential.
        # For now, pick lowest bit.
        v_bit = available & -available
        v = v_bit.bit_length() - 1
        available ^= v_bit
        
        # Option 1: Exclude v
        backtrack(available, current_mask, current_size)
        
        # Option 2: Include v
        # Check if v can be added to current_mask
        # v is compatible if no triangle is formed with any pair in current_mask.
        # A triangle is formed if there is w in current_mask such that
        # v-w is edge AND there is u in current_mask such that v-u, u-w are edges.
        # Actually, simpler:
        # If we add v, we must ensure that no triangle (v, u, w) exists with u,w in current_mask.
        # This is equivalent to: current_mask does not contain any edge (u,w) such that
        # u and w are both neighbors of v.
        # Let N_v = adj_masks[v].
        # Let induced_edges = edges in current_mask that are both in N_v.
        # If such an edge exists, v cannot be added.
        
        # How to check fast?
        # We can maintain a bitmask of "forbidden" vertices?
        # Or just check:
        valid = True
        # Iterate through vertices in current_mask
        cm = current_mask
        while cm:
            lsb = cm & -cm
            u = lsb.bit_length() - 1
            cm ^= lsb
            if (adj_masks[v] >> u) & 1:
                # u is neighbor of v
                # Check if u has any neighbor w in current_mask that is also neighbor of v
                # common = adj_masks[u] & adj_masks[v] & current_mask
                common = adj_masks[u] & adj_masks[v] & current_mask
                if common:
                    valid = False
                    break
        
        if valid:
            # New current mask
            new_current = current_mask | v_bit
            # Update available: remove v (done), remove any vertex that would form a triangle with v and some vertex in new_current?
            # Actually, the `available` set in this recursion represents vertices that can be added to *some* extension?
            # No, standard B&B for Max Independent Set style:
            # `available` contains vertices that are compatible with `current_mask`.
            # When we add v, some vertices in `available` might become incompatible with `current_mask U {v}`.
            # Specifically, a vertex x in `available` is incompatible if adding x would create a triangle with v and some w in current_mask?
            # No, x is incompatible if x, v, and some w in current_mask form a triangle?
            # Or x, v, and some other x' in available? No, we only care about compatibility with the *current* set.
            # If x is compatible with current_mask, but (x, v, w) is a triangle for some w in current_mask, then x is no longer compatible.
            # So we need to remove from `available` any x that forms a triangle with v and some w in current_mask.
            
            # conflict_mask = { x | exists w in current_mask s.t. x-v, x-w, v-w are edges }
            # x is in adj[v] & adj[w] for some w in current_mask s.t. v-w is edge.
            
            conflict = 0
            cm = current_mask
            while cm:
                lsb = cm & -cm
                w = lsb.bit_length() - 1
                cm ^= lsb
                if (adj_masks[v] >> w) & 1:
                    # v and w are connected
                    # x must be neighbor of v and w
                    conflict |= adj_masks[v] & adj_masks[w]
            
            new_available = available & ~conflict & ~v_bit # v_bit already removed from available
            
            backtrack(new_available, new_current, current_size + 1)
            
    backtrack(full_mask, 0, 0)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 18
    best_score = 18
    best_edges = []
    
    # Initial construction: K6,6,6
    # Parts 0-5, 6-11, 12-17
    adj_masks = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
        for k in range(12, 18):
            adj_masks[i] |= (1 << k)
            adj_masks[k] |= (1 << i)
    for i in range(6, 12):
        for k in range(12, 18):
            adj_masks[i] |= (1 << k)
            adj_masks[k] |= (1 << i)
            
    if _is_k4_free_masks(adj_masks):
        score = _compute_score_masks(adj_masks)
        if score < best_score:
            best_score = score
            best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if (adj_masks[a] >> b) & 1]
            
    # Local search
    random.seed(42)
    current_masks = adj_masks[:]
    current_score = best_score
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        # Flip edge
        bit = 1 << v
        if current_masks[u] & bit:
            current_masks[u] ^= bit
            current_masks[v] ^= (1 << u)
        else:
            current_masks[u] |= bit
            current_masks[v] |= (1 << u)
            
        # Check K4-free
        if not _is_k4_free_masks(current_masks):
            # Revert
            if current_masks[u] & bit:
                current_masks[u] ^= bit
                current_masks[v] ^= (1 << u)
            else:
                current_masks[u] |= bit
                current_masks[v] |= (1 << u)
            continue
            
        # Compute score
        new_score = _compute_score_masks(current_masks)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if (current_masks[a] >> b) & 1]
            current_score = new_score
        else:
            # Revert
            if current_masks[u] & bit:
                current_masks[u] ^= bit
                current_masks[v] ^= (1 << u)
            else:
                current_masks[u] |= bit
                current_masks[v] |= (1 << u)
                
    return best_edges
# EVOLVE-BLOCK-END
