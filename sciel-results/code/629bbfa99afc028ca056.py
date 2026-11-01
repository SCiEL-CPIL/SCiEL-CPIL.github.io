import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin

    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2
    conflicts = (dots > 2).astype(np.bool_)
    np.fill_diagonal(conflicts, False)
    
    rng = np.random.default_rng(42)
    best_set = []
    best_size = 0

    def get_free_mask(S_mask):
        """
        Returns a boolean mask of vertices that have NO conflict with any vertex in S_mask.
        S_mask is a boolean array of length N.
        """
        if not np.any(S_mask):
            return np.ones(N, dtype=bool)
        # conflicts[:, S_mask] has shape (N, |S|)
        # any(axis=1) gives True if any conflict exists
        has_conflict = conflicts[:, S_mask].any(axis=1)
        return ~has_conflict

    def greedy_init():
        S_mask = np.zeros(N, dtype=bool)
        order = rng.permutation(N)
        for v in order:
            if time.time() > deadline - 5:
                break
            if not S_mask[v]:
                # Check if v conflicts with current S
                # S_mask[v] is False, so we check against S
                if not np.any(conflicts[v][S_mask]):
                    S_mask[v] = True
        return S_mask

    def local_improve(S_mask):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Add all free vertices
            free_mask = get_free_mask(S_mask)
            # Exclude vertices already in S
            free_mask &= ~S_mask
            
            if np.any(free_mask):
                S_mask |= free_mask
                improved = True
                continue
            
            # 2. Try 1-for-2 swap
            # Find a u in S such that removing it frees up at least 2 new vertices
            S_indices = np.where(S_mask)[0]
            if len(S_indices) == 0:
                break
                
            # Shuffle to randomize search
            rng.shuffle(S_indices)
            
            swap_found = False
            for u in S_indices:
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove u
                S_mask[u] = False
                
                free_mask = get_free_mask(S_mask)
                # Exclude u itself (it's not in S now, but it might be free)
                # Actually, u is not in S, so it's a candidate.
                # But we want to add NEW vertices.
                # free_mask includes u if u has no conflict with S\{u}.
                # We want to pick 2 vertices from free_mask.
                # Ideally, they should not be u.
                
                # Get indices of free vertices
                free_indices = np.where(free_mask)[0]
                
                # We want to pick 2 from free_indices.
                # If we pick u, we just added it back. Net change 0.
                # We want to pick v1, v2 != u.
                
                # Filter out u
                if u in free_indices:
                     # Create mask for free_indices excluding u
                     # Since free_indices is small (hopefully), we can check
                     pass
                
                # Count how many free vertices are NOT u
                # We need at least 2.
                # If u is free, and there are 2 others, we can pick the others.
                # If u is not free, we need 2 others.
                
                # Simple check:
                if len(free_indices) >= 2:
                    # Pick 2 random ones
                    # Ensure we don't just pick u and nothing else?
                    # If we pick u and v, size stays same.
                    # We want size to increase.
                    # So we need to pick 2 vertices from free_indices \ {u}.
                    
                    # Mask for free_indices excluding u
                    if len(free_indices) == 2 and u in free_indices:
                        # Only 2 free, one is u. We can only pick 1 other.
                        # Size change: -1 (u) + 1 (v) = 0.
                        # Not an improvement in size.
                        # But maybe it helps future steps?
                        # Let's require size increase for now.
                        pass
                    else:
                        # We have at least 2 candidates excluding u?
                        # Or 3 total including u?
                        
                        # Let's just pick 2 distinct vertices from free_indices.
                        # If one is u, we replace u with 2 others? No, we removed u.
                        # We add 2.
                        # If one of them is u, we add u and v.
                        # Net: -1 (u removed) + 2 (u, v added) = +1.
                        # This is valid.
                        
                        # Pick 2 random from free_indices
                        sample_size = min(2, len(free_indices))
                        if sample_size < 2:
                             S_mask[u] = True
                             continue
                             
                        chosen = rng.choice(free_indices, size=2, replace=False)
                        
                        S_mask[chosen] = True
                        S_mask[u] = False # Already False
                        
                        improved = True
                        swap_found = True
                        break
                
                # Restore u if no swap made
                if not swap_found:
                    S_mask[u] = True
        
        return S_mask

    # Main loop
    while time.time() < deadline - 5:
        S_mask = greedy_init()
        size = np.sum(S_mask)
        
        if size > best_size:
            best_size = size
            best_set = np.where(S_mask)[0].copy()
        
        if time.time() < deadline - 10:
            S_mask = local_improve(S_mask)
            size = np.sum(S_mask)
            
            if size > best_size:
                best_size = size
                best_set = np.where(S_mask)[0].copy()
        
        # Optional: Perturb best_set for next restart?
        # For now, just restart from scratch.

    if len(best_set) == 0:
        return np.array([], dtype=np.int64)

    return vectors[best_set].astype(np.int64)
