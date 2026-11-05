import numpy as np
import itertools

N = 16
PTS = list(itertools.product(range(N), repeat=2))
PTS_ARR = np.array(PTS)
NUM_PTS = len(PTS)

# Precompute IsoConflict[a, b, c]
# a, b, c are indices
# IsoConflict[a, b, c] = 1 if dist(a,b)==dist(a,c) or dist(a,b)==dist(b,c) or dist(a,c)==dist(b,c)

def precompute_conflicts():
    # d[i, j] is squared distance between PTS[i] and PTS[j]
    # d is NUM_PTS x NUM_PTS
    x = PTS_ARR[:, 0]
    y = PTS_ARR[:, 1]
    # d2[i, j] = (xi-xj)^2 + (yi-yj)^2
    # Use broadcasting
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    # d2 is shape (NUM_PTS, NUM_PTS)
    # We need a 3D array or check conditions.
    # Condition 1: d2[a, b] == d2[a, c]
    # Condition 2: d2[a, b] == d2[b, c]
    # Condition 3: d2[a, c] == d2[b, c]
    
    # Let's create a boolean array of shape (NUM_PTS, NUM_PTS, NUM_PTS)
    # This might be large: 256^3 * 1 byte = 16 MB. OK.
    
    # d2_ab = d2[:, :, None] # a, b, dummy
    # d2_ac = d2[:, None, :] # a, dummy, c
    # d2_bc = ? Need d2[b, c].
    
    # d2 is (i, j).
    # d2_ab[a, b, c] = d2[a, b]
    # d2_ac[a, b, c] = d2[a, c]
    # d2_bc[a, b, c] = d2[b, c]
    
    # d2_ab = d2[:, :, None]
    # d2_ac = d2[:, None, :]
    # d2_bc = d2.T[:, :, None] ? No.
    # d2[b, c] for fixed a.
    # We can just loop a? No, vectorize.
    
    # d2_bc[a, b, c] = d2[b, c]
    # d2 is (i, j). We want d2[b, c] where b, c are the 2nd and 3rd indices.
    # This is not a standard slice.
    # d2_bc = d2[:, :, None] doesn't work for bc.
    
    # Let's use np.einsum or just simple broadcasting tricks.
    # d2_ab = d2[:, :, None] # Shape (N, N, 1)
    # d2_ac = d2[:, None, :] # Shape (N, 1, N)
    # d2_bc: We need d2[i, j] where i=b, j=c.
    # d2 is (i, j).
    # If we have indices b, c, d2[b, c] is just d2[b, c].
    # In a 3D array context, d2_bc[a, b, c] = d2[b, c].
    # This is equivalent to d2[None, :, :] broadcast to (1, N, N) and then... wait.
    # d2[None, :, :] is (1, N, N).
    # We want to align b with the 2nd dim and c with the 3rd dim of the result.
    # So d2_bc = d2[None, :, :] works if we broadcast a to match.
    
    d2_ab = d2[:, :, None]
    d2_ac = d2[:, None, :]
    d2_bc = d2[None, :, :]
    
    iso = (d2_ab == d2_ac) | (d2_ab == d2_bc) | (d2_ac == d2_bc)
    
    # Exclude cases where points are not distinct?
    # The problem says "no three points form an isosceles triangle".
    # Usually implies distinct points.
    # If a==b, d2_ab=0. d2_ac=d2_bc. So 0==d2_ac? Only if a==c.
    # If a==b!=c, d2_ab=0, d2_ac=d2_bc. 0==d2_ac is false (unless dist 0).
    # So distinctness is handled naturally by distance equality if we assume non-zero distances for distinct points.
    # However, if a==b, it's not a triangle.
    # We should ensure a, b, c are distinct indices.
    
    # Mask out non-distinct indices
    # a != b, b != c, a != c
    idx = np.arange(NUM_PTS)
    a_idx, b_idx, c_idx = np.indices((NUM_PTS, NUM_PTS, NUM_PTS))
    distinct = (a_idx != b_idx) & (b_idx != c_idx) & (a_idx != c_idx)
    iso = iso & distinct
    
    return iso.astype(np.uint8)

ISO_CONFLICT = None

def construct():
    global ISO_CONFLICT
    if ISO_CONFLICT is None:
        ISO_CONFLICT = precompute_conflicts()
        
    best_size = N * N
    best_set = None
    
    for seed in range(100): # Increase restarts if time allows
        import random
        random.seed(seed)
        S = []
        start_idx = random.randint(0, NUM_PTS - 1)
        p0 = start_idx
        S.append(p0)
        
        valid = np.ones(NUM_PTS, dtype=bool)
        valid[p0] = False
        
        while True:
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                break
            
            if len(valid_indices) == 1:
                next_p = valid_indices[0]
            else:
                # Calculate score for each candidate in valid_indices
                # Score(p) = count of q in valid_indices \ {p} such that 
                # exists u in S with ISO_CONFLICT[q, p, u] == 1
                
                best_p = None
                best_score = -1
                
                # S_indices = np.array(S)
                # For each p in valid_indices:
                #   blocked_q = ISO_CONFLICT[:, p, S_indices].sum(axis=1) > 0
                #   score = blocked_q[valid_indices].sum() - 1 (exclude p itself if blocked, but p is not in S so ISO_CONFLICT[p, p, u] is 0 due to distinct mask? 
                #   Actually, distinct mask ensures a!=b. So ISO_CONFLICT[p, p, u] is 0.
                #   So p is never blocked by itself.
                #   So score = blocked_q[valid_indices].sum()
                
                # Vectorize over p?
                # valid_indices is 1D array of indices.
                # We can compute blocked_q for all p in valid_indices at once?
                # ISO_CONFLICT shape (N, N, N).
                # We want for each p in valid_indices, a vector of length N (for q).
                # ISO_CONFLICT[:, p, :] for each p.
                
                # This is a bit hard to vectorize fully in one shot without large memory.
                # Let's loop over p in valid_indices.
                # len(valid_indices) <= 256.
                # Inner op is vectorized over q and u.
                
                S_arr = np.array(S)
                
                for p in valid_indices:
                    # blocked_q[q] = 1 if exists u in S such that ISO_CONFLICT[q, p, u] == 1
                    # ISO_CONFLICT[:, p, S_arr] has shape (N, len(S))
                    # Sum over axis 1 (u)
                    # result shape (N,)
                    blocked_q = (ISO_CONFLICT[:, p, S_arr].sum(axis=1) > 0).astype(int)
                    
                    # Count how many valid q are blocked
                    # q must be in valid_indices
                    # blocked_q[valid_indices] gives a vector of 0/1 for each valid q
                    score = blocked_q[valid_indices].sum()
                    
                    if score > best_score:
                        best_score = score
                        best_p = p
            
            S.append(best_p)
            
            # Update valid
            # A point q is invalid if exists u in S_old such that ISO_CONFLICT[q, best_p, u] == 1
            # S_old = S[:-1]
            S_old_arr = np.array(S[:-1])
            if len(S_old_arr) > 0:
                newly_blocked = (ISO_CONFLICT[:, best_p, S_old_arr].sum(axis=1) > 0)
                valid &= ~newly_blocked
            
            valid[best_p] = False

        if len(S) < best_size:
            best_size = len(S)
            best_set = [PTS[i] for i in S]
            
    return best_set
