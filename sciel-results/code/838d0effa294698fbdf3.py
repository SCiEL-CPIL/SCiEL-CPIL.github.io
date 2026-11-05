import random
import time
import itertools
import numpy as np

N = 16
N_PTS = N * N
ALL_POINTS = list(itertools.product(range(N), repeat=2))

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute ISO_PAIR[p][u] = bitmask of q such that {p, q, u} is isosceles
ISO_PAIR = [[0] * N_PTS for _ in range(N_PTS)]

def _precompute_iso_pairs():
    for p in range(N_PTS):
        for u in range(N_PTS):
            if p == u:
                continue
            mask = 0
            dp_u = dist2[p, u]
            for q in range(N_PTS):
                if q == p or q == u:
                    continue
                dp_q = dist2[p, q]
                du_q = dist2[u, q]
                if dp_q == dp_u or dp_q == du_q or dp_u == du_q:
                    mask |= (1 << q)
            ISO_PAIR[p][u] = mask

_start_time = time.time()
_precompute_iso_pairs()

def _mask_to_list(mask):
    pts = []
    temp = mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        pts.append(idx)
        temp ^= lsb
    return pts

def _greedy_bitset(S_init_mask=0, C_init_mask=None, deadline=None):
    S_mask = S_init_mask
    
    if C_init_mask is None:
        C_mask = (1 << N_PTS) - 1
        C_mask &= ~S_mask
        s_points = []
        temp_s = S_mask
        while temp_s:
            lsb = temp_s & -temp_s
            idx = lsb.bit_length() - 1
            s_points.append(idx)
            temp_s ^= lsb
            
        for i in range(len(s_points)):
            for j in range(i+1, len(s_points)):
                u = s_points[i]
                v = s_points[j]
                C_mask &= ~ISO_PAIR[u][v]
    else:
        C_mask = C_init_mask

    while C_mask:
        if deadline and time.time() > deadline:
            break
            
        best_p = -1
        best_c_size = float('inf')
        best_new_c = 0
        
        c_temp = C_mask
        while c_temp:
            lsb = c_temp & -c_temp
            p = lsb.bit_length() - 1
            c_temp ^= lsb
            
            f_p = 0
            s_temp = S_mask
            while s_temp:
                lsb_u = s_temp & -s_temp
                u = lsb_u.bit_length() - 1
                s_temp ^= lsb_u
                f_p |= ISO_PAIR[p][u]
            
            new_c = C_mask & ~(1 << p) & ~f_p
            c_size = new_c.bit_count()
            
            if c_size < best_c_size:
                best_c_size = c_size
                best_p = p
                best_new_c = new_c
        
        if best_p == -1:
            break
            
        S_mask |= (1 << best_p)
        C_mask = best_new_c
        
    return S_mask

def _is_maximal_mask(S_mask):
    s_points = _mask_to_list(S_mask)
    if len(s_points) < 2:
        return False

    blocked_mask = 0
    for i in range(len(s_points)):
        for j in range(i+1, len(s_points)):
            u = s_points[i]
            v = s_points[j]
            blocked_mask |= ISO_PAIR[u][v]
            
    all_mask = (1 << N_PTS) - 1
    available = all_mask & ~S_mask & ~blocked_mask
    return available == 0

def _branch_and_bound_minimal_maximal(deadline, max_depth=20):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    Uses bitset-based constraint propagation.
    """
    best_S = None
    best_size = float('inf')
    
    # State: (S_mask, C_mask, blocked_mask)
    # S_mask: points in current set
    # C_mask: candidates that can still be added
    # blocked_mask: points that are blocked (can't be added)
    
    def _propagate(S_mask, C_mask, blocked_mask):
        """
        Propagate constraints. Returns (S_mask, C_mask, blocked_mask, feasible)
        """
        # Compute blocked from S
        s_points = _mask_to_list(S_mask)
        computed_blocked = 0
        for i in range(len(s_points)):
            for j in range(i+1, len(s_points)):
                u = s_points[i]
                v = s_points[j]
                computed_blocked |= ISO_PAIR[u][v]
        
        # Update blocked_mask
        blocked_mask |= computed_blocked
        
        # Update C_mask: remove blocked points and S points
        C_mask = C_mask & ~S_mask & ~blocked_mask
        
        # Check if we can still complete to a maximal set
        # If C_mask is empty, we need S to be maximal
        # A point x not in S is addable if x not in blocked_mask
        # So if there exists x not in S and not in blocked_mask, we can add it
        # But we're trying to minimize, so we only need to check if S can become maximal
        
        # Actually, for feasibility: we need to be able to reach a maximal set
        # This means: either C_mask is empty and S is maximal,
        # or we can add some points from C_mask to make it maximal
        
        # Simple check: if C_mask is empty, S must be maximal
        if C_mask == 0:
            all_mask = (1 << N_PTS) - 1
            available = all_mask & ~S_mask & ~blocked_mask
            if available != 0:
                return S_mask, C_mask, blocked_mask, False
        
        return S_mask, C_mask, blocked_mask, True
    
    def _bnb(S_mask, C_mask, blocked_mask, depth):
        nonlocal best_S, best_size
        
        if time.time() > deadline:
            return
        
        if depth > max_depth:
            return
        
        # Propagate
        S_mask, C_mask, blocked_mask, feasible = _propagate(S_mask, C_mask, blocked_mask)
        if not feasible:
            return
        
        # Update best if S is maximal
        all_mask = (1 << N_PTS) - 1
        available = all_mask & ~S_mask & ~blocked_mask
        if available == 0:
            # S is maximal
            size = S_mask.bit_count()
            if size < best_size:
                best_size = size
                best_S = S_mask
            return
        
        # Prune: if current size + 0 >= best_size, no improvement possible
        # (we can't add negative points)
        if S_mask.bit_count() >= best_size:
            return
        
        # Branch: try adding each candidate from C_mask
        # Order by some heuristic (e.g., blocking power)
        c_list = _mask_to_list(C_mask)
        
        # Sort by blocking power (points that block more should be tried first)
        def _blocking_power(p):
            f = 0
            s_temp = S_mask
            while s_temp:
                lsb_u = s_temp & -s_temp
                u = lsb_u.bit_length() - 1
                s_temp ^= lsb_u
                f |= ISO_PAIR[p][u]
            return f.bit_count()
        
        c_list.sort(key=lambda p: -_blocking_power(p))
        
        for p in c_list:
            if time.time() > deadline:
                return
            
            # Branch: add p
            new_S = S_mask | (1 << p)
            new_blocked = blocked_mask
            new_C = C_mask & ~(1 << p)
            
            # Propagate immediately
            new_S, new_C, new_blocked, feasible = _propagate(new_S, new_C, new_blocked)
            if not feasible:
                continue
            
            _bnb(new_S, new_C, new_blocked, depth + 1)
    
    # Initial state
    initial_S = 0
    initial_C = (1 << N_PTS) - 1
    initial_blocked = 0
    
    _bnb(initial_S, initial_C, initial_blocked, 0)
    
    return best_S

def _local_search_bitset(S_mask, deadline, max_iter=100):
    current_S = S_mask
    current_size = current_S.bit_count()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        s_points = _mask_to_list(current_S)
        random.shuffle(s_points)
        
        for p in s_points:
            if time.time() > deadline:
                break
            S_temp = current_S & ~(1 << p)
            
            s_temp_pts = _mask_to_list(S_temp)
            blocked = 0
            for i in range(len(s_temp_pts)):
                for j in range(i+1, len(s_temp_pts)):
                    u = s_temp_pts[i]
                    v = s_temp_pts[j]
                    blocked |= ISO_PAIR[u][v]
            
            C_temp = ((1 << N_PTS) - 1) & ~S_temp & ~blocked
            
            S_new = _greedy_bitset(S_temp, C_temp, deadline)
            
            new_size = S_new.bit_count()
            if new_size < current_size:
                current_S = S_new
                current_size = new_size
                improved = True
                break
        
        if improved:
            continue
            
        if len(s_points) >= 2:
            for i in range(min(len(s_points), 10)):
                if time.time() > deadline:
                    break
                p1 = s_points[i]
                for j in range(i+1, min(len(s_points), i+10)):
                    p2 = s_points[j]
                    S_temp = current_S & ~(1 << p1) & ~(1 << p2)
                    
                    s_temp_pts = _mask_to_list(S_temp)
                    blocked = 0
                    for k in range(len(s_temp_pts)):
                        for l in range(k+1, len(s_temp_pts)):
                            u = s_temp_pts[k]
                            v = s_temp_pts[l]
                            blocked |= ISO_PAIR[u][v]
                    
                    C_temp = ((1 << N_PTS) - 1) & ~S_temp & ~blocked
                    S_new = _greedy_bitset(S_temp, C_temp, deadline)
                    
                    new_size = S_new.bit_count()
                    if new_size < current_size:
                        current_S = S_new
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            break
            
    return current_S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_S_mask = 0
    best_size = float('inf')
    
    # First try branch and bound with limited depth
    bnb_S = _branch_and_bound_minimal_maximal(deadline - 200.0, max_depth=15)
    if bnb_S is not None:
        bnb_size = bnb_S.bit_count()
        if bnb_size < best_size:
            best_size = bnb_size
            best_S_mask = bnb_S
    
    # Multiple restarts with greedy + local search
    seed = 42
    while time.time() < deadline - 20.0:
        random.seed(seed)
        
        S_mask = _greedy_bitset(0, None, deadline - 10.0)
        
        S_mask = _local_search_bitset(S_mask, deadline - 5.0, max_iter=50)
        
        size = S_mask.bit_count()
        if size < best_size:
            best_size = size
            best_S_mask = S_mask
            
        if seed % 2 == 0:
            pass
        else:
            S_mask = 0
            C_mask = (1 << N_PTS) - 1
            while C_mask:
                if time.time() > deadline - 10.0:
                    break
                c_list = _mask_to_list(C_mask)
                p = random.choice(c_list)
                
                S_mask |= (1 << p)
                
                s_pts = _mask_to_list(S_mask)
                f_p = 0
                for u in s_pts:
                    f_p |= ISO_PAIR[p][u]
                    
                C_mask = C_mask & ~(1 << p) & ~f_p
                
            S_mask = _local_search_bitset(S_mask, deadline - 5.0, max_iter=50)
            size = S_mask.bit_count()
            if size < best_size:
                best_size = size
                best_S_mask = S_mask
                 
        seed += 1
        
    if best_S_mask == 0:
        best_S_mask = _greedy_bitset(0, None, deadline - 5.0)
        
    return [ALL_POINTS[i] for i in _mask_to_list(best_S_mask)]
