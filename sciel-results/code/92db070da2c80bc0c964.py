# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    num_dirs = q + 1  # 0..48 slopes, 49 vertical

    # Field arithmetic helpers
    def add(z1, z2):
        return ((z1[0] + z2[0]) % p, (z1[1] + z2[1]) % p)
    
    def mul(z1, z2):
        return ((z1[0]*z2[0] + 3*z1[1]*z2[1]) % p, (z1[0]*z2[1] + z1[1]*z2[0]) % p)

    def get_idx(xa, xb, ya, yb):
        return xa + xb*p + ya*p*p + yb*p*p*p
        
    def get_coords(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p*p)) % p
        yb = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)

    # Precompute directions
    dir_mat = np.full((n_points, n_points), -1, dtype=np.int32)
    
    # Generate points and compute dir_mat
    # y_idx is the base point. x_idx is a point on the line.
    for y_idx in range(n_points):
        y_coords = get_coords(y_idx)
        y_x = y_coords[0]
        y_y = y_coords[1]
        
        # Vertical
        for x_idx in range(n_points):
            x_coords = get_coords(x_idx)
            if x_coords[0] == y_x and x_idx != y_idx:
                dir_mat[y_idx, x_idx] = 49
                
        # Sloped
        for ma in range(p):
            for mb in range(p):
                m = (ma, mb)
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        dx = add(y_x, t)
                        dy = add(y_y, mul(t, m))
                        x_idx = get_idx(dx[0], dx[1], dy[0], dy[1])
                        if x_idx != y_idx:
                            dir_mat[y_idx, x_idx] = ma + mb*p

    # Seed S construction
    k = 3 # alpha^2 = 3
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    seed_S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        seed_S.add(get_idx(xa, xb, ya, yb))

    best_S = seed_S.copy()
    best_size = len(seed_S)
    
    start_time = time.time()
    timeout = 590

    # Helper to initialize cnt and num_wit from a set S
    def init_state(S):
        cnt = np.zeros((n_points, num_dirs), dtype=np.int32)
        num_wit = np.zeros(n_points, dtype=np.int32)
        if not S:
            return cnt, num_wit, S
        
        S_list = np.array(list(S), dtype=np.int32)
        
        # Update cnt for all y based on points in S
        for x in S_list:
            v_indices = dir_mat[:, x]
            mask = v_indices != -1
            ys = np.arange(n_points)[mask]
            vs = v_indices[mask]
            cnt[ys, vs] += 1
            
        # Calculate num_wit for points in S
        for u in S_list:
            num_wit[u] = (cnt[u] == 0).sum()
            
        return cnt, num_wit, S

    # Run greedy add from seed
    cnt, num_wit, S = init_state(seed_S)
    remaining = [i for i in range(n_points) if i not in S]
    np.random.shuffle(remaining)
    
    for x in remaining:
        if not (cnt[x] == 0).any():
            continue
            
        if S:
            u_list = np.array(list(S), dtype=np.int32)
            v_indices = dir_mat[u_list, x]
            old_cnts = cnt[u_list, v_indices]
            mask_was_wit = (old_cnts == 0)
            if np.any(mask_was_wit):
                potential_kills = u_list[mask_was_wit]
                if np.any(num_wit[potential_kills] == 1):
                    continue
        
        S.add(x)
        
        if S:
            u_list = np.array(list(S), dtype=np.int32)
            v_indices = dir_mat[u_list, x]
            old_cnts = cnt[u_list, v_indices]
            num_wit[u_list] -= (old_cnts == 0)
            
        num_wit[x] = (cnt[x] == 0).sum()
        
        v_indices_all = dir_mat[:, x]
        mask = v_indices_all != -1
        ys = np.arange(n_points)[mask]
        vs = v_indices_all[mask]
        cnt[ys, vs] += 1
        
    if len(S) > best_size:
        best_S = S.copy()
        best_size = len(S)

    # Random restarts with empty S
    while time.time() - start_time < timeout:
        S = set()
        cnt, num_wit, _ = init_state(S)
        order = np.random.permutation(n_points)
        
        for x in order:
            if not (cnt[x] == 0).any():
                continue
            if S:
                u_list = np.array(list(S), dtype=np.int32)
                v_indices = dir_mat[u_list, x]
                old_cnts = cnt[u_list, v_indices]
                mask_was_wit = (old_cnts == 0)
                if np.any(mask_was_wit):
                    potential_kills = u_list[mask_was_wit]
                    if np.any(num_wit[potential_kills] == 1):
                        continue
            S.add(x)
            if S:
                u_list = np.array(list(S), dtype=np.int32)
                v_indices = dir_mat[u_list, x]
                old_cnts = cnt[u_list, v_indices]
                num_wit[u_list] -= (old_cnts == 0)
            num_wit[x] = (cnt[x] == 0).sum()
            v_indices_all = dir_mat[:, x]
            mask = v_indices_all != -1
            ys = np.arange(n_points)[mask]
            vs = v_indices_all[mask]
            cnt[ys, vs] += 1
            
        if len(S) > best_size:
            best_S = S.copy()
            best_size = len(S)

    N_points = []
    for i in range(n_points):
        if i not in best_S:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)
# EVOLVE-BLOCK-END
