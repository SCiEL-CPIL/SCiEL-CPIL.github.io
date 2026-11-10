import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k = 3

    # Field operations
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Precompute line_id
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
    for d in range(n_dirs):
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d] = field_to_int(c)
        else:
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Precompute line members for fast affected point retrieval
    # line_members[d, lid] = list of points
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    class NikodymTracker:
        def __init__(self):
            self.S = np.zeros(n_points, dtype=bool)
            self.counts = np.zeros((n_dirs, q), dtype=np.int32)
            self.sat = np.ones(n_points, dtype=bool) # Initially all satisfied (empty S)
            
        def initialize(self, S):
            self.S = S.copy()
            self.counts = np.zeros((n_dirs, q), dtype=np.int32)
            for d in range(n_dirs):
                lids = line_id[self.S, d]
                self.counts[d] = np.bincount(lids, minlength=q)
            self.update_sat()
            
        def update_sat(self):
            # Global update
            # For each point y, check if any direction d satisfies it
            # Vectorized:
            # lids: (n_points, n_dirs)
            # counts_gathered: (n_points, n_dirs)
            lids = line_id # (n_points, n_dirs)
            # We need counts[d, lid] for each y, d
            # counts is (n_dirs, q). lids is (n_points, n_dirs).
            # We can use advanced indexing.
            # counts_gathered[y, d] = counts[d, lids[y, d]]
            # This is tricky with standard indexing.
            # Let's transpose.
            # counts_T is (q, n_dirs).
            # We want to gather from counts_T using lids? No.
            
            # Simpler: loop over directions? No, too slow.
            # Let's use the fact that we only update affected points in the search.
            # For initialization, we can just loop or use a slow vectorized method.
            # Given n_points=2401, a Python loop is 2401 * 50 = 120k ops. Fast enough for init.
            for y in range(n_points):
                is_sat = False
                for d in range(n_dirs):
                    lid = line_id[y, d]
                    c = self.counts[d, lid]
                    if self.S[y]:
                        if c == 1:
                            is_sat = True
                            break
                    else:
                        if c == 0:
                            is_sat = True
                            break
                self.sat[y] = is_sat

        def try_add(self, x):
            if self.S[x]:
                return True
            # Check if x has an empty line
            # If not, it cannot be added
            # lids_x = line_id[x, :]
            # counts_x = self.counts[:, lids_x]
            # if not np.any(counts_x == 0): return False
            
            # Tentatively add
            self.S[x] = True
            affected = set()
            for d in range(n_dirs):
                lid = line_id[x, d]
                self.counts[d, lid] += 1
                members = line_members[d][lid]
                affected.update(members.tolist())
            
            self.update_sat_local(affected)
            
            if np.all(self.sat):
                return True
            else:
                # Revert
                self.S[x] = False
                for d in range(n_dirs):
                    lid = line_id[x, d]
                    self.counts[d, lid] -= 1
                self.update_sat_local(affected)
                return False

        def try_remove(self, x):
            if not self.S[x]:
                return True
            self.S[x] = False
            affected = set()
            for d in range(n_dirs):
                lid = line_id[x, d]
                self.counts[d, lid] -= 1
                members = line_members[d][lid]
                affected.update(members.tolist())
            
            self.update_sat_local(affected)
            return True # Removal always keeps validity if it was valid? 
            # Wait, removing a point might make an S point invalid (if it was the only one on a line? No, count becomes 0, so it's not in S anymore? 
            # If we remove x from S, x is now in N.
            # x needs an empty line.
            # Other S points z on lines through x: count decreases.
            # If count becomes 0, z is still in S? No, z is in S.
            # If z is in S, it needs count 1. If count becomes 0, z is invalid.
            # So removal can break validity.
            
            if np.all(self.sat):
                return True
            else:
                self.S[x] = True
                for d in range(n_dirs):
                    lid = line_id[x, d]
                    self.counts[d, lid] += 1
                self.update_sat_local(affected)
                return False

        def update_sat_local(self, affected_list):
            if not affected_list:
                return
            affected_arr = np.array(list(affected_list), dtype=np.int32)
            # lids: (len, n_dirs)
            lids = line_id[affected_arr, :]
            # counts_gathered: (len, n_dirs)
            # We need counts[d, lids[i, d]]
            # Let's build it manually for speed or use a trick.
            # Since n_dirs is small (50), we can loop d and update sat for affected points?
            # No, we want to check if ANY d works.
            
            # Let's just loop over affected points and check all directions.
            # len(affected) <= 2450. 2450 * 50 = 122k ops.
            # In Python, this is ~0.05s. Too slow for many moves.
            
            # We need a faster local update.
            # Let's use NumPy broadcasting if possible.
            # counts is (50, 49). lids is (M, 50).
            # We want M x 50 matrix C where C[i, d] = counts[d, lids[i, d]].
            # This is a gather operation.
            # We can do:
            # indices = lids.T # (50, M)
            # C = counts[indices] # (50, M) -> (50, M)
            # C = C.T # (M, 50)
            
            indices = lids.T # (50, M)
            C = self.counts[indices] # (50, M)
            C = C.T # (M, 50)
            
            S_aff = self.S[affected_arr] # (M,)
            
            valid = (S_aff[:, None] & (C == 1)) | (~S_aff[:, None] & (C == 0))
            new_sat = valid.any(axis=1)
            
            self.sat[affected_arr] = new_sat

    start_time = time.time()
    deadline = start_time + 590
    
    best_size = 0
    best_S = None
    
    # Strategy: Random Restart Greedy
    for trial in range(100):
        if time.time() > deadline:
            break
        tracker = NikodymTracker()
        tracker.initialize(np.zeros(n_points, dtype=bool))
        
        order = np.random.permutation(n_points)
        for x in order:
            if time.time() > deadline:
                break
            if tracker.try_add(x):
                pass
            # If add fails, we just skip.
        
        size = np.sum(tracker.S)
        if size > best_size:
            best_size = size
            best_S = tracker.S.copy()
            
        # Optional: Local Search / Swaps
        # Try to swap points to improve?
        
    if best_S is None:
        best_S = np.zeros(n_points, dtype=bool)

    N_points = []
    for idx in range(n_points):
        if not best_S[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        for idx in range(n_points):
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
