import time
import random
import numpy as np
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Precompute "others" for each number for fast local search
    # ap_others[x] is a list of tuples (o1, o2, o3) such that (x, o1, o2, o3) is a 4-AP
    ap_others = {i: [] for i in range(1, N + 1)}
    for ap in all_aps:
        for pos, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != pos)
            ap_others[x].append(others)

    best_set = set()
    
    # 1. Try MILP if available
    if HAS_SCIPY_MILP:
        try:
            # Objective: maximize sum x_i -> minimize -sum x_i
            c = np.ones(N, dtype=float) * -1.0
            
            # Constraints: sum_{j in AP} x_j <= 3
            row_idx = []
            col_idx = []
            data = []
            for i, ap in enumerate(all_aps):
                for x in ap:
                    row_idx.append(i)
                    col_idx.append(x - 1)  # 0-indexed for milp
                    data.append(1.0)
            
            A = coo_matrix((data, (row_idx, col_idx)), shape=(num_aps, N)).tocsr()
            
            cons = LinearConstraint(A, lb=-np.inf, ub=3)
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(N, dtype=int)  # All integer
            
            # Set a time limit if possible? milp doesn't expose it directly in all versions.
            # We rely on it being fast for N=512.
            res = milp(c, constraints=cons, bounds=bounds, integrality=integrality)
            
            if res.success and res.x is not None:
                milp_set = set(int(x) + 1 for x in res.x if x > 0.5)
                # Verify validity
                in_set_check = bytearray(N + 1)
                for x in milp_set:
                    in_set_check[x] = 1
                valid = True
                for ap in all_aps:
                    if in_set_check[ap[0]] and in_set_check[ap[1]] and in_set_check[ap[2]] and in_set_check[ap[3]]:
                        valid = False
                        break
                if valid:
                    best_set = milp_set
        except Exception:
            pass

    # 2. Local Search (always run to polish or if MILP failed)
    if time.time() < deadline - 2:
        # Initialize best_set if empty
        if not best_set:
            in_set = bytearray(N + 1)
            order = list(range(1, N + 1))
            random.shuffle(order)
            for x in order:
                can_add = True
                for o1, o2, o3 in ap_others[x]:
                    if in_set[o1] and in_set[o2] and in_set[o3]:
                        can_add = False
                        break
                if can_add:
                    in_set[x] = 1
                    best_set.add(x)
        
        # Setup for local search
        in_set = bytearray(N + 1)
        for x in best_set:
            in_set[x] = 1
        
        best_size = len(best_set)
        
        # Local search with kicks
        for restart in range(20000):
            if time.time() > deadline - 1:
                break
            
            # Save current state
            saved_in_set = in_set[:]
            current_set = set(best_set)
            
            # Kick: remove k random elements
            k = random.randint(2, 12)
            elements = list(current_set)
            random.shuffle(elements)
            
            for _ in range(min(k, len(elements))):
                rem = elements.pop()
                current_set.remove(rem)
                in_set[rem] = 0
            
            # Refill greedily
            candidates = [j for j in range(1, N + 1) if not in_set[j]]
            random.shuffle(candidates)
            
            for c in candidates:
                can_add = True
                for o1, o2, o3 in ap_others[c]:
                    if in_set[o1] and in_set[o2] and in_set[o3]:
                        can_add = False
                        break
                if can_add:
                    in_set[c] = 1
                    current_set.add(c)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
            else:
                # Restore state
                in_set[:] = saved_in_set

    return sorted(best_set)
