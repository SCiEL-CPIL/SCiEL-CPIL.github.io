import time
import math
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build conflict list
    conflicts = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                conflicts.append((i, j))

    # 4. Try ILP first
    if conflicts:
        c_obj = np.ones(n_vars) * (-1.0)
        m_conf = len(conflicts)
        A_mat = np.zeros((m_conf, n_vars))
        for idx, (i, j) in enumerate(conflicts):
            A_mat[idx, i] = 1.0
            A_mat[idx, j] = 1.0
        constraints = LinearConstraint(A_mat, ub=np.ones(m_conf))
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        try:
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0 and res.x is not None:
                selected = []
                for i in range(n_vars):
                    if res.x[i] > 0.5:
                        selected.append(valid_vertices[i])
                if selected:
                    valid = True
                    for v in selected:
                        if 2 * v in squares:
                            valid = False
                            break
                    if valid:
                        for ii in range(len(selected)):
                            for jj in range(ii + 1, len(selected)):
                                if selected[ii] + selected[jj] in squares:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        return selected
        except Exception:
            pass
    
    def greedy_init(randomize=False):
        """
        Greedy construction of an independent set.
        If randomize, pick vertices with some randomness among low-degree candidates.
        """
        avail = (1 << n_vars) - 1
        S = 0
        while avail:
            # Find vertex with min degree in remaining graph
            # Degree of v in avail is popcount(adj_bits[v] & avail)
            best_v = -1
            best_deg = float('inf')
            
            if randomize:
                # Pick a few candidates and choose one
                # Simple: just pick first with min degree, but break ties randomly
                candidates = []
                min_deg = float('inf')
                for v in range(n_vars):
                    if avail & (1 << v):
                        deg = bin(adj_bits[v] & avail).count('1')
                        if deg < min_deg:
                            min_deg = deg
                            candidates = [v]
                        elif deg == min_deg:
                            candidates.append(v)
                if candidates:
                    best_v = random.choice(candidates)
                    best_deg = min_deg
            else:
                for v in range(n_vars):
                    if avail & (1 << v):
                        deg = bin(adj_bits[v] & avail).count('1')
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
            
            if best_v == -1:
                break
                
            S |= (1 << best_v)
            # Remove best_v and its neighbors from avail
            avail &= ~(adj_bits[best_v] | (1 << best_v))
            
        return S

    def local_search(S):
        """
        Improve S using 1-for-2 swaps.
        Returns improved S.
        """
        improved = True
        current_S = S
        while improved:
            improved = False
            # For each u in S, find C_u
            # C_u = {v not in S | N(v) & S == {u}}
            # We need to find an independent pair in C_u
            
            # Precompute for each u in S, the list of candidates
            # To speed up, we can iterate u in S
            
            S_list = [i for i in range(n_vars) if (current_S >> i) & 1]
            
            for u in S_list:
                # Find candidates v not in S such that N(v) & S is exactly {u}
                # N(v) & S should have only bit u set.
                # i.e., (adj_bits[v] & current_S) == (1 << u)
                
                C_u = []
                for v in range(n_vars):
                    if (current_S >> v) & 1:
                        continue
                    # Check if v is connected to any node in S other than u
                    # And is connected to u
                    neigh_in_S = adj_bits[v] & current_S
                    if neigh_in_S == (1 << u):
                        C_u.append(v)
                        
                if len(C_u) < 2:
                    continue
                    
                # Check if there is an independent pair in C_u
                # i.e., v, w in C_u such that not (adj_bits[v] & (1 << w))
                found_pair = False
                for i in range(len(C_u)):
                    v = C_u[i]
                    for j in range(i + 1, len(C_u)):
                        w = C_u[j]
                        # Check if v and w are connected
                        if not (adj_bits[v] & (1 << w)):
                            # They are independent.
                            # New S = S - {u} + {v, w}
                            new_S = current_S
                            new_S &= ~(1 << u)
                            new_S |= (1 << v)
                            new_S |= (1 << w)
                            
                            # Verify new_S is independent (sanity check, should be by construction)
                            # But we must ensure v and w are not connected to each other (checked)
                            # And v, w not connected to S \ {u} (checked by neigh_in_S == 1<<u)
                            
                            current_S = new_S
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    # Restart the outer loop as S changed
                    break
                    
        return current_S

    best_S = 0
    best_size = 0
    
    # Fallback: greedy + local search
    adj_bits = [0] * n_vars
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    # First, a deterministic greedy to get a baseline
    S0 = greedy_init(randomize=False)
    S0 = local_search(S0)
    size0 = bin(S0).count('1')
    best_S = S0
    best_size = size0
    
    # Random restarts
    restarts = 0
    while time.time() < deadline - 5.0: # Keep 5s for final processing
        # Random greedy
        S_rand = greedy_init(randomize=True)
        # Local search
        S_rand = local_search(S_rand)
        size_rand = bin(S_rand).count('1')
        
        if size_rand > best_size:
            best_S = S_rand
            best_size = size_rand
            
        restarts += 1
        
    # Convert best_S (bitset) to list of values
    final_set = []
    for i in range(n_vars):
        if (best_S >> i) & 1:
            final_set.append(valid_vertices[i])
            
    # Verify result
    # Check diagonal
    for v in final_set:
        if 2 * v in squares:
            return [valid_vertices[0]]
    # Check pairs
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if final_set[i] + final_set[j] in squares:
                return [valid_vertices[0]]
                
    return final_set
