import time
import numpy as np
from scipy.optimize import linprog

def sign_edges(graph, budget):
    deadline = time.time() + 80.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m_val = n // 2

    # Precompute edge indices
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Precompute 4-cycles
    cycles_4 = []
    for i in range(m_val):
        v1 = i
        v2 = (i + 1) % m_val
        v3 = m_val + i
        v4 = m_val + (i + 1) % m_val
        
        e1 = edge_idx.get((v1, v2))
        e2 = edge_idx.get((v2, v4))
        e3 = edge_idx.get((v4, v3))
        e4 = edge_idx.get((v3, v1))
        
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycles_4.append([e1, e2, e3, e4])

    all_pairs = [(i, j) for i in range(num_edges) for j in range(i+1, num_edges)]
    
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_obj = float('inf')
    
    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    def get_objective(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
        
    def is_connected(signs):
        parent = list(range(2*n))
        rank = [0] * (2*n)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            u0, u1 = 2*u, 2*u + 1
            v0, v1 = 2*v, 2*v + 1
            if s == 1:
                union(u0, v0)
                union(u1, v1)
            else:
                union(u0, v1)
                union(u1, v0)
        
        root = find(0)
        return all(find(i) == root for i in range(2*n))

    def local_search(initial_signs, time_limit=None):
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline
        current_signs = initial_signs.copy()
        current_obj = get_objective(current_signs)
        if not is_connected(current_signs):
             return
        
        update_best(current_signs, current_obj)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try single flips
            best_flip_idx = -1
            best_flip_obj = current_obj
            for i in range(num_edges):
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < best_flip_obj - 1e-12:
                        best_flip_obj = obj
                        best_flip_idx = i
                current_signs[i] = old
                
            if best_flip_idx != -1:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_obj = best_flip_obj
                update_best(current_signs, current_obj)
                improved = True
            else:
                # 2. Try 4-cycle flips
                best_cycle_idx = -1
                best_cycle_obj = current_obj
                for c_idx, cyc in enumerate(cycles_4):
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < best_cycle_obj - 1e-12:
                        best_cycle_obj = obj
                        best_cycle_idx = c_idx
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                        
                if best_cycle_idx != -1:
                    for e in cycles_4[best_cycle_idx]:
                        current_signs[e] = -current_signs[e]
                    current_obj = best_cycle_obj
                    update_best(current_signs, current_obj)
                    improved = True
                else:
                    # 3. Systematic pair flips
                    best_pair = None
                    best_pair_obj = current_obj
                    for e1, e2 in all_pairs:
                        if time.time() > time_limit:
                            break
                        current_signs[e1] = -current_signs[e1]
                        current_signs[e2] = -current_signs[e2]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < best_pair_obj - 1e-12:
                                best_pair_obj = obj
                                best_pair = (e1, e2)
                        current_signs[e1] = -current_signs[e1]
                        current_signs[e2] = -current_signs[e2]
                        
                    if best_pair is not None:
                        e1, e2 = best_pair
                        current_signs[e1] = -current_signs[e1]
                        current_signs[e2] = -current_signs[e2]
                        current_obj = best_pair_obj
                        update_best(current_signs, current_obj)
                        improved = True
                    else:
                        # 4. Triple flips
                        found_triple = False
                        for _ in range(30):
                            if time.time() > time_limit:
                                break
                            e1 = rng.integers(num_edges)
                            e2 = rng.integers(num_edges)
                            e3 = rng.integers(num_edges)
                            if e1 == e2 or e2 == e3 or e1 == e3:
                                continue
                            current_signs[e1] = -current_signs[e1]
                            current_signs[e2] = -current_signs[e2]
                            current_signs[e3] = -current_signs[e3]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-12:
                                    current_obj = obj
                                    update_best(current_signs, current_obj)
                                    found_triple = True
                                    improved = True
                                    break
                            current_signs[e1] = -current_signs[e1]
                            current_signs[e2] = -current_signs[e2]
                            current_signs[e3] = -current_signs[e3]
                        
                        if not found_triple:
                            break

    def polish(signs):
        nonlocal best_signs, best_obj
        current_signs = signs.copy()
        current_obj = get_objective(current_signs)
        if not is_connected(current_signs):
            return
        
        update_best(current_signs, current_obj)
        
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-14:
                        current_obj = obj
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old
            
            if not improved:
                # Systematic pair flip sweep
                for e1, e2 in all_pairs:
                    if time.time() > deadline - 2.0:
                        break
                    current_signs[e1] = -current_signs[e1]
                    current_signs[e2] = -current_signs[e2]
                    if is_connected(current_signs):
                        obj = get_objective(current_signs)
                        if obj < current_obj - 1e-14:
                            current_obj = obj
                            improved = True
                            break
                    current_signs[e1] = -current_signs[e1]
                    current_signs[e2] = -current_signs[e2]
            
            if not improved:
                break
        
        update_best(current_signs, current_obj)

    # SDP-inspired initialization
    def sdp_init():
        signs = np.ones(num_edges, dtype=np.int8)
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            A[u, v] = 1
            A[v, u] = 1
        
        ev, evec = np.linalg.eigh(A)
        v_max = evec[:, -1]
        
        for i, (u, v) in enumerate(edges):
            prod = v_max[u] * v_max[v]
            signs[i] = 1 if prod >= 0 else -1
        
        return signs

    # Multiple Restarts
    # 1. All 1s
    signs = np.ones(num_edges, dtype=np.int8)
    if is_connected(signs):
        local_search(signs, deadline - 35.0)
    
    # 2. SDP-inspired initialization
    signs = sdp_init()
    if is_connected(signs):
        local_search(signs, deadline - 30.0)
    
    # 3. Structured initializations
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 25.0)
    
    # 4. Alternating signs on bottom cycle
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u >= m_val and v >= m_val:
            signs[i] = 1 if ((u - m_val) % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 20.0)
    
    # 5. Random restarts
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 5.0)

    # Final polish
    if best_signs is not None:
        polish(best_signs)
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        
    return best_signs
