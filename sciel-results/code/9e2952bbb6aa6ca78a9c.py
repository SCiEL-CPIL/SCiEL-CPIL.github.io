import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m_val = n // 2

    # Precompute edge indices
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Precompute edge arrays for fast objective
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)

    def build_S(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[eu, ev] = signs
        S[ev, eu] = signs
        return S

    def get_objective(signs):
        S = build_S(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(signs):
        # Union-find on 2n lift vertices
        parent = list(range(2 * n))
        rank = [0] * (2 * n)

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
            u0, u1 = 2 * u, 2 * u + 1
            v0, v1 = 2 * v, 2 * v + 1
            if s == 1:
                union(u0, v0)
                union(u1, v1)
            else:
                union(u0, v1)
                union(u1, v0)

        root = find(0)
        return all(find(i) == root for i in range(2 * n))

    # Precompute 4-cycles (side faces of prism)
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

    # Also precompute top cycle and bottom cycle edge lists
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            top_cycle_edges.append(i)
        elif u >= m_val and v >= m_val:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)

    rng = np.random.default_rng(3)

    best_signs = None
    best_obj = float('inf')

    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    def local_search(initial_signs, time_limit=None):
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline
        current_signs = initial_signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        while time.time() < time_limit:
            improved = False

            # 1. Single flips
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
                # 2. 4-cycle flips (preserve connectivity)
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
                    # 3. Top/bottom cycle flips
                    found = False
                    for cyc in [top_cycle_edges, bottom_cycle_edges]:
                        if len(cyc) == 0:
                            continue
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-12:
                                current_obj = obj
                                update_best(current_signs, current_obj)
                                found = True
                                improved = True
                                break
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                    if found:
                        continue

                    # 4. Pair flips (systematic, limited)
                    best_pair = None
                    best_pair_obj = current_obj
                    for a in range(num_edges):
                        if time.time() > time_limit:
                            break
                        for b in range(a + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < best_pair_obj - 1e-12:
                                    best_pair_obj = obj
                                    best_pair = (a, b)
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]

                    if best_pair is not None:
                        a, b = best_pair
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        current_obj = best_pair_obj
                        update_best(current_signs, current_obj)
                        improved = True
                    else:
                        # 5. Random triple flips
                        found_triple = False
                        for _ in range(20):
                            if time.time() > time_limit:
                                break
                            idxs = rng.choice(num_edges, size=3, replace=False)
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-12:
                                    current_obj = obj
                                    update_best(current_signs, current_obj)
                                    found_triple = True
                                    improved = True
                                    break
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                        if not found_triple:
                            break

    def polish(signs):
        nonlocal best_signs, best_obj
        current_signs = signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        for _ in range(60):
            if time.time() > deadline - 1.0:
                break
            improved = False
            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 1.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old
            if not improved:
                # 4-cycle sweep
                for cyc in cycles_4:
                    if time.time() > deadline - 1.0:
                        break
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
            if not improved:
                # Pair sweep
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-15:
                                current_obj = obj
                                improved = True
                                break
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                    if improved:
                        break
            if not improved:
                # Triple flip sweep (limited)
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        if time.time() > deadline - 1.0:
                            break
                        for c in range(b + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-15:
                                    current_obj = obj
                                    improved = True
                                    break
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        update_best(current_signs, current_obj)

    # ====== Phase-space continuous optimization with soft-max ======
    def phase_optimize():
        """
        Continuous optimization over phases. Uses a smooth soft-max surrogate
        for the spectral radius to get better gradients, then rounds.
        """
        from scipy.optimize import minimize

        def soft_objective(x, temperature=5.0):
            """Smooth approximation of max|eigenvalue| using log-sum-exp."""
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev = np.linalg.eigvalsh(S)
            abs_ev = np.abs(ev)
            # Soft max: (1/t) * log(sum(exp(t * abs_ev)))
            max_val = np.max(abs_ev)
            shifted = t * (abs_ev - max_val)
            log_sum = np.log(np.sum(np.exp(shifted))) / t + max_val
            return float(log_sum)

        def soft_gradient(x, temperature=5.0):
            """Gradient of soft-max spectral radius."""
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev, evec = np.linalg.eigh(S)
            abs_ev = np.abs(ev)
            max_val = np.max(abs_ev)
            shifted = temperature * (abs_ev - max_val)
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            # Gradient of soft-max: sum_k weights_k * sign(ev_k) * (v_k v_k^T)
            # d/dx_i = sum_k weights_k * sign(ev_k) * (v_k[eu_i]*v_k[ev_i] + v_k[ev_i]*v_k[eu_i])
            grad = np.zeros(num_edges, dtype=np.float64)
            for k in range(n):
                if weights[k] > 1e-10:
                    sign_k = np.sign(ev[k])
                    v_k = evec[:, k]
                    grad += weights[k] * sign_k * 2.0 * (v_k[eu] * v_k[ev])
            return grad

        def hard_objective(x):
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev = np.linalg.eigvalsh(S)
            return float(np.max(np.abs(ev)))

        # Multiple restarts in continuous space
        best_x = None
        best_f = float('inf')

        inits = [
            np.ones(num_edges),
            -np.ones(num_edges),
            rng.uniform(-1, 1, size=num_edges),
            rng.uniform(-1, 1, size=num_edges),
            rng.uniform(-1, 1, size=num_edges),
        ]
        # Structured: top cycle alternating
        x_struct = np.ones(num_edges)
        for i, (u, v) in enumerate(edges):
            if u < m_val and v < m_val:
                x_struct[i] = 1.0 if (u % 2 == 0) else -1.0
            elif u >= m_val and v >= m_val:
                x_struct[i] = 1.0 if ((u - m_val) % 2 == 0) else -1.0
        inits.append(x_struct)

        # Structured: matching alternating
        x_struct2 = np.ones(num_edges)
        for i, (u, v) in enumerate(edges):
            if not (u < m_val and v < m_val) and not (u >= m_val and v >= m_val):
                x_struct2[i] = 1.0 if (min(u, v) % 2 == 0) else -1.0
        inits.append(x_struct2)

        for x0 in inits:
            if time.time() > deadline - 20.0:
                break
            try:
                # First pass: soft objective for smooth gradients
                res = minimize(
                    soft_objective, x0, jac=soft_gradient,
                    method='L-BFGS-B',
                    bounds=[(-1, 1)] * num_edges,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_soft = res.x.copy()
                
                # Second pass: hard objective from soft result
                res2 = minimize(
                    hard_objective, x_soft,
                    method='L-BFGS-B',
                    bounds=[(-1, 1)] * num_edges,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                f = hard_objective(res2.x)
                if f < best_f:
                    best_f = f
                    best_x = res2.x.copy()
            except Exception:
                pass

        if best_x is not None:
            # Round to signs
            signs = np.where(best_x >= 0, 1, -1).astype(np.int8)
            return signs, best_f

        return None, float('inf')

    # ====== Exhaustive search for small m ======
    def exhaustive_search():
        """Exhaustive search over all 2^E sign assignments. Only feasible for small E."""
        if num_edges > 16:
            return None, float('inf')
        best = None
        best_f = float('inf')
        for bits in product([1, -1], repeat=num_edges):
            signs = np.array(bits, dtype=np.int8)
            if not is_connected(signs):
                continue
            f = get_objective(signs)
            if f < best_f:
                best_f = f
                best = signs.copy()
        return best, best_f

    # ====== Main search ======

    # 1. Exhaustive search for m=4 (num_edges = 3*4 = 12, 2^12 = 4096 - feasible)
    if m_val == 4:
        ex_signs, ex_f = exhaustive_search()
        if ex_signs is not None:
            update_best(ex_signs, ex_f)

    # 2. Phase-space optimization
    phase_signs, phase_f = phase_optimize()
    if phase_signs is not None:
        if is_connected(phase_signs):
            local_search(phase_signs, deadline - 25.0)
            polish(phase_signs)

    # 3. Standard local search from multiple initializations
    # All 1s
    signs = np.ones(num_edges, dtype=np.int8)
    if is_connected(signs):
        local_search(signs, deadline - 20.0)

    # Alternating top
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 15.0)

    # Alternating bottom
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u >= m_val and v >= m_val:
            signs[i] = 1 if ((u - m_val) % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 12.0)

    # Alternating matching
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if not (u < m_val and v < m_val) and not (u >= m_val and v >= m_val):
            signs[i] = 1 if (min(u, v) % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 10.0)

    # Random restarts
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 5.0)

    # Final polish
    if best_signs is not None:
        polish(best_signs)

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    # Verify final result is valid
    if not is_connected(best_signs):
        fallback = np.ones(num_edges, dtype=np.int8)
        if is_connected(fallback):
            best_signs = fallback
        else:
            for pattern in range(4):
                test = np.ones(num_edges, dtype=np.int8)
                for i, (u, v) in enumerate(edges):
                    if (u // m_val) == (v // m_val):
                        if (u % 2) != (pattern % 2):
                            test[i] = -1
                if is_connected(test):
                    best_signs = test
                    break

    return best_signs
