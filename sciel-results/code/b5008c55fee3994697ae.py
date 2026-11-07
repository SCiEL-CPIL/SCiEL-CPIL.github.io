"""Program for spectral graph lifts on prism graphs exploiting block structure."""
import numpy as np
from collections import deque
import time


def _is_connected_lift_fast(m, top_signs, bottom_signs, rung_signs):
    """Check connectivity of 2-lift using BFS on base graph with sign propagation.
    
    top_signs[i]: sign of edge (i, i+1 mod m) in top cycle
    bottom_signs[i]: sign of edge (m+i, m+i+1 mod m) in bottom cycle
    rung_signs[i]: sign of edge (i, m+i)
    """
    n = 2 * m
    label = [-1] * n
    label[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        lu = label[u]
        
        if u < m:
            # Top vertex
            # Top cycle neighbors
            v1 = (u + 1) % m
            expected = lu if top_signs[u] > 0 else 1 - lu
            if label[v1] == -1:
                label[v1] = expected
                q.append(v1)
            elif label[v1] != expected:
                return True
            
            v2 = (u - 1) % m
            # Edge (v2, u) has sign top_signs[v2]
            expected2 = lu if top_signs[v2] > 0 else 1 - lu
            if label[v2] == -1:
                label[v2] = expected2
                q.append(v2)
            elif label[v2] != expected2:
                return True
            
            # Rung
            v3 = m + u
            expected3 = lu if rung_signs[u] > 0 else 1 - lu
            if label[v3] == -1:
                label[v3] = expected3
                q.append(v3)
            elif label[v3] != expected3:
                return True
        else:
            # Bottom vertex
            u_local = u - m
            # Bottom cycle neighbors
            v1 = m + (u_local + 1) % m
            expected = lu if bottom_signs[u_local] > 0 else 1 - lu
            if label[v1] == -1:
                label[v1] = expected
                q.append(v1)
            elif label[v1] != expected:
                return True
            
            v2 = m + (u_local - 1) % m
            expected2 = lu if bottom_signs[(u_local - 1) % m] > 0 else 1 - lu
            if label[v2] == -1:
                label[v2] = expected2
                q.append(v2)
            elif label[v2] != expected2:
                return True
            
            # Rung
            v3 = u_local
            expected3 = lu if rung_signs[u_local] > 0 else 1 - lu
            if label[v3] == -1:
                label[v3] = expected3
                q.append(v3)
            elif label[v3] != expected3:
                return True
    
    return False


def _build_prism_adj(m, top_signs, bottom_signs, rung_signs):
    """Build the 2m x 2m signed adjacency matrix for prism graph."""
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    
    # Top cycle
    for i in range(m):
        j = (i + 1) % m
        s = top_signs[i]
        A[i, j] = s
        A[j, i] = s
    
    # Bottom cycle
    for i in range(m):
        j = (i + 1) % m
        s = bottom_signs[i]
        A[m + i, m + j] = s
        A[m + j, m + i] = s
    
    # Rungs
    for i in range(m):
        s = rung_signs[i]
        A[i, m + i] = s
        A[m + i, i] = s
    
    return A


def _spectral_radius_prism(m, top_signs, bottom_signs, rung_signs):
    """Compute spectral radius of signed adjacency matrix."""
    A = _build_prism_adj(m, top_signs, bottom_signs, rung_signs)
    vals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(vals)))


def _eval_and_check(m, top_signs, bottom_signs, rung_signs):
    """Evaluate spectral radius and check connectivity. Returns (radius, is_valid)."""
    if not _is_connected_lift_fast(m, top_signs, bottom_signs, rung_signs):
        return np.inf, False
    radius = _spectral_radius_prism(m, top_signs, bottom_signs, rung_signs)
    return radius, True


def _local_search(m, top_signs, bottom_signs, rung_signs, max_iters=50000):
    """Single-flip local search to minimize spectral radius while maintaining connectivity."""
    best_top = top_signs.copy()
    best_bot = bottom_signs.copy()
    best_rung = rung_signs.copy()
    best_val, _ = _eval_and_check(m, best_top, best_bot, best_rung)
    
    if not np.isfinite(best_val):
        return best_top, best_bot, best_rung, best_val
    
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        
        # Try flipping each top edge
        for i in range(m):
            cand_top = best_top.copy()
            cand_top[i] = -cand_top[i]
            val, valid = _eval_and_check(m, cand_top, best_bot, best_rung)
            if valid and val < best_val - 1e-12:
                best_top = cand_top
                best_val = val
                improved = True
                break
        
        if improved:
            continue
        
        # Try flipping each bottom edge
        for i in range(m):
            cand_bot = best_bot.copy()
            cand_bot[i] = -cand_bot[i]
            val, valid = _eval_and_check(m, best_top, cand_bot, best_rung)
            if valid and val < best_val - 1e-12:
                best_bot = cand_bot
                best_val = val
                improved = True
                break
        
        if improved:
            continue
        
        # Try flipping each rung
        for i in range(m):
            cand_rung = best_rung.copy()
            cand_rung[i] = -cand_rung[i]
            val, valid = _eval_and_check(m, best_top, best_bot, cand_rung)
            if valid and val < best_val - 1e-12:
                best_rung = cand_rung
                best_val = val
                improved = True
                break
    
    return best_top, best_bot, best_rung, best_val


def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge, minimizing spectral radius
    of the signed adjacency matrix, subject to the 2-lift being connected.
    
    Uses structure-aware search with local optimization.
    """
    rng = np.random.default_rng(3)
    start_time = time.time()
    n_edges = len(graph['edges'])
    m = graph['n'] // 2
    
    # Extract edge structure: map edge indices to (type, position)
    # type: 0=top, 1=bottom, 2=rung
    edge_map = []  # (edge_idx, type, pos)
    
    for idx, (u, v) in enumerate(graph['edges']):
        if u < m and v < m:
            # Top cycle
            if v == (u + 1) % m:
                pos = u
            else:
                pos = m - 1
            edge_map.append((idx, 0, pos))
        elif u >= m and v >= m:
            # Bottom cycle
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = m - 1
            edge_map.append((idx, 1, pos))
        else:
            # Rung
            i = min(u, v)
            edge_map.append((idx, 2, i))
    
    def make_signs(top, bot, rung):
        """Create full sign array from component signs."""
        signs = np.zeros(n_edges, dtype=int)
        for idx, etype, pos in edge_map:
            if etype == 0:
                signs[idx] = top[pos]
            elif etype == 1:
                signs[idx] = bot[pos]
            else:
                signs[idx] = rung[pos]
        return signs
    
    def check_valid_full(signs):
        """Check validity using full sign array."""
        top = np.zeros(m, dtype=int)
        bot = np.zeros(m, dtype=int)
        rung = np.zeros(m, dtype=int)
        for idx, etype, pos in edge_map:
            if etype == 0:
                top[pos] = signs[idx]
            elif etype == 1:
                bot[pos] = signs[idx]
            else:
                rung[pos] = signs[idx]
        return _is_connected_lift_fast(m, top, bot, rung)
    
    best_signs = None
    best_val = np.inf
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Alternating pattern on cycles, all +1 on rungs
    top1 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    bot1 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    rung1 = np.ones(m, dtype=int)
    candidates.append((top1, bot1, rung1))
    
    # Candidate 2: All +1 except one -1 rung for connectivity
    top2 = np.ones(m, dtype=int)
    bot2 = np.ones(m, dtype=int)
    rung2 = np.ones(m, dtype=int)
    rung2[0] = -1
    candidates.append((top2, bot2, rung2))
    
    # Candidate 3: Alternating rungs
    top3 = np.ones(m, dtype=int)
    bot3 = np.ones(m, dtype=int)
    rung3 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    candidates.append((top3, bot3, rung3))
    
    # Candidate 4: Mixed alternating
    top4 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    bot4 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    rung4 = np.array([1 if i % 2 == 0 else -1 for i in range(m)])
    candidates.append((top4, bot4, rung4))
    
    # Add random candidates
    for _ in range(30):
        tr = rng.choice([-1, 1], size=m).astype(int)
        br = rng.choice([-1, 1], size=m).astype(int)
        rr = rng.choice([-1, 1], size=m).astype(int)
        candidates.append((tr, br, rr))
    
    # Evaluate and locally optimize each candidate
    for top_c, bot_c, rung_c in candidates:
        if time.time() - start_time > 50:
            break
        
        val, valid = _eval_and_check(m, top_c, bot_c, rung_c)
        
        if not valid:
            # Try to fix connectivity by flipping edges
            fixed = False
            # Try flipping single edges
            for i in range(m):
                t2 = top_c.copy(); t2[i] = -t2[i]
                v2, val2 = _eval_and_check(m, t2, bot_c, rung_c)
                if val2:
                    top_c = t2
                    fixed = True
                    break
            if not fixed:
                for i in range(m):
                    b2 = bot_c.copy(); b2[i] = -b2[i]
                    v2, val2 = _eval_and_check(m, top_c, b2, rung_c)
                    if val2:
                        bot_c = b2
                        fixed = True
                        break
            if not fixed:
                for i in range(m):
                    r2 = rung_c.copy(); r2[i] = -r2[i]
                    v2, val2 = _eval_and_check(m, top_c, bot_c, r2)
                    if val2:
                        rung_c = r2
                        fixed = True
                        break
            if not fixed:
                continue
        
        # Local search
        top_opt, bot_opt, rung_opt, val_opt = _local_search(
            m, top_c, bot_c, rung_c, max_iters=10000
        )
        
        if val_opt < best_val:
            best_val = val_opt
            best_signs = make_signs(top_opt, bot_opt, rung_opt)
    
    # Additional random restarts if time permits
    if best_signs is not None and time.time() - start_time < 40:
        for restart in range(50):
            if time.time() - start_time > 50:
                break
            
            tr = rng.choice([-1, 1], size=m).astype(int)
            br = rng.choice([-1, 1], size=m).astype(int)
            rr = rng.choice([-1, 1], size=m).astype(int)
            
            val, valid = _eval_and_check(m, tr, br, rr)
            if not valid:
                # Quick fix
                fixed = False
                for i in range(m):
                    r2 = rr.copy(); r2[i] = -r2[i]
                    v2, val2 = _eval_and_check(m, tr, br, r2)
                    if val2:
                        rr = r2
                        fixed = True
                        break
                if not fixed:
                    for i in range(m):
                        t2 = tr.copy(); t2[i] = -t2[i]
                        v2, val2 = _eval_and_check(m, t2, br, rr)
                        if val2:
                            tr = t2
                            fixed = True
                            break
                if not fixed:
                    continue
            
            top_opt, bot_opt, rung_opt, val_opt = _local_search(
                m, tr, br, rr, max_iters=5000
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_signs = make_signs(top_opt, bot_opt, rung_opt)
    
    if best_signs is None:
        # Fallback: all +1 except one -1
        signs = np.ones(n_edges, dtype=int)
        # Find a rung edge and flip it
        for idx, etype, pos in edge_map:
            if etype == 2:
                signs[idx] = -1
                break
        best_signs = signs
    
    # Final validation
    if not check_valid_full(best_signs):
        # Emergency fix
        for i in range(n_edges):
            cand = best_signs.copy()
            cand[i] = -cand[i]
            if check_valid_full(cand):
                best_signs = cand
                break
    
    return best_signs.astype(int)
