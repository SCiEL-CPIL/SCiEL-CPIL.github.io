"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    time_limit = 300.0  # Safety margin (evaluator kills at 330s)
    
    N = len(prizes)
    
    # Precompute d_j0 = distance from each node to node 0
    d_j0 = distance_matrix[:, 0].copy()
    
    # Vectorized ACO simulation for speed
    def run_aco(eta, seed):
        np.random.seed(seed)
        tau = np.ones((N, N))
        
        best_prize_global = 0.0
        
        for it in range(30):
            best_prize_iter = 0.0
            
            for ant in range(20):
                current = 0
                path_len = 0.0
                prize_total = 0.0
                visited = np.zeros(N, dtype=bool)
                visited[0] = True
                
                while True:
                    # Find eligible nodes: unvisited and within budget
                    remaining = max_len - path_len
                    # d[i,j] + d[j,0] <= remaining
                    costs = distance_matrix[current, :] + d_j0
                    eligible_mask = (~visited) & (costs <= remaining)
                    
                    if not np.any(eligible_mask):
                        break
                    
                    eligible_idx = np.where(eligible_mask)[0]
                    weights = tau[current, eligible_idx] * eta[current, eligible_idx]
                    total_w = weights.sum()
                    if total_w <= 0:
                        break
                    
                    probs = weights / total_w
                    r = np.random.rand()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(eligible_idx):
                        idx = len(eligible_idx) - 1
                    
                    next_node = eligible_idx[idx]
                    path_len += distance_matrix[current, next_node]
                    prize_total += prizes[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if prize_total > best_prize_iter:
                    best_prize_iter = prize_total
            
            if best_prize_iter > best_prize_global:
                best_prize_global = best_prize_iter
            
            tau = tau * 0.9
            tau += best_prize_iter / 10.0
        
        return best_prize_global

    # Heuristic generators
    def gen_heuristic_A(lam):
        denom = distance_matrix + lam * d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_B(lam):
        denom = distance_matrix * (1.0 + lam * d_j0[None, :])
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_C(lam):
        # (Prize_j)^lam / (d_ij + d_j0)
        denom = distance_matrix + d_j0[None, :]
        denom = np.maximum(denom, 1e-9)
        prize_pow = np.power(np.maximum(prizes, 1e-9), lam)[None, :]
        eta = prize_pow / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_D(lam):
        # Prize_j / (d_ij^lam)
        denom = np.power(np.maximum(distance_matrix, 1e-9), lam)
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_E(lam):
        # (Prize_j) / (d_ij * d_j0^lam)
        denom = distance_matrix * np.power(np.maximum(d_j0, 1e-9)[None, :], lam)
        denom = np.maximum(denom, 1e-9)
        eta = prizes[None, :] / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_F(a, b):
        # Prize_j^a / (d_ij^b)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    def gen_heuristic_G(a, b, c):
        # Prize_j^a / (d_ij^b * d_j0^c)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) * np.power(np.maximum(d_j0, 1e-9)[None, :], c)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Build candidate list: (type, param1, param2)
    candidates = []
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        candidates.append(('A', lam, 0))
        candidates.append(('B', lam, 0))
        candidates.append(('C', lam, 0))
        candidates.append(('D', lam, 0))
        candidates.append(('E', lam, 0))
    
    # Add F and G candidates
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 1.0, 1.5, 2.0]:
            candidates.append(('F', a, b))
    for a in [0.5, 1.0, 1.5]:
        for b in [0.5, 1.0, 1.5]:
            for c in [0.5, 1.0, 1.5]:
                candidates.append(('G', a, b, c) if False else ('G', a, b))  # Simplify: use 2 params for G too
                # Actually let's just use F and G with 2 params each for consistency

    # Let me redefine G to take 2 params (a, b) where b controls both d_ij and d_j0
    def gen_heuristic_G2(a, b):
        # Prize_j^a / (d_ij^b * d_j0^b)
        prize_pow = np.power(np.maximum(prizes, 1e-9), a)[None, :]
        denom = np.power(np.maximum(distance_matrix, 1e-9), b) * np.power(np.maximum(d_j0, 1e-9)[None, :], b)
        denom = np.maximum(denom, 1e-9)
        eta = prize_pow / denom
        eta = np.where(eta > 0, eta, 1e-9)
        return eta

    # Rebuild candidates properly
    candidates = []
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        candidates.append(('A', lam, 0))
        candidates.append(('B', lam, 0))
        candidates.append(('C', lam, 0))
        candidates.append(('D', lam, 0))
        candidates.append(('E', lam, 0))
    for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0]:
            candidates.append(('F', a, b))
            candidates.append(('G2', a, b))

    def get_eta(c_type, p1, p2):
        if c_type == 'A':
            return gen_heuristic_A(p1)
        elif c_type == 'B':
            return gen_heuristic_B(p1)
        elif c_type == 'C':
            return gen_heuristic_C(p1)
        elif c_type == 'D':
            return gen_heuristic_D(p1)
        elif c_type == 'E':
            return gen_heuristic_E(p1)
        elif c_type == 'F':
            return gen_heuristic_F(p1, p2)
        elif c_type == 'G2':
            return gen_heuristic_G2(p1, p2)
        else:
            return gen_heuristic_A(0.0)

    # Phase 1: Quick screen with seed 0
    quick_results = []
    for c_type, p1, p2 in candidates:
        if time.time() - start_time > time_limit - 80:
            break
        eta = get_eta(c_type, p1, p2)
        s = run_aco(eta, 0)
        quick_results.append((s, c_type, p1, p2, eta))

    # Sort by quick score descending
    quick_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: Evaluate top candidates with multiple seeds
    top_k = min(10, len(quick_results))
    best_score = -1
    best_eta = quick_results[0][4] if quick_results else gen_heuristic_A(0.0)

    for s_quick, c_type, p1, p2, eta in quick_results[:top_k]:
        if time.time() - start_time > time_limit - 20:
            break
        scores = [s_quick]  # seed 0 already computed
        for seed in [1, 2, 3]:
            if time.time() - start_time > time_limit - 15:
                break
            s = run_aco(eta, seed)
            scores.append(s)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_eta = eta

    # Phase 3: Local refinement around best candidate
    if time.time() - start_time < time_limit - 25:
        best_entry = quick_results[0]
        best_type = best_entry[1]
        best_p1 = best_entry[2]
        best_p2 = best_entry[3]
        
        refinement_candidates = []
        
        # Neighbors in parameter space
        if best_type in ['A', 'B', 'C', 'D', 'E', 'F', 'G2']:
            for dp1 in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]:
                for dp2 in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]:
                    np1 = best_p1 + dp1
                    np2 = best_p2 + dp2
                    if np1 >= 0 and np2 >= 0:
                        refinement_candidates.append((best_type, np1, np2))
        
        # Also try other types at best params
        other_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G2']
        for ot in other_types:
            if ot != best_type:
                refinement_candidates.append((ot, best_p1, best_p2))
        
        for c_type, p1, p2 in refinement_candidates:
            if time.time() - start_time > time_limit - 15:
                break
            eta = get_eta(c_type, p1, p2)
            s = run_aco(eta, 0)
            if s > best_score:
                scores = [s]
                for seed in [1, 2]:
                    if time.time() - start_time > time_limit - 10:
                        break
                    s2 = run_aco(eta, seed)
                    scores.append(s2)
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_eta = eta

    # Final safety check
    if not np.all(np.isfinite(best_eta)) or best_eta.shape != distance_matrix.shape:
        best_eta = (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
        best_eta = np.where(best_eta > 0, best_eta, 1e-9)

    return best_eta

# EVOLVE-BLOCK-END
