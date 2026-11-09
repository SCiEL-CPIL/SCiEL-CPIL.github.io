import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 570.0  # safety margin

    # ---------- Base: all weight-2 vectors (220 points, known valid) ----------
    base_list = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_list.append(p)
    base_points = np.array(base_list, dtype=np.int64)
    best_points = base_points.copy()
    best_k = base_points.shape[0]  # 220

    # ---------- Candidate generation ----------
    def generate_candidates(weights):
        cands = []
        for w in weights:
            for combo in combinations(range(d), w):
                for mask in range(1 << w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        v[pos] = 1 if (mask >> idx) & 1 else -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    rng = random.Random(42)

    # ---------- Vectorized greedy selection ----------
    def greedy_once(cands, norms, order):
        selected_idx = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            p = cands[idx]
            pn = int(norms[idx])
            if not selected_idx:
                selected_idx.append(idx)
                continue
            sel = cands[selected_idx]
            sel_norms = norms[selected_idx]
            dots = sel @ p
            dist_sq = pn + sel_norms - 2 * dots
            required = max(int(sel_norms.max()), pn)
            if dist_sq.min() >= required:
                selected_idx.append(idx)
        return selected_idx

    # ---------- Local search: try to add points ----------
    def local_search(cands, norms, selected_idx, max_iters=30):
        sel = list(selected_idx)
        m = len(sel)
        n = len(cands)
        for _ in range(max_iters):
            if time.time() > deadline - 35:
                break
            sel_arr = cands[sel]
            sel_norms = norms[sel]
            max_sel_norm = int(sel_norms.max())
            
            dots = cands @ sel_arr.T
            dist_sq = norms[:, None] + sel_norms[None, :] - 2 * dots
            required = np.maximum(max_sel_norm, norms)
            conflicts = (dist_sq < required[:, None]).sum(axis=1)
            
            in_sel = np.zeros(n, dtype=bool)
            in_sel[sel] = True
            conflicts[in_sel] = 1
            
            free = np.where(conflicts == 0)[0]
            
            if len(free) > 0:
                free_norms = norms[free]
                pick = free[np.argmin(free_norms)]
                sel.append(int(pick))
                m += 1
            else:
                break
        return sel

    # ---------- Main search: start from base, try to add weight-3,4 vectors ----------
    # First, verify base is valid and use it as starting point
    base_norms = (base_points * base_points).sum(axis=1).astype(np.int64)
    
    # Try adding weight-3 vectors to the base
    w3_cands = generate_candidates([3])
    w3_norms = (w3_cands * w3_cands).sum(axis=1).astype(np.int64)
    
    # Combine base + weight-3 candidates
    all_cands = np.vstack([base_points, w3_cands])
    all_norms = np.concatenate([base_norms, w3_norms])
    n_total = all_cands.shape[0]
    
    # Start with all base points selected
    base_indices = list(range(220))
    
    # Try to add weight-3 points
    improved = local_search(all_cands, all_norms, base_indices, max_iters=20)
    if len(improved) > best_k:
        best_k = len(improved)
        best_points = all_cands[improved].copy()
    
    # Try different weight sets with greedy
    weight_sets = [
        [2, 3],
        [2, 3, 4],
    ]
    
    for ws in weight_sets:
        if time.time() > deadline - 60:
            break
        cands = generate_candidates(ws)
        n = cands.shape[0]
        norms = (cands * cands).sum(axis=1).astype(np.int64)
        
        max_restarts = 15 if n <= 2000 else 8
        for restart in range(max_restarts):
            if time.time() > deadline - 40:
                break
            if restart % 3 == 0:
                order = list(range(n))
                rng.shuffle(order)
            elif restart % 3 == 1:
                order = sorted(range(n), key=lambda i: (int(norms[i]), i))
            else:
                order = sorted(range(n), key=lambda i: (-int(norms[i]), i))

            selected_idx = greedy_once(cands, norms, order)
            improved = local_search(cands, norms, selected_idx, max_iters=30)
            if len(improved) > len(selected_idx):
                selected_idx = improved
            
            if len(selected_idx) > best_k:
                best_k = len(selected_idx)
                best_points = cands[selected_idx].copy()

    # ---------- Final safety ----------
    if best_points is None or best_points.shape[0] < 2:
        best_points = base_points
        best_k = base_points.shape[0]

    return np.array(best_points, dtype=np.int64)
