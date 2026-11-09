import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

def construct():
    n = 256
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)

    # Build incidence matrix A: rows = APs, cols = positions
    # A[i, j] = 1 if position j is in AP i
    row_idx = []
    col_idx = []
    for i, (a, b, c, d) in enumerate(aps):
        for pos in (a, b, c, d):
            row_idx.append(i)
            col_idx.append(pos)
    A = coo_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(num_aps, n)).tocsr()

    # Thue-Morse initialization
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono():
        """Count monochromatic 4-APs."""
        count = 0
        for i in range(num_aps):
            a, b, c, d = aps[i]
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count

    def delta_flip(x):
        """Calculate change in cost if we flip position x."""
        delta = 0
        for i in range(num_aps):
            a, b, c, d = aps[i]
            if x not in (a, b, c, d):
                continue
            # Check if this AP is currently monochromatic
            vals = [colors[a], colors[b], colors[c], colors[d]]
            is_mono = len(set(vals)) == 1
            # After flip
            # Simpler: just recompute
            new_vals = [colors[a], colors[b], colors[c], colors[d]]
            idx = (a, b, c, d).index(x)
            new_vals[idx] = 1 - new_vals[idx]
            new_is_mono = len(set(new_vals)) == 1
            if is_mono and not new_is_mono:
                delta -= 1
            elif not is_mono and new_is_mono:
                delta += 1
        return delta

    current_cost = count_mono()
    best_cost = current_cost
    best_colors = colors.copy()

    # Spectral method: project onto subspace orthogonal to top singular vectors of A
    # We work with a real-valued representation: map colors to {-1, +1}
    def to_real(colors):
        return 2 * colors - 1  # 0 -> -1, 1 -> +1

    def from_real(v):
        return (v > 0).astype(np.int8)

    def count_mono_real(v):
        """Count mono APs given real-valued vector v in {-1,+1}."""
        count = 0
        for i in range(num_aps):
            a, b, c, d = aps[i]
            if v[a] == v[b] == v[c] == v[d]:
                count += 1
        return count

    # Start with Thue-Morse in real form
    v = to_real(colors).astype(np.float64)

    # Get top singular vectors of A
    # A is (num_aps x n). We want to suppress the modes that contribute most to monochromatic APs.
    # The objective is sum over APs of indicator(all four same sign).
    # Approximate: minimize sum_i (v[a]*v[b]*v[c]*v[d] + 3)/4 for each AP
    # This is a degree-4 polynomial. Spectral method: project v away from top singular vectors of A.
    
    # Compute a few top singular vectors of A
    try:
        k_sv = min(8, min(num_aps, n) - 1)
        U, s, Vt = svds(A, k=k_sv, return_singular_vectors=True)
        # Vt is (k_sv x n), so V = Vt.T is (n x k_sv)
        V = Vt.T
    except Exception:
        V = None

    # Iterative projection: v_new = v - V * (V.T @ v)
    # Then re-binarize
    if V is not None:
        for proj_iter in range(20):
            if time.time() > deadline:
                break
            # Project
            coeff = V.T @ v
            v_proj = v - V @ coeff
            # Re-normalize to unit length or keep magnitude
            v = v_proj
            # Re-binarize
            v_bin = (v > 0).astype(np.float64) * 2 - 1
            cost_bin = count_mono_real(v_bin)
            if cost_bin < best_cost:
                best_cost = cost_bin
                best_colors = from_real(v_bin)
            v = v_bin

    # Now do local search (simulated annealing style) on the best coloring
    current_cost = count_mono()
    best_cost = min(best_cost, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_colors = colors.copy()
    else:
        colors = best_colors.copy()
        current_cost = best_cost

    # Simulated annealing with vectorized delta
    # Precompute for each position, which APs it belongs to
    pos_aps = [[] for _ in range(n)]
    for i in range(num_aps):
        for pos in aps[i]:
            pos_aps[pos].append(i)

    def fast_delta_flip(x):
        """Fast delta computation using precomputed AP lists."""
        delta = 0
        for i in pos_aps[x]:
            a, b, c, d = aps[i]
            vals = (colors[a], colors[b], colors[c], colors[d])
            is_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            # Flip x
            new_vals = list(vals)
            idx = (a, b, c, d).index(x)
            new_vals[idx] = 1 - new_vals[idx]
            new_is_mono = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
            if is_mono and not new_is_mono:
                delta -= 1
            elif not is_mono and new_is_mono:
                delta += 1
        return delta

    # Simulated annealing
    T_start = 2.0
    T_end = 0.01
    n_steps = 50000
    for step in range(n_steps):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        x = np.random.randint(n)
        d = fast_delta_flip(x)
        if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-10)):
            colors[x] = 1 - colors[x]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()

    # Final polish: deterministic local search
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for x in range(n):
            d = fast_delta_flip(x)
            if d < 0:
                colors[x] = 1 - colors[x]
                current_cost += d
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        if not improved:
            break

    return [int(c) for c in best_colors]
