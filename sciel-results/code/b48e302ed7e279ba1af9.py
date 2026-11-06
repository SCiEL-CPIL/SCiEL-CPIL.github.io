import math
import time
import numpy as np

def design(problem):
    """Design a symmetric 32x32-pixel unit cell for maximum TM 4-5 band gap.

    Strategy: Use a symmetric 16x16 quadrant representation to enforce C4v symmetry.
    Start from square rods, refine parameters, then perform pixel-level hill climbing
    on the symmetric quadrant.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    start_time = time.time()
    deadline = start_time + 105  # safety margin well under the 120s budget

    N = size
    Q = N // 2  # 16

    # ------------------------------------------------------------------
    # Fast exact plane-wave TM band-gap evaluator
    # ------------------------------------------------------------------
    npw = 8

    mx = np.arange(-npw, npw + 1)
    my = np.arange(-npw, npw + 1)
    GX, GY = np.meshgrid(mx, my, indexing='ij')
    GX = GX.ravel()
    GY = GY.ravel()
    nG = len(GX)

    G_dict = {}
    for idx in range(nG):
        G_dict[(GX[idx], GY[idx])] = idx

    D = np.full((nG, nG), -1, dtype=int)
    for i in range(nG):
        for j in range(nG):
            key = (GX[i] - GX[j], GY[i] - GY[j])
            if key in G_dict:
                D[i, j] = G_dict[key]

    x = (np.arange(N) + 0.5) / N
    y = (np.arange(N) + 0.5) / N

    phase2d = -1j * (GX[:, None, None] * x[None, None, :] +
                     GY[:, None, None] * y[None, :, None])
    exp2d = np.exp(phase2d)  # (nG, N, N)

    k_points = [
        (0.0, 0.0),
        (math.pi, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi / 2, 0.0),
        (math.pi / 4, math.pi / 4),
        (math.pi / 2, math.pi / 4),
        (3 * math.pi / 8, math.pi / 8),
        (math.pi / 3, math.pi / 6),
        (2 * math.pi / 5, math.pi / 5),
        (math.pi / 2, math.pi / 6),
        (math.pi / 4, 0.0),
        (3 * math.pi / 4, 0.0),
        (math.pi / 6, 0.0),
        (math.pi / 2, math.pi / 8),
        (5 * math.pi / 8, math.pi / 8),
        (math.pi / 5, math.pi / 5),
        (math.pi / 2, 3 * math.pi / 8),
        (math.pi / 12, math.pi / 12),
        (math.pi / 3, 0.0),
        (math.pi / 2, math.pi / 3),
    ]

    def compute_gap(cell_2d):
        """Compute the gap-midgap ratio between TM bands 4 and 5."""
        eps = np.where(np.asarray(cell_2d, dtype=int) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps

        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp2d, axis=(1, 2)) / (N * N)

        max_omega4 = 0.0
        min_omega5 = np.inf

        for kx, ky in k_points:
            kG_sq = (kx + 2 * math.pi * GX) ** 2 + (ky + 2 * math.pi * GY) ** 2

            A = inv_eps_coeffs[np.maximum(D, 0)]
            A = np.where(D >= 0, A, 0.0)
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2.0

            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)

            if len(eigvals) >= 6:
                o4 = eigvals[3]
                o5 = eigvals[4]
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5

        if min_omega5 == np.inf or max_omega4 <= 0:
            return 0.0
        midgap = 0.5 * (min_omega5 + max_omega4)
        if midgap <= 0:
            return 0.0
        return (min_omega5 - max_omega4) / midgap

    # ------------------------------------------------------------------
    # Symmetric Quadrant Representation
    # ------------------------------------------------------------------
    def q_to_cell(q):
        """Expand a symmetric 16x16 quadrant to a 32x32 C4v symmetric cell."""
        q = np.asarray(q, dtype=int)
        # Ensure symmetry of q itself (just in case, though we maintain it)
        # q is assumed symmetric: q[i,j] == q[j,i]
        
        # Top-left: q
        # Top-right: q[:, ::-1] (mirror horizontally)
        # Bottom-left: q[::-1, :] (mirror vertically)
        # Bottom-right: q[::-1, ::-1] (rotate 180)
        
        # Wait, standard C4v symmetry about center for even N:
        # C[i,j] = q[i,j] for i,j < 16
        # C[i,j] = q[i, 31-j] for i < 16, j >= 16
        # C[i,j] = q[31-i, j] for i >= 16, j < 16
        # C[i,j] = q[31-i, 31-j] for i >= 16, j >= 16
        
        # Let's verify if this requires q to be symmetric.
        # If we use the above mapping, C is C4v symmetric IF q is symmetric.
        # Proof:
        # C[i,j] for i,j<16 is q[i,j].
        # Rotated 90 deg: (j, 31-i).
        # If j<16 and 31-i >= 16 (since i<16), then C[j, 31-i] = q[j, 31-(31-i)] = q[j,i].
        # For C to be invariant, q[i,j] must equal q[j,i].
        
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])

    def make_quadrant(rod_size, offset=0):
        """Create a symmetric 16x16 quadrant with a square rod centered at (7.5, 7.5)."""
        q = np.zeros((Q, Q), dtype=int)
        center = 7.5 + offset
        half = rod_size / 2.0
        
        r_min = int(round(center - half))
        r_max = int(round(center + half))
        c_min = int(round(center - half))
        c_max = int(round(center + half))
        
        # Clamp to quadrant bounds [0, 15]
        r_min = max(0, min(Q-1, r_min))
        r_max = max(0, min(Q-1, r_max))
        c_min = max(0, min(Q-1, c_min))
        c_max = max(0, min(Q-1, c_max))
        
        if r_max >= r_min and c_max >= c_min:
            q[r_min:r_max+1, c_min:c_max+1] = 1
            
        # Enforce symmetry q[i,j] = q[j,i]
        # Since the rod is a square centered on the diagonal, it is naturally symmetric.
        # But if offset or size creates asymmetry due to rounding, we force it.
        # Actually, a square defined by integer bounds [r_min, r_max] x [c_min, c_max]
        # where r_min=c_min and r_max=c_max is symmetric.
        # Our construction ensures r_min=c_min and r_max=c_max.
        # So q is symmetric.
        
        return q

    # Initial best
    best_q = make_quadrant(8)
    best_cell = q_to_cell(best_q)
    best_gap = compute_gap(best_cell)

    # ------------------------------------------------------------------
    # Local search over rod size and position offset
    # ------------------------------------------------------------------
    rod_sizes = [6, 7, 8, 9, 10, 11, 12]
    offsets = [-1, 0, 1]

    # First pass: scan the grid
    for s in rod_sizes:
        if time.time() > deadline:
            break
        for off in offsets:
            if time.time() > deadline:
                break
            q = make_quadrant(s, off)
            cell = q_to_cell(q)
            gap = compute_gap(cell)
            if gap > best_gap:
                best_gap = gap
                best_q = q.copy()
                best_cell = cell.copy()

    # Find best s, off for refinement
    best_s, best_off = 8, 0
    for s in rod_sizes:
        for off in offsets:
            if np.array_equal(make_quadrant(s, off), best_q):
                best_s, best_off = s, off
                break
        if np.array_equal(make_quadrant(s, off), best_q):
            break

    # Second pass: fine local search
    for _ in range(3):
        improved = False
        for ds in [-1, 1]:
            for do in [-1, 0, 1]:
                if time.time() > deadline:
                    break
                s2 = best_s + ds
                off2 = best_off + do
                if s2 < 4 or s2 > 14:
                    continue
                q = make_quadrant(s2, off2)
                cell = q_to_cell(q)
                gap = compute_gap(cell)
                if gap > best_gap + 1e-12:
                    best_gap = gap
                    best_q = q.copy()
                    best_cell = cell.copy()
                    best_s, best_off = s2, off2
                    improved = True
        if not improved:
            break

    # ------------------------------------------------------------------
    # Final pixel-level refinement: flip individual pixels in the symmetric
    # quadrant (only r <= c to maintain symmetry and avoid redundant work)
    # ------------------------------------------------------------------
    q = best_q.copy()
    
    # Precompute list of valid (r,c) pairs in the upper triangle of the quadrant
    # including the diagonal
    valid_pixels = [(r, c) for r in range(Q) for c in range(r, Q)]
    
    for _ in range(10):  # Increase rounds if time permits
        if time.time() > deadline:
            break
        best_flip = None
        best_flip_gain = 1e-12
        for r, c in valid_pixels:
            if time.time() > deadline:
                break
            q2 = q.copy()
            q2[r, c] = 1 - q2[r, c]
            # If we flip (r,c), we must also flip (c,r) if r != c to maintain symmetry
            # However, our representation 'q' is the full 16x16 matrix.
            # If we just flip q[r,c], q is no longer symmetric.
            # We must flip both q[r,c] and q[c,r].
            if r != c:
                q2[c, r] = 1 - q2[c, r]
                
            cell = q_to_cell(q2)
            gap = compute_gap(cell)
            gain = gap - best_gap
            if gain > best_flip_gain:
                best_flip_gain = gain
                best_flip = (r, c)
        if best_flip is not None:
            r, c = best_flip
            q[r, c] = 1 - q[r, c]
            if r != c:
                q[c, r] = 1 - q[c, r]
            best_q = q.copy()
            best_cell = q_to_cell(q)
            best_gap += best_flip_gain
        else:
            break

    # Return as nested list of 0/1
    return best_cell.tolist()
