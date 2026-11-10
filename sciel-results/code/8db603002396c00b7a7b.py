# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Nikodym set in F_49^2, F_49 = F_7[alpha], alpha^2 = 3.

    Strategy:
      1. Start from the baseline construction (Hermitian unital with y_a square
         excluded), which is known to be a valid Nikodym set.
      2. Greedily enlarge the complement S (shrink N) using a vectorized
         safety-margin heuristic: a point x can be added to S iff for every
         direction v, at least one of the other 48 points on the line
         {x + t*v : t in F_49} is already in S.
      3. Return N = F_49^2 \\ S as an (k, 2, 2) int64 array.
    """
    p = 7
    q = p * p  # 49
    total = q * q  # 2401

    # alpha^2 = 3 (smallest quadratic non-residue mod 7)
    k_const = 3

    deadline = time.time() + 580  # safety margin before 630s kill

    # ---------- flat-index helpers ----------
    # Point (xa, xb, ya, yb) -> flat index = xa * 343 + xb * 49 + ya * 7 + yb
    def to_flat(pt):
        """pt: shape (..., 4) array of (xa, xb, ya, yb) ints."""
        return (pt[..., 0] * 343 + pt[..., 1] * 49 + pt[..., 2] * 7 + pt[..., 3])

    def from_flat(idx):
        """idx: array of flat indices -> shape (..., 4) array."""
        idx = idx.astype(np.int64)
        xa = (idx // 343) % p
        xb = (idx // 49) % p
        ya = (idx // 7) % p
        yb = idx % p
        return np.stack([xa, xb, ya, yb], axis=-1)

    # Build flat index for every point
    all_flat = np.arange(total, dtype=np.int64)
    all_pts = from_flat(all_flat)  # shape (2401, 4)

    # ---------- baseline complement S ----------
    # S = { (xa, xb, ya, yb) : 2*xa + ya^2 - yb^2*k = 0 mod p  AND  ya^2 != ya mod p }
    # i.e., ya is NOT a quadratic residue (including 0? 0 is a residue: 0=0^2)
    # Squares mod 7: {0, 1, 2, 4}. Non-squares: {3, 5, 6}.
    # Baseline excludes points on the unital whose ya is a non-square.
    ya_arr = all_pts[:, 2]
    yb_arr = all_pts[:, 3]
    xa_arr = all_pts[:, 0]

    # in_unital: 2*xa + ya^2 - yb^2 * k_const == 0 mod 7
    in_unital = (2 * xa_arr + ya_arr * ya_arr - yb_arr * yb_arr * k_const) % p == 0

    # ya is a square mod 7?
    squares_mod7 = {0, 1, 2, 4}
    ya_is_square = np.array([ya in squares_mod7 for ya in ya_arr])

    # S = unital points with ya NOT a square
    S_mask = np.zeros(total, dtype=bool)
    S_mask[in_unital & ~ya_is_square] = True

    # ---------- directions ----------
    # v = (1, m) for m in F_49  ->  49 directions
    # v = (0, 1)                ->   1 direction
    # Total: 50 directions
    # Each direction v is a point in F_49^2: (vx, vy) where vx, vy in F_49.
    # v = (1, m): vx = (1, 0), vy = (ma, mb)
    # v = (0, 1): vx = (0, 1), vy = (0, 0)

    dirs = []  # list of (vxa, vxb, vya, vyb)
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 1, 0, 0))
    n_dirs = len(dirs)  # 50

    # ---------- precompute line mates ----------
    # For each point x (flat index) and each direction v,
    # compute the 48 other points on the line {x + t*v : t in F_49, t != 0}.
    # Store as line_mates[point_idx, dir_idx, j] = flat index of j-th mate.
    # Shape: (2401, 50, 48)

    # All non-zero t in F_49: 48 elements
    t_vals = []
    for ta in range(p):
        for tb in range(p):
            if ta == 0 and tb == 0:
                continue
            t_vals.append((ta, tb))
    t_arr = np.array(t_vals, dtype=np.int64)  # shape (48, 2)

    # For each direction v, compute t*v for all t (vectorized)
    # v as (vxa, vxb, vya, vyb)
    # t*v: (ta*vxa + tb*vxb*k, ta*vxb + tb*vxa, ta*vya + tb*vyb*k, ta*vyb + tb*vya) mod p

    line_mates = np.zeros((total, n_dirs, 48), dtype=np.int64)

    for di, v in enumerate(dirs):
        vxa, vxb, vya, vyb = v
        # t*v for all 48 t values
        ta = t_arr[:, 0]
        tb = t_arr[:, 1]
        tv_xa = (ta * vxa + tb * vxb * k_const) % p
        tv_xb = (ta * vxb + tb * vxa) % p
        tv_ya = (ta * vya + tb * vyb * k_const) % p
        tv_yb = (ta * vyb + tb * vya) % p
        # tv as flat offset: since addition is componentwise mod p,
        # the flat index of x + tv is NOT simply flat(x) + flat(tv) because
        # of the mixed-radix encoding. We need to compute it properly.
        # Actually, flat(x + tv) = (xa+tv_xa)*343 + (xb+tv_xb)*49 + (ya+tv_ya)*7 + (yb+tv_yb)
        # but with mod p on each component. Since flat is a bijection from
        # F_p^4 to {0,...,p^4-1} via mixed radix, and addition mod p in each
        # component corresponds to addition in the mixed-radix representation
        # ONLY if there's no carry. But there IS carry: e.g., xa=6, tv_xa=1 -> xa+tv_xa=7 -> 0 mod 7,
        # but in flat index 6*343 + 1*343 = 7*343 = 2401 = 0 mod 2401. So it works!
        # flat(x) + flat(tv) mod 2401 = flat(x + tv) because the mixed-radix
        # representation with modulus p in each digit makes addition mod p
        # equivalent to integer addition mod p^4.
        # Let's verify: flat(x) = xa*343 + xb*49 + ya*7 + yb
        # flat(tv) = tv_xa*343 + tv_xb*49 + tv_ya*7 + tv_yb
        # flat(x) + flat(tv) = (xa+tv_xa)*343 + (xb+tv_xb)*49 + (ya+tv_ya)*7 + (yb+tv_yb)
        # If xa+tv_xa >= 7, then (xa+tv_xa)*343 = xa*343 + tv_xa*343 + (xa+tv_xa-7)*343
        # and (xa+tv_xa-7)*343 = (xa+tv_xa-7)*p^3. Since we're working mod p^4 = 2401,
        # (xa+tv_xa-7)*343 mod 2401 = (xa+tv_xa-7)*343. But we want (xa+tv_xa mod 7)*343.
        # (xa+tv_xa mod 7)*343 = (xa+tv_xa)*343 - (xa+tv_xa-7)*343 if xa+tv_xa >= 7.
        # So flat(x) + flat(tv) mod 2401 = flat(x + tv) only if the carries work out.
        # Actually, since 343 = 7^3, and we're adding digits that can be up to 6+6=12,
        # the carry from the yb digit (max 6+6=12) goes into ya: 12 = 1*7 + 5, so carry 1.
        # This means flat(x) + flat(tv) mod 2401 != flat(x + tv) in general!
        #
        # So I need to compute flat(x + tv) properly. Let me just do it directly.

        # For each point x (all 2401), compute x + tv for each of the 48 tv values.
        # x: shape (2401, 4), tv: shape (48, 4)
        # x + tv: shape (2401, 48, 4) via broadcasting
        x_all = all_pts[:, np.newaxis, :]  # (2401, 1, 4)
        tv_all = np.stack([tv_xa, tv_xb, tv_ya, tv_yb], axis=-1)  # (48, 4)
        sum_pts = (x_all + tv_all[np.newaxis, :, :]) % p  # (2401, 48, 4)
        flat_mates = to_flat(sum_pts)  # (2401, 48)
        line_mates[:, di, :] = flat_mates

    # ---------- greedy refinement ----------
    # S_mask: shape (2401,) bool. True = in S (excluded from N).
    # For a candidate point x, it can be added to S iff for every direction v,
    # at least one of the 48 line-mates is already in S.

    # Precompute for each point x and direction v: the 48 flat indices of mates.
    # line_mates[x, v, :] gives those indices.

    # To check if x can be added:
    #   For each v: count how many of line_mates[x, v, :] are in S_mask.
    #   x can be added iff min over v of that count >= 1.

    # Vectorized check:
    #   S_mask_flat = S_mask (already a 1D bool array)
    #   For each point x: mates_in_S[x, v] = sum(S_mask[line_mates[x, v, :]])
    #   can_add[x] = all(mates_in_S[x, :] >= 1, axis=1)

    # Build a 2D boolean array: S_mask is (2401,), line_mates is (2401, 50, 48)
    # We want: for each (x, v), is any of the 48 mates in S?
    # S_mask[line_mates[x, v, :]] gives a (48,) bool array
    # We want: any(S_mask[line_mates[x, v, :]]) for all (x, v)
    # => shape (2401, 50) bool

    # Efficient vectorized approach:
    # S_mask_expanded = S_mask[line_mates]  # shape (2401, 50, 48)
    # covered = S_mask_expanded.any(axis=2)  # shape (2401, 50)
    # can_add = covered.all(axis=1)  # shape (2401,)

    # But S_mask_expanded would be 2401*50*48 = 5.76M bools = ~5.76 MB, which is fine.

    # Let's do it in a loop over points to save memory, or just do it all at once.
    # Actually, let's do it all at once for speed.

    def compute_can_add(S_mask, line_mates):
        # S_mask: (2401,) bool
        # line_mates: (2401, 50, 48) int64
        # Returns: (2401,) bool, can_add[x] = True iff x can be added to S
        # Expand S_mask to (2401, 50, 48)
        S_expanded = S_mask[line_mates]  # (2401, 50, 48)
        covered = S_expanded.any(axis=2)  # (2401, 50)
        return covered.all(axis=1)  # (2401,)

    # Greedy: repeatedly find all points that can be added, add them all at once.
    # This is a "parallel greedy" approach.

    changed = True
    iterations = 0
    while changed:
        changed = False
        if time.time() > deadline:
            break
        iterations += 1

        can_add = compute_can_add(S_mask, line_mates)

        # Exclude points already in S
        candidates = can_add & ~S_mask

        if not candidates.any():
            break

        # Add all candidates at once
        S_mask[candidates] = True
        changed = True

    # Build N = all_points - S
    N_mask = ~S_mask
    N_flat = all_flat[N_mask]
    N_pts = from_flat(N_flat)  # shape (k, 4)

    # Convert to (k, 2, 2) array
    k = len(N_pts)
    points = np.zeros((k, 2, 2), dtype=np.int64)
    points[:, 0, 0] = N_pts[:, 0]
    points[:, 0, 1] = N_pts[:, 1]
    points[:, 1, 0] = N_pts[:, 2]
    points[:, 1, 1] = N_pts[:, 3]

    return points

# EVOLVE-BLOCK-END
