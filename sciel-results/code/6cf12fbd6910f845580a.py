import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 585.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7
    total_points = p * p * p * p  # 2401
    num_dirs = p * p + 1  # 50 directions
    line_len = p * p  # 49 points per line
    other_len = p * p - 1  # 48 other points

    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def point_unflat(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Precompute line_other: shape (num_dirs, total_points, other_len)
    # line_other[d, x, i] = index of the i-th other point on line L(d, x)
    line_other = np.zeros((num_dirs, total_points, other_len), dtype=np.int32)

    # Direction 0: v = (0, 1)
    d = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_flat(xa, xb, ya, yb)
                    t_idx = 0
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            line_other[d, x_idx, t_idx] = point_flat(xa, xb, nya, nyb)
                            t_idx += 1

    # Directions 1..49: v = (1, m) for m in F_49
    d = 1
    for ma in range(p):
        for mb in range(p):
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x_idx = point_flat(xa, xb, ya, yb)
                            t_idx = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    tma = (ta * ma + tb * mb * k) % p
                                    tmb = (ta * mb + tb * ma) % p
                                    nya = (ya + tma) % p
                                    nyb = (yb + tmb) % p
                                    line_other[d, x_idx, t_idx] = point_flat(nxa, nxb, nya, nyb)
                                    t_idx += 1
            d += 1

    # Precompute: for each point, which (direction, position) pairs it appears in
    # point_dirs[p_idx] = list of (d, pos) where line_other[d, x, pos] == p_idx
    # This helps for incremental verification
    point_dirs = [[] for _ in range(total_points)]
    for d in range(num_dirs):
        for x in range(total_points):
            for pos in range(other_len):
                pt = line_other[d, x, pos]
                point_dirs[pt].append((d, x, pos))

    # For faster verification: precompute for each point x, the set of (d, other_points)
    # Actually, let's use a different approach for fast verification:
    # Maintain a count array: for each point x, count how many "other" points on each line are in N
    # But that's complex. Let's use the vectorized approach but optimize it.

    # Fast verification using bitsets
    # Represent in_N as a bitset: in_N_bits = 2401-bit integer or array of uint64
    # For each direction d and point x, we need to check if all 48 other points are in N.
    # Using numpy: in_N[line_other[d, x]] all True

    def verify_nikodym(in_N):
        """
        in_N: boolean array of shape (total_points,)
        Returns True if Nikodym property holds.
        """
        if time.time() - start_time > deadline:
            return False

        # For each direction d, compute which points x have all other points on L(d,x) in N.
        # in_N[line_other] has shape (50, 2401, 48)
        # This is expensive. Let's do it direction by direction and short-circuit.

        for d in range(num_dirs):
            # in_N[line_other[d]] has shape (2401, 48)
            # all_in[d] = np.all(in_N[line_other[d]], axis=1) has shape (2401,)
            pass

        # Actually, let's just do the full vectorized check
        # in_N[line_other] has shape (50, 2401, 48) - this is 50*2401*48 = 5,762,400 booleans
        # That's about 5.7 MB, should be fine
        all_in = np.all(in_N[line_other], axis=2)  # shape (50, 2401)
        valid_points = np.any(all_in, axis=0)  # shape (2401,)
        return np.all(valid_points)

    # Start with a known valid construction:
    # Hermitian unital: U = {(xa, xb, ya, yb) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}
    # Complement S = {points in U where y_a is a square mod 7}
    # N = F_49^2 \ S

    in_S = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        # Check if ya is a square mod 7: squares are {0, 1, 2, 4}
                        if ya in {0, 1, 2, 4}:
                            in_S[point_flat(xa, xb, ya, yb)] = True

    in_N = ~in_S

    # Verify initial construction
    if not verify_nikodym(in_N):
        # Fallback: return a trivial valid set (all points)
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Initial complement size
    initial_S_size = int(np.sum(in_S))

    # Greedy expansion: try to add more points to S (remove from N)
    # A point can be added to S if removing it from N doesn't break the Nikodym property.

    # For efficiency, we'll use a randomized local search approach:
    # 1. Start with the initial valid set
    # 2. Try to add random points to S
    # 3. If it remains valid, keep it; otherwise revert

    best_in_S = in_S.copy()
    best_S_size = initial_S_size

    # Try multiple random orders with greedy addition
    np.random.seed(42)

    for restart in range(200):  # 200 restarts
        if time.time() - start_time > deadline:
            break

        current_in_S = in_S.copy()
        current_S_size = initial_S_size

        # Create random order of all points
        all_point_indices = np.arange(total_points)
        np.random.shuffle(all_point_indices)

        # Try to add each point to S
        for p_idx in all_point_indices:
            if time.time() - start_time > deadline:
                break

            if current_in_S[p_idx]:
                continue

            # Try adding p_idx to S
            current_in_S[p_idx] = True
            current_in_N = ~current_in_S

            if verify_nikodym(current_in_N):
                current_S_size += 1
            else:
                # Revert
                current_in_S[p_idx] = False

        # Update best
        if current_S_size > best_S_size:
            best_S_size = current_S_size
            best_in_S = current_in_S.copy()

        # If we've used up most of our time, break
        if time.time() - start_time > deadline - 60:
            break

    # Build output from best_in_S
    in_N_final = ~best_in_S

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not best_in_S[point_flat(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
