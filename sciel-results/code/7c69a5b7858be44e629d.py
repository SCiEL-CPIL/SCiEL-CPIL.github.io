"""Stronger randomized greedy / local-search construction for 4-colourings of K_n
without monochromatic triangles.  Starts from the known n=32 Greenwood-Gleason
colouring and tries to grow it or find a larger one via randomised edge perturbations
and local swaps.  A hard 150 s wall-clock deadline is respected."""
import time
import random
import sys

# ---- Known good n=32 colouring (Greenwood-Gleason / Clebsch doubling) ----
def _make_base32():
    """Return adjacency matrix for the n=32 4-colouring (index 0..31)."""
    # GF(16) arithmetic
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    mat = [[-1] * 32 for _ in range(32)]
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            mat[i][j] = c
            mat[j][i] = c
    return mat


def _triangle_count(mat, n):
    """Count monochromatic triangles in mat[0:n][0:n]."""
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = mat[i][j]
            if c < 0:
                continue
            for k in range(j + 1, n):
                if mat[i][k] == c and mat[j][k] == c:
                    cnt += 1
    return cnt


def _slack(mat, n):
    """For each edge (i,j) with colour c, slack = min over k of (1 if mat[i][k]==c and mat[j][k]==c else 0).
    Returns the minimum slack over all edges (0 means a triangle exists)."""
    # If any triangle exists, slack is 0.
    if _triangle_count(mat, n) > 0:
        return 0
    # Otherwise compute how close we are: for each edge, count k that would complete a triangle.
    min_slack = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            c = mat[i][j]
            if c < 0:
                continue
            threats = 0
            for k in range(n):
                if k != i and k != j:
                    if mat[i][k] == c and mat[j][k] == c:
                        threats += 1
            if threats < min_slack:
                min_slack = threats
            if min_slack == 0:
                return 0
    return min_slack


def _try_grow(mat, target_n, rng, deadline, max_iters=5000):
    """Try to extend mat (size current_n) to target_n by adding one vertex at a time.
    Returns (success, new_mat, new_n)."""
    cur_n = len(mat)
    if target_n <= cur_n:
        return False, mat, cur_n

    # Build new matrix of size target_n
    new_mat = [[-1] * target_n for _ in range(target_n)]
    for i in range(cur_n):
        for j in range(cur_n):
            new_mat[i][j] = mat[i][j]

    # Try to add vertices one by one
    for target in range(cur_n + 1, target_n + 1):
        added = False
        for attempt in range(max_iters):
            if time.time() > deadline:
                return False, new_mat, target - 1
            # Try to add vertex 'target' by assigning colours to edges (target, i) for i in 0..target-1
            # Greedy: for each existing vertex i, pick a colour that doesn't create a triangle.
            # Randomise order of vertices.
            order = list(range(target))
            rng.shuffle(order)
            assignment = {}
            ok = True
            for i in order:
                # For each colour c, check if assigning edge (target, i) = c creates a monochromatic triangle.
                # Triangle (target, i, k) is monochromatic in colour c if mat[i][k] == c for some k < target.
                # Actually: triangle (target, i, k) has colours c (target-i), mat[i][k], mat[target][k].
                # We haven't assigned mat[target][k] yet for k not yet processed.
                # So we can only check against already-assigned vertices.
                # Let's use a different approach: for vertex i, the forbidden colours are those c
                # such that there exists k already assigned with mat[i][k]==c and assignment[k]==c.
                forbidden = set()
                for k in range(target):
                    if k in assignment and k != i:
                        c2 = assignment[k]
                        if mat[i][k] == c2:
                            forbidden.add(c2)
                available = [c for c in range(4) if c not in forbidden]
                if not available:
                    ok = False
                    break
                # Choose randomly from available
                assignment[i] = rng.choice(available)
            if ok:
                # Verify no monochromatic triangle involving the new vertex
                valid = True
                for i in range(target):
                    for k in range(i + 1, target):
                        c = assignment[i]
                        if c == assignment[k] and mat[i][k] == c:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Commit
                    for i in range(target):
                        new_mat[target][i] = assignment[i]
                        new_mat[i][target] = assignment[i]
                    added = True
                    break
        if not added:
            return False, new_mat, target - 1
        mat = new_mat

    return True, new_mat, target_n


def _local_search(mat, n, rng, deadline, max_steps=2000):
    """Try to improve slack or reduce triangle count via edge colour swaps."""
    tc = _triangle_count(mat, n)
    if tc == 0:
        return mat
    best_mat = [row[:] for row in mat]
    best_tc = tc
    for step in range(max_steps):
        if time.time() > deadline:
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or mat[i][j] < 0:
            continue
        old_c = mat[i][j]
        new_c = rng.randrange(4)
        if new_c == old_c:
            continue
        # Try swap
        mat[i][j] = new_c
        mat[j][i] = new_c
        new_tc = _triangle_count(mat, n)
        if new_tc < best_tc:
            best_tc = new_tc
            best_mat = [row[:] for row in mat]
            if best_tc == 0:
                break
        else:
            mat[i][j] = old_c
            mat[j][i] = old_c
    # Restore best
    for i in range(n):
        for j in range(n):
            mat[i][j] = best_mat[i][j]
    return mat


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 s safety margin

    base = _make_base32()
    best_n = 32
    best_mat = base
    best_tc = 0

    rng = random.Random(42)

    # Phase 1: Try to grow from 32 upward
    for target in range(33, 63):
        if time.time() > deadline:
            break
        success, new_mat, new_n = _try_grow(base, target, rng, deadline, max_iters=2000)
        if success:
            tc = _triangle_count(new_mat, new_n)
            if tc == 0:
                best_n = new_n
                best_mat = new_mat
                best_tc = 0
            elif tc < best_tc:
                # Keep as candidate but not valid
                pass
        else:
            # If we couldn't even grow to target, break
            if new_n < target:
                # Try to use the partial result
                tc = _triangle_count(new_mat, new_n)
                if tc == 0 and new_n > best_n:
                    best_n = new_n
                    best_mat = new_mat
                    best_tc = 0
                break

    # Phase 2: Local search on best valid mat to see if we can fix triangles or improve
    if best_tc > 0:
        best_mat = _local_search(best_mat, best_n, rng, deadline, max_steps=5000)
        best_tc = _triangle_count(best_mat, best_n)

    # Phase 3: Try random restarts with perturbed base colourings
    for restart in range(50):
        if time.time() > deadline:
            break
        # Perturb the base colouring
        perturbed = [row[:] for row in base]
        n_edges = best_n * (best_n - 1) // 2
        # Randomly flip some edges
        for _ in range(min(20, n_edges // 10)):
            i = rng.randrange(best_n)
            j = rng.randrange(best_n)
            if i != j:
                perturbed[i][j] = rng.randrange(4)
                perturbed[j][i] = perturbed[i][j]
        # Try local search
        tc = _triangle_count(perturbed, best_n)
        if tc == 0:
            # Try to grow
            for target in range(best_n + 1, min(best_n + 5, 63)):
                if time.time() > deadline:
                    break
                success, new_mat, new_n = _try_grow(perturbed, target, rng, deadline, max_iters=500)
                if success and _triangle_count(new_mat, new_n) == 0:
                    best_n = new_n
                    best_mat = new_mat
                    break
                else:
                    break
        else:
            # Local search to fix
            fixed = _local_search(perturbed, best_n, rng, deadline, max_steps=1000)
            fixed_tc = _triangle_count(fixed, best_n)
            if fixed_tc == 0:
                # Try to grow
                for target in range(best_n + 1, min(best_n + 5, 63)):
                    if time.time() > deadline:
                        break
                    success, new_mat, new_n = _try_grow(fixed, target, rng, deadline, max_iters=500)
                    if success and _triangle_count(new_mat, new_n) == 0:
                        best_n = new_n
                        best_mat = new_mat
                        break
                    else:
                        break

    # Final check
    final_tc = _triangle_count(best_mat, best_n)
    if final_tc > 0:
        # Fall back to base
        best_n = 32
        best_mat = _make_base32()

    # Build colors list in lexicographic order
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_mat[i][j])

    return {'n': best_n, 'colors': colors}
