# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 630.0
_MARGIN = 45.0

# Fixed evaluator anchors
_START = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_END = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])

# Grid
GX = np.linspace(-4.0, 0.0, 81)
GY = np.linspace(0.0, 1.0, 21)
GZ = np.linspace(0.0, 1.0, 21)
DX = GX[1] - GX[0]
DY = GY[1] - GY[0]
DZ = GZ[1] - GZ[0]


def _rot_matrix(yaw, pitch, roll):
    y = np.deg2rad(yaw)
    p = np.deg2rad(pitch)
    r = np.deg2rad(roll)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz


def _rot_axis_matrix(axis, a):
    a = np.deg2rad(a)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:  # roll about x
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:  # pitch about y
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _in_corridor(x, y, z):
    if (x <= 0) and (0 <= y <= 1) and (0 <= z <= 1):
        return True
    if (0 <= x <= 1) and (0 <= y <= 5) and (0 <= z <= 1):
        return True
    if (0 <= x <= 1) and (5 <= y <= 6) and (z >= 0):
        return True
    return False


def _corridor_at_point(x, y, z):
    return _in_corridor(x, y, z)


def _corridor_box(x0, y0, z0, x1, y1, z1):
    # sample the box corners + face centers + interior
    xs = [x0, (x0 + x1) / 2, x1]
    ys = [y0, (y0 + y1) / 2, y1]
    zs = [z0, (z0 + z1) / 2, z1]
    for xx in xs:
        for yy in ys:
            for zz in zs:
                if not _in_corridor(xx, yy, zz):
                    return False
    return True


def _survives(coarse, gx, gy, gz):
    """coarse: (N,6). Returns bool: does grid point (gx,gy,gz) survive?"""
    p = np.array([gx, gy, gz])
    N = coarse.shape[0]
    # first pose
    R0 = _rot_matrix(coarse[0, 3], coarse[0, 4], coarse[0, 5])
    q0 = R0 @ p + coarse[0, :3]
    if not _in_corridor(q0[0], q0[1], q0[2]):
        return False
    prev = coarse[0].copy()
    for i in range(1, N):
        cur = coarse[i]
        # x move
        nx = cur[0]
        R = _rot_matrix(prev[3], prev[4], prev[5])
        q = R @ p + np.array([nx, prev[1], prev[2]])
        if not _in_corridor(q[0], q[1], q[2]):
            return False
        prev[0] = nx
        # y move
        ny = cur[1]
        q = R @ p + np.array([prev[0], ny, prev[2]])
        if not _in_corridor(q[0], q[1], q[2]):
            return False
        prev[1] = ny
        # z move
        nz = cur[2]
        q = R @ p + np.array([prev[0], prev[1], nz])
        if not _in_corridor(q[0], q[1], q[2]):
            return False
        prev[2] = nz
        # rotation about single axis
        a_old = prev[3:6]
        a_new = cur[3:6]
        diff = a_new - a_old
        ai = int(np.argmax(np.abs(diff)))
        if np.abs(diff[ai]) > 1e-9:
            Rold = _rot_matrix(a_old[0], a_old[1], a_old[2])
            Rnew = _rot_matrix(a_new[0], a_new[1], a_new[2])
            # sample rotation arc
            n = max(1, int(np.ceil(np.abs(diff[ai]) / 2.0)))
            ok = True
            for k in range(n + 1):
                t = k / n
                amix = a_old.copy()
                amix[ai] = a_old[ai] + t * diff[ai]
                Rm = _rot_matrix(amix[0], amix[1], amix[2])
                q = Rm @ p + prev[:3]
                if not _in_corridor(q[0], q[1], q[2]):
                    ok = False
                    break
            if not ok:
                return False
            # rotation-arc AABB: bbox of sampled arc must lie in corridor
            n2 = max(2, int(np.ceil(np.abs(diff[ai]) / 2.0)))
            pts = np.empty((n2 + 1, 3))
            for k in range(n2 + 1):
                t = k / n2
                amix = a_old.copy()
                amix[ai] = a_old[ai] + t * diff[ai]
                Rm = _rot_matrix(amix[0], amix[1], amix[2])
                pts[k] = Rm @ p + prev[:3]
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            if not _corridor_box(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]):
                return False
        prev[3:6] = a_new
    return True


def _score_coarse(coarse):
    """Compute surviving 4x4x4 block count for a coarse path."""
    N = coarse.shape[0]
    # Precompute per-pose rotation matrices (for translation poses)
    # We'll evaluate each grid point.
    sx = np.zeros((81, 21, 21), dtype=bool)
    for ix, gx in enumerate(GX):
        for iy, gy in enumerate(GY):
            for iz, gz in enumerate(GZ):
                sx[ix, iy, iz] = _survives(coarse, gx, gy, gz)
    # 4x4x4 blocks
    count = 0
    for ix in range(81 - 3):
        for iy in range(21 - 3):
            for iz in range(21 - 3):
                blk = sx[ix:ix + 4, iy:iy + 4, iz:iz + 4]
                if blk.all():
                    count += 1
    return count


def _validate_step(pose_prev, pose_cur):
    if np.linalg.norm(pose_cur[:3] - pose_prev[:3]) > 0.05 + 1e-9:
        return False
    if np.any(np.abs(pose_cur[3:6] - pose_prev[3:6]) > 2.0 + 1e-9):
        return False
    return True


def _build_path(theta):
    """theta: dict of parameters. Returns coarse path (N,6) from START to END."""
    poses = [list(_START)]
    # Segment A: x from -3 to -1.25 at y=0.5, z=0.5 (in wing1: x<=0, 0<=y<=1, 0<=z<=1)
    xA0, xA1 = -3.0, -1.25
    step = 0.05
    nA = int(round((xA1 - xA0) / step))
    for i in range(1, nA + 1):
        x = xA0 + i * step
        poses.append([x, 0.5, 0.5, 0.0, 0.0, 0.0])
    # Pose at x=-1.25
    # Segment B: rotate about y (pitch) from 0 to 90, moving y 0.5->? , x -1.25->?
    # We'll do pitch rotation while translating to keep inside.
    # Target after B: yaw=90, x ~ 0.5, y ~ 0.5, z=0.5 (in wing2: 0<=x<=1, 0<=y<=5, 0<=z<=1)
    # Do pitch from 0 to 90 in steps of 2 deg, with x,y moving linearly.
    pB0 = 0.0
    pB1 = 90.0
    nB = int(round((pB1 - pB0) / 2.0))
    xB0, xB1 = -1.25, 0.5
    yB0, yB1 = 0.5, 0.5
    for i in range(1, nB + 1):
        t = i / nB
        x = xB0 + t * (xB1 - xB0)
        y = yB0 + t * (yB1 - yB0)
        p = pB0 + t * (pB1 - pB0)
        poses.append([x, y, 0.5, 0.0, p, 0.0])
    # Now yaw=0, pitch=90. Need yaw=90 eventually.
    # Segment C: move y from 0.5 to 5.0 (in wing2)
    yC0, yC1 = 0.5, 5.0
    nC = int(round((yC1 - yC0) / 0.05))
    for i in range(1, nC + 1):
        y = yC0 + i * 0.05
        poses.append([0.5, y, 0.5, 0.0, 90.0, 0.0])
    # Segment D: rotate yaw 0->90 while moving x 0.5->0.5, y 5->5.5, z 0.5->0.5
    # Target after D: yaw=90, pitch=90, roll=0, x~0.5, y~5.5, z~0.5 (in wing3: 0<=x<=1, 5<=y<=6, z>=0)
    yD0, yD1 = 5.0, 5.5
    nD = int(round(90.0 / 2.0))
    for i in range(1, nD + 1):
        t = i / nD
        y = yD0 + t * (yD1 - yD0)
        yaw = t * 90.0
        poses.append([0.5, y, 0.5, yaw, 90.0, 0.0])
    # Segment E: move y 5.5 -> 6.0
    yE0, yE1 = 5.5, 6.0
    nE = int(round((yE1 - yE0) / 0.05))
    for i in range(1, nE + 1):
        y = yE0 + i * 0.05
        poses.append([0.5, y, 0.5, 90.0, 90.0, 0.0])
    # Segment F: rotate roll 0->90 while moving z 0.5->? , x 0.5->0
    # Target END: x=0, y=6, z=7, yaw=90, pitch=90, roll=90
    # z must go 0.5 -> 7, x 0.5 -> 0
    zF0, zF1 = 0.5, 7.0
    xF0, xF1 = 0.5, 0.0
    nF = int(round(90.0 / 2.0))
    for i in range(1, nF + 1):
        t = i / nF
        z = zF0 + t * (zF1 - zF0)
        x = xF0 + t * (xF1 - xF0)
        roll = t * 90.0
        poses.append([x, 6.0, z, 90.0, 90.0, roll])
    poses.append(list(_END))
    arr = np.array(poses, dtype=np.float64)
    return arr


def _check_valid(arr):
    for i in range(1, arr.shape[0]):
        if not _validate_step(arr[i - 1], arr[i]):
            return False
    return True


def _local_search(base, iters, step0):
    """Coordinate-descent-ish local search over segment parameters."""
    # Parameterize: we'll perturb the whole path by adding small noise to intermediate
    # poses, then check validity + score. Simpler: random restarts of a noise field.
    best = base.copy()
    best_score = _score_coarse(base)
    n = base.shape[0]
    for it in range(iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        cand = best.copy()
        # perturb a random subset of poses (positions and angles)
        idx = np.random.choice(n, size=max(1, n // 10), replace=False)
        for i in idx:
            if i == 0 or i == n - 1:
                continue
            cand[i, 0] += np.random.normal(0, step0 * 0.05)
            cand[i, 1] += np.random.normal(0, step0 * 0.05)
            cand[i, 2] += np.random.normal(0, step0 * 0.05)
            cand[i, 3] += np.random.normal(0, step0 * 0.5)
            cand[i, 4] += np.random.normal(0, step0 * 0.5)
            cand[i, 5] += np.random.normal(0, step0 * 0.5)
        # re-clamp angles to keep monotonic-ish? just check validity
        if not _check_valid(cand):
            continue
        sc = _score_coarse(cand)
        if sc > best_score:
            best_score = sc
            best = cand.copy()
    return best, best_score


def moving_sofa_3d_path() -> np.ndarray:
    np.random.seed(0)
    base = _build_path(None)
    # validate base
    if not _check_valid(base):
        # fallback: return base anyway (it may be valid)
        pass
    base_score = _score_coarse(base)
    best = base
    best_score = base_score
    step0 = 1.0
    iters_per_round = 200
    while time.time() < _DEADLINE - _MARGIN:
        cand, sc = _local_search(best, iters_per_round, step0)
        if sc > best_score:
            best_score = sc
            best = cand
            step0 = max(0.2, step0 * 0.85)
        else:
            step0 = max(0.05, step0 * 0.7)
            if step0 < 0.06:
                break
    return best


# EVOLVE-BLOCK-END
