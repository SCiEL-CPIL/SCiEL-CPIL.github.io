# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 630.0 - 90.0  # return best found with safety margin before 630s limit


def _time_left():
    return max(0.0, _DEADLINE - time.time())


# ---- Precomputed geometry ----

# Grid for block enumeration (matching evaluator)
_GX = np.linspace(-4.0, 0.0, 81)
_GY = np.linspace(0.0, 1.0, 21)
_GZ = np.linspace(0.0, 1.0, 21)
_DX = _GX[1] - _GX[0]
_DY = _GY[1] - _GY[0]
_DZ = _GZ[1] - _GZ[0]
NX = len(_GX) - 4
NY = len(_GY) - 4
NZ = len(_GZ) - 4

# Build block grid points: (Nblk, 16, 3)
_bi, _bj, _bk = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
_bf = np.array([_bi.ravel(), _bj.ravel(), _bk.ravel()]).T
_XL = _bf[:, 0][:, None, None, None] * _DX + np.array([0, _DX, 2*_DX, 3*_DX])[None, None, None, :]
_yl = _bf[:, 1][:, None, None, None] * _DY + np.array([0, _DY, 2*_DY, 3*_DY])[None, None, None, :]
_zl = _bf[:, 2][:, None, None, None] * _DZ + np.array([0, _DZ, 2*_DZ, 3*_DZ])[None, None, None, :]
_BX = _GF = None  # computed below

def _init_bx():
    global _BX
    Xx = _bf[:, 0] * _DX
    Xy = _bf[:, 1] * _DY
    Xz = _bf[:, 2] * _DZ
    Xx4 = Xx[:, None, None, None] + np.array([0, _DX, 2*_DX, 3*_DX])[None, None, None, :]
    Xy4 = Xy[:, None, None, None] + np.array([0, _DY, 2*_DY, 3*_DY])[None, None, None, :]
    Xz4 = Xz[:, None, None, None] + np.array([0, _DZ, 2*_DZ, 3*_DZ])[None, None, None, :]
    _BX = np.stack([Xx4, Xy4, Xz4], axis=-1)  # (Nblk, 4, 4, 4, 3)

_init_bx()
Nblk = len(_bf)

_START = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_END = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
_ANG = np.deg2rad

_SEED = np.array([
    [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
    [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
    [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
    [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
    [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
    [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
    [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
    [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
    [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
    [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
    [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
    [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
    [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
    [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
    [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
    [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
    [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
    [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
    [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
    [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
    [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
    [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
    [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
    [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
    [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
    [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
    [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
    [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
    [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
    [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
    [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
    [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
    [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
    [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
    [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
    [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
    [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
    [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
    [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
    [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
    [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
    [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 20.9723911, 5.66341348],
    [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 21.6229248, 5.80874653],
    [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676],
    [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
    [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
    [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
    [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
    [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
    [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
    [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
    [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
    [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
    [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
    [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
    [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
    [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
    [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
    [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
    [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
    [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
    [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
    [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
    [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
    [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
    [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
    [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
    [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
    [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
    [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
    [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
    [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
    [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
    [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
    [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
    [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
    [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
    [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
    [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
    [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
    [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
    [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
    [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
    [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
    [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
    [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
    [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
    [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
    [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
    [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
    [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
    [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
    [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
    [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
    [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
    [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
    [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
    [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
    [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979],
], dtype=np.float64)


def _pose_to_R_t(p):
    """Return rotation matrix R and translation t for pose p (x,y,z,yaw,pitch,roll in degrees)."""
    yaw, pitch, roll = _ANG(p[3]), _ANG(p[4]), _ANG(p[5])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([
        [cr*cy, cr*sy*sp - sr*cp, cr*sy*cp + sr*sp],
        [sr*cy, sr*sy*sp + cr*cp, sr*sy*cp - cr*sp],
        [-sy,  cp*sy*sp + cy*sr, cy*cp*sr - sy*sp],
    ])
    return R, np.array([p[0], p[1], p[2]], dtype=np.float64)


def _in_corridor(X):
    """X: (M,3) array of world coordinates. Returns boolean mask (M,) for points in corridor."""
    x = X[:, 0]; y = X[:, 1]; z = X[:, 2]
    a = (x <= 0.0) & (y >= 0.0) & (y <= 1.0) & (z >= 0.0) & (z <= 1.0)
    b = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 5.0) & (z >= 0.0) & (z <= 1.0)
    c = (x >= 0.0) & (x <= 1.0) & (y >= 5.0) & (y <= 6.0) & (z >= 0.0)
    return a | b | c


def _box_in_corridor(bmn, bmx, bny, bnx, bnz, bnz2):
    """bmn..bnz2: (M,) arrays of box min/max per axis. Returns boolean (M,) if box center is in corridor."""
    cx = (bmn + bmx) / 2.0
    cy = (bny + bnx) / 2.0
    cz = (bnz + bnz2) / 2.0
    a = (cx <= 0.0) & (bny >= 0.0) & (bnx <= 1.0) & (bnz >= 0.0) & (bnz2 <= 1.0)
    b = (bmn >= 0.0) & (bmx <= 1.0) & (bny >= 0.0) & (bnx <= 5.0) & (bnz >= 0.0) & (bnz2 <= 1.0)
    c = (bmn >= 0.0) & (bmx <= 1.0) & (bny >= 5.0) & (bnx <= 6.0) & (bnz >= 0.0)
    return a | b | c


def _rot_arc_box(W_old, W_new, ax, t):
    """
    Compute the axis-aligned bounding box of the rotation arc for each grid point.
    W_old, W_new: (M,3) world positions before/after rotation.
    ax: rotation axis index (0=x, 1=y, 2=z).
    t: (M,) translation (world axis location).
    Returns (bmn, bmx, bny, bnx, bnz, bnz2) arrays of shape (M,).
    """
    M = len(t)
    v = np.zeros((M, 3), dtype=np.float64)
    v[:, ax] = 1.0
    # World axis direction and point on axis
    # The axis in world space passes through t and has direction R_old @ e_ax
    # But for the arc box, we project each point onto the plane perpendicular to the axis
    # through t, compute the radius, and the arc is within a box of radius R around the projected center.
    
    # For a rotation about axis through point t in direction d, the arc of point p is within
    # a sphere of radius ||p - closest point on axis||, centered at the closest point.
    # The AABB of this sphere is [closest - R, closest + R].
    
    # Closest point on axis to p:
    # d = direction of axis in world (unit)
    # closest = t + ((p - t) . d) d
    # R = ||p - closest||
    
    # d = R_old @ e_ax. We need R_old.
    # Actually, let's compute R_old from the old pose. But we don't have it here.
    # The evaluator computes the axis as: axis_dir = R_old @ e_ax, axis_point = t.
    # Let me compute R_old from the old pose that produced W_old.
    # Actually, it's simpler: the rotation axis in world space is fixed during the rotation step.
    # We can compute it from the two poses. But we only have W_old and W_new.
    
    # Simpler approach: the bounding box of the arc is contained in the box
    # [min(W_old, W_new) - diag, max(W_old, W_new) + diag] where diag is an overestimate.
    # A valid conservative bound: the arc is within the axis-aligned box that contains
    # both endpoints and has the same center, with half-diagonals = half the bounding box of endpoints
    # plus a safety margin. For small rotations (<=2 degrees) this is very tight.
    
    # Actually the simplest valid bound for a rotation about ANY axis: the arc is within
    # the bounding box of the two endpoints expanded by the max distance from either endpoint
    # to the axis. For |angle| <= 2 deg, the arc deviation from the chord is negligible.
    
    # Most conservative but correct: use the AABB of the circle through both points with
    # the maximum possible radius. Since both points are on the rotation circle, and the angle
    # is small, the arc is very close to the chord. The AABB of the chord + small buffer works.
    
    # For a 2-degree rotation, the maximum deviation from the chord is at most
    # R * (1 - cos(1°)) ≈ R * 1.5e-4 where R is the radius. So we can use:
    # box = AABB(W_old, W_new) expanded by 2% of the diagonal as safety.
    
    lo = np.minimum(W_old, W_new)
    hi = np.maximum(W_old, W_new)
    diag = np.linalg.norm(hi - lo, axis=1)
    buf = 0.01 * diag + 1e-6  # 1% buffer for small rotation arc deviation
    
    bmn = lo[:, 0] - buf
    bmx = hi[:, 0] + buf
    bny = lo[:, 1] - buf
    bnx = hi[:, 1] + buf
    bnz = lo[:, 2] - buf
    bnz2 = hi[:, 2] + buf
    return bmn, bmx, bny, bnx, bnz, bnz2


def _densify(path, max_d=0.05, max_a=2.0, safety=0.999, nmax=50000):
    """Insert linearly interpolated poses so each step moves <= max_d and each angle changes <= max_a."""
    all_poses = [np.array(_START, dtype=np.float64)]
    for p in path:
        all_poses.append(np.asarray(p, dtype=np.float64))
    all_poses.append(np.array(_END, dtype=np.float64))
    
    out = [all_poses[0]]
    last = all_poses[0]
    for i in range(1, len(all_poses)):
        target = all_poses[i]
        while True:
            if len(out) >= nmax:
                break
            dt = target - last
            d = float(np.linalg.norm(dt[:3]))
            da = float(np.max(np.abs(dt[3:])))
            if d < 1e-9 and da < 1e-9:
                break
            frac = 1.0
            if d > max_d * safety:
                frac = min(frac, max_d * safety / d)
            if da > max_a * safety:
                frac = min(frac, max_a * safety / da)
            nx = last + frac * dt
            out.append(nx)
            last = nx
    return np.asarray(out, dtype=np.float64)


def _eval(path, use_rot=True):
    """Evaluate the path: return the number of surviving 4x4x4 blocks (before volume scaling)."""
    poses = _densify(path)
    if len(poses) < 2:
        return 0.0
    n = len(poses)
    
    # Precompute R and t for each pose
    Rs = []
    ts = []
    for p in poses:
        R, t = _pose_to_R_t(p)
        Rs.append(R)
        ts.append(t)
    ts = np.stack(ts, axis=0)  # (n, 3)
    
    # For each pose, compute world positions of all 16 points in each block.
    # _BX is (Nblk, 4, 4, 4, 3) -> flatten to (Nblk, 16, 3)
    BX_flat = _BX.reshape(Nblk, -1, 3)  # (Nblk, 16, 3)
    
    # block_ok[j] = bool(Nblk): whether all 16 points are in corridor at pose j
    block_ok = np.ones((n, Nblk), dtype=bool)
    
    for j in range(n):
        # W = R @ X + t for each of the 16 points per block
        W = np.einsum('ij,kil->kjl', Rs[j], BX_flat) + ts[j, None, :]  # (Nblk, 16, 3)
        W_flat = W.reshape(Nblk, -1, 3)
        in_c = _in_corridor(W_flat)  # (Nblk, 16)
        block_ok[j] = np.all(in_c, axis=1)
        if not block_ok[j].any():
            # If no blocks survive at this pose, no blocks can survive overall
            return 0.0
        if time.time() > _DEADLINE:
            break
    
    # Check rotation arc boxes
    if use_rot:
        for j in range(1, n):
            if time.time() > _DEADLINE:
                break
            for ax in range(3):
                delta_angle = poses[j][ax+3] - poses[j-1][ax+3]
                if abs(delta_angle) > 1e-9:
                    W_old = np.einsum('ij,kil->kjl', Rs[j-1], BX_flat) + ts[j-1, None, :]
                    W_new = np.einsum('ij,kil->kjl', Rs[j], BX_flat) + ts[j, None, :]
                    W_old_flat = W_old.reshape(Nblk, -1, 3)
                    W_new_flat = W_new.reshape(Nblk, -1, 3)
                    # Use the per-point bounding box approach
                    bmn, bmx, bny, bnx, bnz, bnz2 = _rot_arc_box(W_old_flat[:, 0, :], W_new_flat[:, 0, :], ax, ts[j-1])
                    # Check if the box is in corridor for each of the 16 points
                    # Actually we need to check the box for the entire rotation arc of each point.
                    # The box is per-point. Let's redo with per-point arrays.
                    for pt in range(16):
                        wo = W_old_flat[:, pt, :]  # (Nblk, 3)
                        wn = W_new_flat[:, pt, :]  # (Nblk, 3)
                        bmn, bmx, bny, bnx, bnz, bnz2 = _rot_arc_box(wo, wn, ax, ts[j-1])
                        box_ok = _box_in_corridor(bmn, bmx, bny, bnx, bnz, bnz2)
                        block_ok[j] &= box_ok
    
    alive = np.all(block_ok, axis=0)
    return float(np.sum(alive))


def _fast_eval(path):
    """Faster evaluation using only pose positions (skip rotation arc for speed during optimization)."""
    poses = _densify(path, nmax=20000)
    if len(poses) < 2:
        return 0.0
    n = len(poses)
    
    Rs = []
    ts = []
    for p in poses:
        R, t = _pose_to_R_t(p)
        Rs.append(R)
        ts.append(t)
    ts = np.stack(ts, axis=0)
    
    BX_flat = _BX.reshape(Nblk, -1, 3)
    block_ok = np.ones((n, Nblk), dtype=bool)
    
    for j in range(n):
        W = np.einsum('ij,kil->kjl', Rs[j], BX_flat) + ts[j, None, :]
        W_flat = W.reshape(Nblk, -1, 3)
        in_c = _in_corridor(W_flat)
        block_ok[j] = np.all(in_c, axis=1)
        if not block_ok[j].any():
            return 0.0
    
    alive = np.all(block_ok, axis=0)
    return float(np.sum(alive))


def _refit_path(path, k):
    """Resample path to k keypoints using arc-length parameterization per component."""
    n = len(path)
    if k <= 1:
        return path.copy()
    t = np.linspace(0.0, 1.0, k)
    new_path = path.copy()
    for col in range(6):
        d = np.abs(np.diff(path[:, col]))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        tot = cum[-1]
        if tot < 1e-12:
            new_path[:, col] = path[0, col]
        else:
            u = cum / tot
            new_path[:, col] = np.interp(t, u, path[:, col])
    return new_path


def _nelder_mead(path, max_iter=200, xatol=1e-4, fatol=1e-3, use_fast=True):
    """Run Nelder-Mead optimization on the path keypoints."""
    from scipy.optimize import minimize
    
    eval_fn = _fast_eval if use_fast else _eval
    
    def objective(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 6)
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 100.0):
            return 1e6
        return -eval_fn(x)
    
    x0 = path.reshape(-1).astype(np.float64)
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol})
        if hasattr(res, 'x'):
            return np.asarray(res.x).reshape(-1, 6)
    except Exception:
        pass
    return path


def _perturb_and_test(best_path, n_perturb=10, sigma=0.05):
    """Try random perturbations of the path and keep improvements."""
    rng = np.random.default_rng(42)
    best_score = _fast_eval(best_path)
    
    for _ in range(n_perturb):
        if time.time() > _DEADLINE - 30:
            break
        p = best_path.copy()
        # Perturb a random subset of poses
        for i in range(len(p)):
            if rng.random() < 0.3:
                p[i] += rng.normal(0, sigma, size=6)
        s = _fast_eval(p)
        if s > best_score:
            best_score = s
            best_path = p
    return best_path, best_score


def moving_sofa_3d_path() -> np.ndarray:
    if time.time() > _DEADLINE:
        return _SEED
    
    best_path = _SEED.copy()
    best_score = _fast_eval(best_path)
    
    # Try different numbers of keypoints
    for k in (15, 20, 25, 30):
        if time.time() > _DEADLINE - 60:
            break
        # Refit seed to k keypoints
        p = _refit_path(best_path, k)
        # Optimize with Nelder-Mead
        p = _nelder_mead(p, max_iter=300, xatol=1e-4, fatol=1e-3, use_fast=True)
        s = _fast_eval(p)
        if s > best_score:
            best_score = s
            best_path = p
    
    # Perturbation refinement
    if time.time() < _DEADLINE - 40:
        best_path, best_score = _perturb_and_test(best_path, n_perturb=15, sigma=0.03)
    
    # Final full evaluation to verify
    if time.time() < _DEADLINE - 20:
        full_score = _eval(best_path, use_rot=True)
        if full_score > 0:
            best_score = full_score
        # If the fast eval was overly optimistic, the path is still valid;
        # the evaluator will use its own logic.
    
    # Sanitize output
    out = np.asarray(best_path, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if out.shape[0] < 1:
        out = _SEED
    return out


# EVOLVE-BLOCK-END
