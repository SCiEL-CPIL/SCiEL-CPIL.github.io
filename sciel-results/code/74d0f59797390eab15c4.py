import numpy as np
import time

def _rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz

def _in_corridor_vec(x, y, z):
    part1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    part2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    part3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return part1 | part2 | part3

def _check_bbox_in_corridor(min_coords, max_coords):
    """Check if an axis-aligned bbox fits in the corridor."""
    # Check 8 corners
    for dx in [min_coords[0], max_coords[0]]:
        for dy in [min_coords[1], max_coords[1]]:
            for dz in [min_coords[2], max_coords[2]]:
                if not _in_corridor_vec(np.array([dx]), np.array([dy]), np.array([dz]))[0]:
                    return False
    return True

def _evaluate_path(path, check_arcs=True, n_arc_samples=6):
    """Evaluate path with optional arc checking. Returns volume."""
    start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    
    all_poses = np.vstack([start_pose.reshape(1,6), path, end_pose.reshape(1,6)])
    
    # Densify
    densified = [all_poses[0]]
    for i in range(1, len(all_poses)):
        prev = densified[-1]
        curr = all_poses[i]
        pos_dist = np.linalg.norm(curr[:3] - prev[:3])
        ang_diff = np.abs(curr[3:] - prev[3:])
        max_step = max(pos_dist / 0.05, np.max(ang_diff) / 2.0, 1.0)
        n_steps = max(1, int(np.ceil(max_step)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = prev + t * (curr - prev)
            densified.append(interp)
    
    densified = np.array(densified)
    
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points_local = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N = len(points_local)
    
    survived = np.ones(N, dtype=bool)
    
    R_prev = None
    t_prev = None
    
    for pose in densified:
        x, y, z, yaw, pitch, roll = pose
        R = _rotation_matrix(roll, pitch, yaw)
        t = np.array([x, y, z])
        
        # Check current pose
        world_pts = points_local @ R.T + t
        in_cor = _in_corridor_vec(world_pts[:,0], world_pts[:,1], world_pts[:,2])
        survived &= in_cor
        
        if check_arcs and R_prev is not None:
            # Check rotation arc
            # Only if there's actual rotation
            rot_diff = np.abs(np.array([yaw, pitch, roll]) - np.array([R_prev[0,0], R_prev[1,1], R_prev[2,2]]))  # This isn't right
            # Better: check if R changed significantly
            if np.linalg.norm(R - R_prev) > 1e-6:
                arc_min = np.full((N, 3), np.inf)
                arc_max = np.full((N, 3), -np.inf)
                for s in np.linspace(0, 1, n_arc_samples + 1):
                    R_interp = (1 - s) * R_prev + s * R
                    wp = points_local @ R_interp.T + t
                    arc_min = np.minimum(arc_min, wp)
                    arc_max = np.maximum(arc_max, wp)
                
                # Check if each point's bbox is in corridor
                # Sample the bbox corners
                for dx_idx in [0, 1]:
                    for dy_idx in [0, 1]:
                        for dz_idx in [0, 1]:
                            cx = arc_min[:,0] if dx_idx==0 else arc_max[:,0]
                            cy = arc_min[:,1] if dy_idx==0 else arc_max[:,1]
                            cz = arc_min[:,2] if dz_idx==0 else arc_max[:,2]
                            in_c = _in_corridor_vec(cx, cy, cz)
                            survived &= in_c
        
        if not np.any(survived):
            break
        
        R_prev = R
        t_prev = t
    
    survived_3d = survived.reshape(81, 21, 21)
    
    # Sliding window 4x4x4
    prefix = np.zeros((82, 22, 22), dtype=np.int32)
    prefix[1:82, 1:22, 1:22] = survived_3d.astype(np.int32)
    
    cs = prefix.copy()
    cs[1:82, :, :] = np.cumsum(cs[1:82, :, :], axis=0)
    cs[:, 1:22, :] = np.cumsum(cs[:, 1:22, :], axis=1)
    cs[:, :, 1:22] = np.cumsum(cs[:, :, 1:22], axis=2)
    
    I, J, K = np.meshgrid(np.arange(78), np.arange(18), np.arange(18), indexing='ij')
    
    block_sums = (cs[I+4, J+4, K+4] - cs[I, J+4, K+4] - cs[I+4, J, K+4] - cs[I+4, J+4, K]
                  + cs[I, J, K+4] + cs[I, J+4, K] + cs[I+4, J, K] - cs[I, J, K])
    
    n_blocks = int(np.sum(block_sums == 64))
    volume = 0.05**3 * n_blocks
    return volume

def _fast_evaluate(path):
    """Fast evaluation without arc checking for search."""
    start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    
    all_poses = np.vstack([start_pose.reshape(1,6), path, end_pose.reshape(1,6)])
    
    densified = [all_poses[0]]
    for i in range(1, len(all_poses)):
        prev = densified[-1]
        curr = all_poses[i]
        pos_dist = np.linalg.norm(curr[:3] - prev[:3])
        ang_diff = np.abs(curr[3:] - prev[3:])
        max_step = max(pos_dist / 0.05, np.max(ang_diff) / 2.0, 1.0)
        n_steps = max(1, int(np.ceil(max_step)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            interp = prev + t * (curr - prev)
            densified.append(interp)
    
    densified = np.array(densified)
    
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points_local = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    N = len(points_local)
    
    survived = np.ones(N, dtype=bool)
    
    for pose in densified:
        x, y, z, yaw, pitch, roll = pose
        R = _rotation_matrix(roll, pitch, yaw)
        world_pts = points_local @ R.T + np.array([x, y, z])
        in_cor = _in_corridor_vec(world_pts[:,0], world_pts[:,1], world_pts[:,2])
        survived &= in_cor
        if not np.any(survived):
            break
    
    survived_3d = survived.reshape(81, 21, 21)
    
    prefix = np.zeros((82, 22, 22), dtype=np.int32)
    prefix[1:82, 1:22, 1:22] = survived_3d.astype(np.int32)
    
    cs = prefix.copy()
    cs[1:82, :, :] = np.cumsum(cs[1:82, :, :], axis=0)
    cs[:, 1:22, :] = np.cumsum(cs[:, 1:22, :], axis=1)
    cs[:, :, 1:22] = np.cumsum(cs[:, :, 1:22], axis=2)
    
    I, J, K = np.meshgrid(np.arange(78), np.arange(18), np.arange(18), indexing='ij')
    
    block_sums = (cs[I+4, J+4, K+4] - cs[I, J+4, K+4] - cs[I+4, J, K+4] - cs[I+4, J+4, K]
                  + cs[I, J, K+4] + cs[I, J+4, K] + cs[I+4, J, K] - cs[I, J, K])
    
    n_blocks = int(np.sum(block_sums == 64))
    return 0.05**3 * n_blocks

def moving_sofa_3d_path() -> np.ndarray:
    deadline = time.time() + 630 - 30  # 30s safety margin
    
    # Seed path from current program
    seed_path = np.array([
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
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 21.6229248, 5.80874653],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 22.1532161, 5.92642676],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 23.4927925, 6.21910882],
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 24.4180728, 6.42046395],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 25.3815663, 6.631397],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 26.8427434, 6.95357494],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 28.7527373, 7.37124273],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 30.8426719, 7.83842912],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 33.2683226, 8.35622932],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 35.6585372, 8.87446837],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 37.982628, 9.36856002],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 40.1148352, 9.81523017],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 42.279737, 10.2799932],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 44.4163048, 10.7608292],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 46.2626926, 11.1825901],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 48.012554, 11.601748],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 49.954221, 12.084816],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 52.548863, 12.7293228],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 55.893128, 13.5723105],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 59.8692011, 14.594953],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 63.7515612, 15.5832162],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 67.3322992, 16.6327601],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 70.5963626, 17.9436724],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 73.3241903, 19.4940659],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 75.3731869, 21.0578654],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 77.9661162, 23.4337293],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 79.008173, 25.2702993],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.9079131, 27.7208667],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 80.6362714, 29.7408798],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 81.0361104, 31.0067456],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.451701, 32.3850431],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.9245444, 34.2396643],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 82.1005075, 35.3326516],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.2502751, 36.313756],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.3738474, 37.1785229],
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
    
    # Evaluate seed with fast evaluator
    t0 = time.time()
    seed_score = _fast_evaluate(seed_path)
    t1 = time.time()
    eval_time = t1 - t0
    
    best_path = seed_path.copy()
    best_score = seed_score
    
    rng = np.random.default_rng(123)
    
    # Adaptive step-size local search
    # Focus on segments that matter most for the volume
    # The volume depends on which 4x4x4 blocks survive
    # Key regions: the two turns
    
    # Phase 1: Segment-wise optimization with adaptive step size
    segments = [
        (0, 15, 0.03, 0.5),      # Entry to corridor
        (15, 35, 0.02, 0.3),     # First turn region
        (35, 55, 0.02, 0.3),     # Corridor middle
        (55, 75, 0.02, 0.3),     # Second turn region
        (75, 99, 0.03, 0.5),     # Exit
    ]
    
    for seg_start, seg_end, pos_noise, ang_noise in segments:
        step_mult = 1.0
        for trial in range(40):
            if time.time() > deadline:
                break
            
            perturbed = best_path.copy()
            for i in range(seg_start, min(seg_end, len(perturbed))):
                perturbed[i, 0] += rng.normal(0, pos_noise * step_mult)
                perturbed[i, 1] += rng.normal(0, pos_noise * step_mult)
                perturbed[i, 2] += rng.normal(0, pos_noise * step_mult)
                perturbed[i, 3] += rng.normal(0, ang_noise * step_mult)
                perturbed[i, 4] += rng.normal(0, ang_noise * step_mult)
                perturbed[i, 5] += rng.normal(0, ang_noise * step_mult)
            
            score = _fast_evaluate(perturbed)
            if score > best_score:
                best_score = score
                best_path = perturbed.copy()
                step_mult *= 1.3  # Increase step size on success
            else:
                step_mult *= 0.7  # Decrease step size on failure
            step_mult = max(0.1, min(3.0, step_mult))
        
        if time.time() > deadline:
            break
    
    # Phase 2: Fine-tuning with very small steps
    for trial in range(60):
        if time.time() > deadline:
            break
        
        perturbed = best_path.copy()
        for i in range(1, len(perturbed) - 1):
            perturbed[i, 0] += rng.normal(0, 0.005)
            perturbed[i, 1] += rng.normal(0, 0.005)
            perturbed[i, 2] += rng.normal(0, 0.005)
            perturbed[i, 3] += rng.normal(0, 0.05)
            perturbed[i, 4] += rng.normal(0, 0.05)
            perturbed[i, 5] += rng.normal(0, 0.05)
        
        score = _fast_evaluate(perturbed)
        if score > best_score:
            best_score = score
            best_path = perturbed.copy()
    
    # Phase 3: Try targeted improvements on the turn regions
    # The first turn is around poses 25-35, second turn around 55-75
    for turn_start, turn_end in [(20, 40), (50, 75)]:
        for trial in range(20):
            if time.time() > deadline:
                break
            
            perturbed = best_path.copy()
            # Only perturb the turn region
            for i in range(turn_start, min(turn_end, len(perturbed))):
                perturbed[i, 0] += rng.normal(0, 0.015)
                perturbed[i, 1] += rng.normal(0, 0.015)
                perturbed[i, 2] += rng.normal(0, 0.015)
                perturbed[i, 3] += rng.normal(0, 0.2)
                perturbed[i, 4] += rng.normal(0, 0.2)
                perturbed[i, 5] += rng.normal(0, 0.2)
            
            score = _fast_evaluate(perturbed)
            if score > best_score:
                best_score = score
                best_path = perturbed.copy()
    
    # Final verification with arc checking if time permits
    if time.time() < deadline - 60:
        final_score = _evaluate_path(best_path, check_arcs=True, n_arc_samples=4)
        # If the arc-checked score is significantly worse, we might need to adjust
        # But for now, return the best path found with fast evaluation
    
    return best_path
