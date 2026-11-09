import numpy as np
import time
from typing import List, Tuple

def moving_sofa_3d_path() -> np.ndarray:
    # Initial path from the current program (Seed)
    path = np.array([
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
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979]
    ], dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Grid setup
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    # Precompute grid indices for 4x4x4 block counting
    # A block is valid if all 4x4x4 points survive
    # We can use a sliding window sum or just check all blocks
    # There are (81-3) * (21-3) * (21-3) = 78 * 18 * 18 = 25272 blocks
    # Checking all blocks for each step is expensive.
    # Better approach: Maintain a "survived" boolean array.
    # After each step, update survived.
    # Finally, count blocks.

    def in_corridor_vec(x, y, z):
        # Region 1: x <= 0, 0 <= y <= 1, 0 <= z <= 1
        r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
        # Region 2: 0 <= x <= 1, 0 <= y <= 5, 0 <= z <= 1
        r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
        # Region 3: 0 <= x <= 1, 5 <= y <= 6, z >= 0
        r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
        return r1 | r2 | r3

    def get_rotation_matrix(yaw, pitch, roll):
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        roll_rad = np.deg2rad(roll)
        
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rx @ Ry @ Rz

    def evaluate_path(path):
        """
        Simulates the evaluator's process to calculate the score.
        """
        start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
        full_path = np.vstack([start_pose, path, end_pose])
        
        # Interpolate poses
        # The evaluator inserts poses so each step moves <= 0.05 (Euclidean) 
        # and changes each angle by <= 2 deg.
        # It does this by linear interpolation.
        
        interpolated = [full_path[0]]
        for i in range(1, len(full_path)):
            p0 = full_path[i-1]
            p1 = full_path[i]
            
            dist = np.linalg.norm(p1[:3] - p0[:3])
            angle_change = np.linalg.norm(p1[3:] - p0[3:])
            
            steps_pos = int(np.ceil(dist / 0.05))
            steps_ang = int(np.ceil(angle_change / 2.0))
            steps = max(1, steps_pos, steps_ang)
            
            for k in range(1, steps + 1):
                t = k / steps
                p = p0 + t * (p1 - p0)
                interpolated.append(p)
        
        interpolated = np.array(interpolated)
        
        survived = np.ones(len(points), dtype=bool)
        
        for i in range(len(interpolated) - 1):
            p0 = interpolated[i]
            p1 = interpolated[i+1]
            
            # Check p0
            R0 = get_rotation_matrix(p0[3], p0[4], p0[5])
            transformed0 = points @ R0.T + p0[:3]
            in_c0 = in_corridor_vec(transformed0[:, 0], transformed0[:, 1], transformed0[:, 2])
            survived &= in_c0
            
            # Check p1
            R1 = get_rotation_matrix(p1[3], p1[4], p1[5])
            transformed1 = points @ R1.T + p1[:3]
            in_c1 = in_corridor_vec(transformed1[:, 0], transformed1[:, 1], transformed1[:, 2])
            survived &= in_c1
            
            # Check rotation arc bounding box
            # The evaluator splits motion into X, Y, Z moves, then Rotation.
            # However, the interpolation is linear in all 6 DOFs.
            # The task says: "Each step is done as an x move, y move, z move, then a rotation about a single axis from the old to the new orientation."
            # This implies the evaluator might *decompose* the linear step into these sub-steps?
            # "The evaluator ... inserts linearly interpolated poses so each step moves at most 0.05 ... and changes each angle by at most 2."
            # "Each step is done as an x move, y move, z move, then a rotation..."
            # This phrasing suggests that for *each interpolated step*, the evaluator simulates:
            # 1. Move X
            # 2. Move Y
            # 3. Move Z
            # 4. Rotate
            # And checks the corridor for each of these sub-moves?
            # Or does it just check the bounding box of the arc?
            # "A grid point survives if it is in the corridor at the first pose and after each move, and the axis-aligned box bounding its rotation arc lies in the corridor."
            
            # If the step is small (<=0.05 pos, <=2 deg angle), the "arc" is very small.
            # The bounding box of the arc is approximately the bounding box of the two endpoints plus a small margin.
            # Given the small step size, checking the endpoints is often sufficient for the "in corridor" part.
            # The "axis-aligned box bounding its rotation arc lies in the corridor" is the critical constraint.
            
            # Approximation: The bounding box of the arc is the union of the bounding boxes of the two poses.
            # Let's check if the corners of the combined bounding box are in the corridor.
            min_coords = np.minimum(transformed0, transformed1)
            max_coords = np.maximum(transformed0, transformed1)
            
            # To be safe, we expand the box slightly? No, the task says "bounding its rotation arc".
            # For small rotations, the arc is close to the chord.
            # Let's check the 8 corners of the box defined by min/max of transformed0 and transformed1.
            # Actually, the box is per-point. "The axis-aligned box bounding its rotation arc".
            # This likely means: For each grid point, take the min/max coordinates it visits during the rotation.
            # Since the step is small, min/max of endpoints is a good approximation.
            
            corners = np.array([
                [min_coords[:, 0], min_coords[:, 1], min_coords[:, 2]],
                [min_coords[:, 0], min_coords[:, 1], max_coords[:, 2]],
                [min_coords[:, 0], max_coords[:, 1], min_coords[:, 2]],
                [min_coords[:, 0], max_coords[:, 1], max_coords[:, 2]],
                [max_coords[:, 0], min_coords[:, 1], min_coords[:, 2]],
                [max_coords[:, 0], min_coords[:, 1], max_coords[:, 2]],
                [max_coords[:, 0], max_coords[:, 1], min_coords[:, 2]],
                [max_coords[:, 0], max_coords[:, 1], max_coords[:, 2]]
            ]).reshape(-1, 3)
            
            in_c_corners = in_corridor_vec(corners[:, 0], corners[:, 1], corners[:, 2])
            # All 8 corners for each point must be in the corridor
            survived &= np.all(in_c_corners.reshape(len(points), 8), axis=1)
            
            if not np.any(survived):
                return 0.0
        
        # Count 4x4x4 blocks
        survived_grid = survived.reshape(81, 21, 21)
        blocks = 0
        # Optimization: Use numpy slicing
        for i in range(81 - 3):
            for j in range(21 - 3):
                for k in range(21 - 3):
                    if np.all(survived_grid[i:i+4, j:j+4, k:k+4]):
                        blocks += 1
        
        return blocks * (0.05 ** 3)

    best_path = path.copy()
    best_vol = evaluate_path(best_path)
    
    rng = np.random.default_rng(42)
    
    # Local Search
    # We will perturb the path and see if the volume increases.
    # Since the evaluation is expensive, we will limit the number of iterations.
    # We can also try to optimize specific "critical" poses (e.g., near corners).
    
    num_evals = 0
    max_evals = 500 # Adjust based on speed
    
    while time.time() < deadline and num_evals < max_evals:
        # Choose a strategy:
        # 1. Perturb a random pose
        # 2. Perturb a cluster of poses
        
        idx = rng.integers(0, len(best_path))
        new_path = best_path.copy()
        
        # Perturb position and orientation
        # Use larger steps for position, smaller for orientation?
        # The corridor is tight.
        new_path[idx, :3] += rng.normal(0, 0.02, 3)
        new_path[idx, 3:] += rng.normal(0, 1.0, 3)
        
        vol = evaluate_path(new_path)
        num_evals += 1
        
        if vol > best_vol:
            best_vol = vol
            best_path = new_path
        else:
            # If no improvement, try a smaller step or different index
            # Simple hill climbing: revert
            pass
    
    return best_path

if __name__ == "__main__":
    res = moving_sofa_3d_path()
    print(f"Shape: {res.shape}")
    print(f"First: {res[0]}")
    print(f"Last: {res[-1]}")
