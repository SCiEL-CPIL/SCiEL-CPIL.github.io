import numpy as np
import time

N = 11
LOCAL_VERTS = np.column_stack((np.cos(np.linspace(0, 2*np.pi, 7)[:-1]), np.sin(np.linspace(0, 2*np.pi, 7)[:-1])))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    best_score = float('inf')
    best_state = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        base_coords = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (2,-1), (1,-2), (0,-2)]
        positions = np.array([[(q + r/2.0) * np.sqrt(3) * 1.05, r * 1.5 * 1.05] for q, r in base_coords])
        angles = np.zeros(N)
        cx, cy = 0.0, 0.0
        theta = 0.0
        
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        for i in range(2000):
            if time.time() > deadline:
                break
                
            # Calculate Current Cost
            # Transform vertices
            all_verts = np.zeros((N*6, 2))
            for j in range(N):
                c = np.cos(angles[j])
                s = np.sin(angles[j])
                R = np.array([[c, -s], [s, c]])
                all_verts[j*6:(j+1)*6] = LOCAL_VERTS @ R.T + positions[j]
                
            pts = all_verts - np.array([cx, cy])
            max_width = 0.0
            for k in range(3):
                angle = theta + k * np.pi / 3
                u = np.array([np.cos(angle), np.sin(angle)])
                projections = pts @ u
                width = np.max(projections) - np.min(projections)
                if width > max_width:
                    max_width = width
            current_cost = max_width / np.sqrt(3)
            
            if current_cost < best_score:
                best_score = current_cost
                best_state = (positions.copy(), angles.copy(), cx, cy, theta)
                
            # Move
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            old_pos = positions[idx].copy()
            positions[idx] += [dx, dy]
            
            move_angle = False
            old_angle = angles[idx]
            if np.random.rand() < 0.2:
                angles[idx] += np.random.normal(0, 0.1)
                move_angle = True
                
            old_cx, old_cy, old_theta = cx, cy, theta
            if np.random.rand() < 0.05:
                cx += np.random.normal(0, 0.1)
                cy += np.random.normal(0, 0.1)
                theta += np.random.normal(0, 0.05)
                
            # New Cost
            all_verts = np.zeros((N*6, 2))
            for j in range(N):
                c = np.cos(angles[j])
                s = np.sin(angles[j])
                R = np.array([[c, -s], [s, c]])
                all_verts[j*6:(j+1)*6] = LOCAL_VERTS @ R.T + positions[j]
                
            pts = all_verts - np.array([cx, cy])
            max_width = 0.0
            for k in range(3):
                angle = theta + k * np.pi / 3
                u = np.array([np.cos(angle), np.sin(angle)])
                projections = pts @ u
                width = np.max(projections) - np.min(projections)
                if width > max_width:
                    max_width = width
            new_cost = max_width / np.sqrt(3)
            
            if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / T):
                pass
            else:
                positions[idx] = old_pos
                if move_angle:
                    angles[idx] = old_angle
                cx, cy, theta = old_cx, old_cy, old_theta
                
            T = max(T * cooling, T_min)

    if best_state is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pos, ang, cx, cy, theta = best_state
    inner = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(best_score), float(np.degrees(theta))
