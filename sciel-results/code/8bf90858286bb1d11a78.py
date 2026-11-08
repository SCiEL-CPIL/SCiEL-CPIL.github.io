import numpy as np
import time

N = 11

def s_req(x, y):
    term1 = abs(x)
    term2 = (2.0 / np.sqrt(3.0)) * abs(y)
    term3 = abs(0.5 * x + (np.sqrt(3.0) / 2.0) * y)
    term4 = abs(0.5 * x - (np.sqrt(3.0) / 2.0) * y)
    return max(term1, term2, term3, term4)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_L = float('inf')
    best_positions = None
    
    # Initialization
    init_centers = np.array([
        [0.0, 0.0],
        [np.sqrt(3.0), 0.0],
        [np.sqrt(3.0)/2, 1.5],
        [-np.sqrt(3.0)/2, 1.5],
        [-np.sqrt(3.0), 0.0],
        [-np.sqrt(3.0)/2, -1.5],
        [np.sqrt(3.0)/2, -1.5],
        [1.5*np.sqrt(3.0), 1.5],
        [-1.5*np.sqrt(3.0), 1.5],
        [1.5*np.sqrt(3.0), -1.5],
        [-1.5*np.sqrt(3.0), -1.5]
    ])
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart == 0:
            centers = init_centers.copy()
        else:
            centers = init_centers + np.random.normal(0, 0.5, init_centers.shape)
            
        for i_iter in range(2000):
            if time.time() > deadline:
                break
                
            # 1. Check overlaps
            overlap_found = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    s = s_req(dx, dy)
                    if s < 2.0:
                        overlap_found = True
                        # Push apart
                        # Normalize direction
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 1e-6:
                            push_x = dx / norm
                            push_y = dy / norm
                        else:
                            push_x = 1.0
                            push_y = 0.0
                        
                        # Push magnitude
                        # We want to increase s to 2.
                        # s is roughly proportional to distance in this direction?
                        # Just push a fixed amount
                        push_mag = 0.05
                        centers[i, 0] += push_x * push_mag
                        centers[i, 1] += push_y * push_mag
                        centers[j, 0] -= push_x * push_mag
                        centers[j, 1] -= push_y * push_mag
                        
            if overlap_found:
                continue
            
            # 2. No overlaps, try to shrink
            s_reqs = np.array([s_req(centers[j, 0], centers[j, 1]) for j in range(N)])
            max_idx = np.argmax(s_reqs)
            
            # Move max_idx towards center
            # Step size
            step = 0.05
            centers[max_idx, 0] -= step * np.sign(centers[max_idx, 0])
            centers[max_idx, 1] -= step * np.sign(centers[max_idx, 1])
            
            # Check if this move caused overlap
            # If so, revert and try another point or stop
            # For simplicity, just check and revert if overlap
            valid_move = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if s_req(dx, dy) < 2.0:
                        valid_move = False
                        break
                if not valid_move: break
            
            if not valid_move:
                # Revert
                centers[max_idx, 0] += step * np.sign(centers[max_idx, 0])
                centers[max_idx, 1] += step * np.sign(centers[max_idx, 1])
                # Try moving other points?
                # Or just break to next restart
                # Let's try moving the second max point
                # Or just break
                break
                
        # End local search
        s_reqs = np.array([s_req(centers[j, 0], centers[j, 1]) for j in range(N)])
        final_L = 1.0 + np.max(s_reqs)
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if s_req(dx, dy) < 2.0 - 1e-6:
                    valid = False
                    break
            if not valid: break
        
        if valid and final_L < best_L:
            best_L = final_L
            best_positions = centers.copy()
            
    inner = []
    if best_positions is not None:
        for i in range(N):
            inner.append([best_positions[i, 0], best_positions[i, 1], 0.0])
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_L = 40.0
        
    return inner, (0.0, 0.0), best_L, 0.0
