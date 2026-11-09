import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 14

def _calculate_score_circular(angles):
    """
    Calculate the score for points on a unit circle given by angles.
    Score = min_triangle_area / hull_area
    """
    if not np.all(np.isfinite(angles)):
        return 0.0
        
    # Normalize angles to [0, 2*pi)
    angles = np.sort(angles % (2 * np.pi))
    
    # Hull Area (Polygon Area)
    # Shoelace formula
    x = np.cos(angles)
    y = np.sin(angles)
    # x_i y_{i+1} - x_{i+1} y_i
    # Wrap around
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    hull_area = 0.5 * np.sum(x * y_next - x_next * y)
    
    if hull_area <= 0:
        return 0.0
        
    # Min Triangle Area
    # Area(i, j, k) = 2 * sin((theta_j - theta_i)/2) * sin((theta_k - theta_j)/2) * sin((theta_k - theta_i)/2)
    # We need min over all i < j < k
    
    min_area = np.inf
    n = len(angles)
    
    for i in range(n):
        for j in range(i + 1, n):
            a1 = angles[j] - angles[i]
            for k in range(j + 1, n):
                a2 = angles[k] - angles[j]
                a3 = angles[k] - angles[i]
                # Note: a3 = a1 + a2
                area = 2.0 * np.sin(0.5 * a1) * np.sin(0.5 * a2) * np.sin(0.5 * a3)
                if area < min_area:
                    min_area = area
                    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0
        
    return min_area / hull_area

def _negative_score_circular(angles):
    return -_calculate_score_circular(angles)

def construct():
    start_time = time.time()
    timeout = 145.0
    
    best_points = None
    best_score = 0.0
    best_angles = None
    
    # Initial guess: Regular polygon
    init_angles = 2 * np.pi * np.arange(N) / N
    best_angles = init_angles.copy()
    best_score = _calculate_score_circular(best_angles)
    
    # Optimization
    # Use Differential Evolution for global search in 1D angular space
    # Bounds: [0, 2*pi) for each angle
    bounds = [(0.0, 2.0 * np.pi)] * N
    
    try:
        res_de = differential_evolution(
            _negative_score_circular, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-4, 
            disp=False,
            polish=False,
            seed=42
        )
        
        if res_de.x is not None:
            cand_angles = res_de.x
            cand_score = _calculate_score_circular(cand_angles)
            if cand_score > best_score:
                best_score = cand_score
                best_angles = cand_angles.copy()
                
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        # Convert best angles to points
        best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
        return best_points
    
    # Local Refinement using Powell
    try:
        def _time_check_callback(xk):
            if time.time() - start_time > timeout - 5:
                raise TimeoutError("Local optimization timeout")
        
        res_powell = minimize(
            _negative_score_circular, 
            best_angles, 
            method='Powell', 
            options={'maxiter': 100, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8},
            callback=_time_check_callback
        )
        
        if res_powell.x is not None:
            cand_angles = res_powell.x
            cand_score = _calculate_score_circular(cand_angles)
            if cand_score > best_score:
                best_score = cand_score
                best_angles = cand_angles.copy()
                
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # If time remains, try Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        try:
            def _nm_time_check_callback(xk):
                if time.time() - start_time > timeout - 5:
                    raise TimeoutError("NM optimization timeout")
            
            res_nm = minimize(
                _negative_score_circular, 
                best_angles, 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'disp': False},
                callback=_nm_time_check_callback
            )
            
            if res_nm.x is not None:
                cand_angles = res_nm.x
                cand_score = _calculate_score_circular(cand_angles)
                if cand_score > best_score:
                    best_score = cand_score
                    best_angles = cand_angles.copy()
                    
        except TimeoutError:
            pass
        except Exception:
            pass
            
    # Convert best angles to points
    best_points = np.column_stack((np.cos(best_angles), np.sin(best_angles)))
    
    return best_points
