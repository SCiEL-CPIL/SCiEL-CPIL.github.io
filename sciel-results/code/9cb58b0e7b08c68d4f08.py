"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """
    Vectorized reflectance calculation for a stack of layers.
    Returns an array of reflectance values (fraction, not percent) for the given wavelengths.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wls, r * r)
    
    # Precompute n and d for each layer
    ns = np.array([mat_dict[m] for m in materials])
    ds = np.array(thicknesses)
    
    # Initialize total matrix as identity for all wavelengths
    # Shape: (n_wls, 2, 2)
    M_total = np.zeros((len(wls), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n = ns[i]
        d = ds[i]
        if d < 1e-9:
            continue
        beta = 2.0 * np.pi * n * d / wls
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Characteristic matrix for layer i:
        # [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
        # M_layer shape: (n_wls, 2, 2)
        M_layer = np.zeros((len(wls), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = c
        M_layer[:, 0, 1] = 1j * s / n
        M_layer[:, 1, 0] = 1j * n * s
        M_layer[:, 1, 1] = c
        
        # M_total = M_total @ M_layer
        # Using einsum for batched matrix multiplication
        # (n_wls, 2, 2) @ (n_wls, 2, 2) -> (n_wls, 2, 2)
        M_total = np.einsum('wij,wjk->wik', M_total, M_layer)
        
    # Calculate input admittance Y
    # Y = (M[1,0] + M[1,1]*n_sub) / (M[0,0] + M[0,1]*n_sub)
    Y = (M_total[:, 1, 0] + M_total[:, 1, 1] * n_sub) / (M_total[:, 0, 0] + M_total[:, 0, 1] * n_sub)
    
    # Reflectance
    r = (n_amb - Y) / (n_amb + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """Calculate mean reflectance in percent."""
    if len(materials) == 0:
        # Bare substrate reflectance
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return 100.0 * (r * r)
    refls = _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
    return 100.0 * np.mean(refls)

def _decode_and_evaluate(x, wls, n_amb, n_sub, mat_dict, problem):
    """
    Decode the optimization vector x into a valid stack and evaluate it.
    x: array of length 40.
    x[0:20] : material genes (0.0-1.0). <0.5 -> 'L', >=0.5 -> 'H'
    x[20:40]: thickness genes (0.0-1.0). Mapped to [0, max_layer_nm]
    """
    max_layers = problem.get("max_layers", 20)
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Decode materials
    mat_genes = x[:max_layers]
    th_genes = x[max_layers:2*max_layers]
    
    materials = []
    thicknesses = []
    
    for i in range(max_layers):
        # If material gene is close to 0, treat as "empty" (skip layer)
        # But to ensure we use a contiguous stack from air side, we should
        # probably just evaluate all 20 and let zero-thickness layers vanish?
        # The problem says "list of 'L'/'H' labels". 
        # If we include a layer with 0 thickness, is it valid?
        # "every thickness must lie in [0, max_layer_nm]"
        # So 0 is allowed.
        # However, having a 0-thickness layer in the middle might be physically
        # equivalent to removing it, but computationally it adds a matrix
        # multiplication. 
        # Let's just include all 20 layers. If thickness is ~0, it's effectively gone.
        # But wait, if we have L, 0, H, it's just H. 
        # The evaluator checks constraints. 
        # Let's just generate 20 layers.
        
        mat = 'L' if mat_genes[i] < 0.5 else 'H'
        # Map thickness gene to physical thickness
        # Using a scaling that allows fine control but covers the range
        # We can use a non-linear mapping or just linear.
        # Linear: t = gene * max_layer_nm
        t = th_genes[i] * max_layer_nm
        
        # Enforce constraints
        if t < 0: t = 0
        if t > max_layer_nm: t = max_layer_nm
        
        materials.append(mat)
        thicknesses.append(t)
        
    # Check total thickness constraint
    total_thick = sum(thicknesses)
    if total_thick > max_total_nm:
        # Penalize heavily or scale down?
        # Scaling down preserves the relative structure.
        if total_thick > 0:
            scale = max_total_nm / total_thick
            thicknesses = [t * scale for t in thicknesses]
        else:
            # All zero? Return a high penalty
            return 100.0 # High reflectance penalty
    
    # Evaluate
    score = _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
    
    # Optional: Add a small penalty for unused layers to encourage sparsity?
    # No, the task allows up to 20. More layers can be better.
    
    return score

def design(problem):
    """
    Design a broadband antireflection coating.
    Strategy:
    1. Use Differential Evolution to search a 40-dimensional space.
    2. The space encodes 20 layers, each with a material bit and a thickness.
    3. This allows the optimizer to discover both the optimal sequence and thicknesses.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin (limit is 150s)
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Bounds for DE
    # 20 material genes [0, 1]
    # 20 thickness genes [0, 1]
    bounds = [(0.0, 1.0)] * (2 * max_layers)
    
    # DE requires a population size. 
    # With 40 dimensions, a population of 20-30 is reasonable.
    # Max iterations: We have ~140s. 
    # Each evaluation is ~1-2ms (vectorized).
    # 20 layers * 401 wls * 2x2 matrix mult.
    # Let's estimate: 20 * 401 * (8 flops) ~ 64k flops. Very fast.
    # We can afford thousands of evaluations.
    
    popsize = 15
    max_iter = 500 # Upper bound, will stop earlier due to time
    
    def objective(x):
        if time.time() - start_time > deadline:
            return 100.0 # High penalty to stop
        return _decode_and_evaluate(x, wls, n_amb, n_sub, mat_dict, problem)
    
    # Initial best is bare substrate or a simple guess
    # Bare substrate:
    r_bare = (n_amb - n_sub) / (n_amb + n_sub)
    best_score = 100.0 * (r_bare * r_bare)
    best_materials = []
    best_thicknesses = []
    
    # Try a simple quarter-wave stack as a better initial best
    # L quarter wave at 600nm
    d_l = 600.0 / (4.0 * mat_dict['L'])
    best_materials = ['L']
    best_thicknesses = [d_l]
    best_score = _mean_reflectance(best_materials, best_thicknesses, wls, n_amb, n_sub, mat_dict)
    
    try:
        # Run Differential Evolution
        # tol=1e-5 is tight, but we have time
        # disp=False to keep output clean
        # callback to monitor time
        def callback(xk, convergence):
            if time.time() - start_time > deadline:
                raise TimeoutError("Time limit reached")
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_iter, 
            popsize=popsize, 
            tol=1e-5, 
            mutation=(0.5, 1.5), 
            recombination=0.9, 
            seed=42,
            callback=callback,
            polish=False # Polish is local search, might be slow if not careful, but DE is global. Let's skip polish to save time for more DE iterations or do a manual polish later.
        )
        
        if result.fun < best_score:
            best_x = result.x
            # Decode result
            mat_genes = best_x[:max_layers]
            th_genes = best_x[max_layers:2*max_layers]
            
            new_materials = []
            new_thicknesses = []
            for i in range(max_layers):
                mat = 'L' if mat_genes[i] < 0.5 else 'H'
                t = th_genes[i] * problem.get("max_layer_nm", 400)
                
                # Clip
                if t < 0: t = 0
                if t > problem.get("max_layer_nm", 400): t = problem.get("max_layer_nm", 400)
                
                new_materials.append(mat)
                new_thicknesses.append(t)
                
            # Check total thickness
            total_thick = sum(new_thicknesses)
            max_total_nm = problem.get("max_total_nm", 1000)
            if total_thick > max_total_nm:
                if total_thick > 0:
                    scale = max_total_nm / total_thick
                    new_thicknesses = [t * scale for t in new_thicknesses]
                else:
                    # Fallback
                    pass
            
            # Remove layers with negligible thickness to clean up?
            # The evaluator accepts 0 thickness. But cleaner is better.
            # However, removing layers changes the index. 
            # If we keep 0-thickness layers, they are just identity matrices.
            # Let's keep them to avoid re-indexing issues, or just filter them.
            # Filtering is safer for "list of labels".
            
            final_materials = []
            final_thicknesses = []
            for m, t in zip(new_materials, new_thicknesses):
                if t > 1.0: # > 1nm
                    final_materials.append(m)
                    final_thicknesses.append(t)
            
            if len(final_materials) > 0:
                # Re-evaluate the filtered stack to be sure
                final_score = _mean_reflectance(final_materials, final_thicknesses, wls, n_amb, n_sub, mat_dict)
                if final_score < best_score:
                    best_score = final_score
                    best_materials = final_materials
                    best_thicknesses = final_thicknesses
            else:
                # If all layers were zero, keep the previous best
                pass
                
    except TimeoutError:
        pass
    except Exception as e:
        # If DE fails, fall back to the initial best
        pass

    # If time remains, do a quick local polish on the best result
    if time.time() - start_time < deadline - 10:
        if len(best_materials) > 0:
            # Simple coordinate descent polish
            current_thicks = np.array(best_thicknesses, dtype=float)
            current_score = best_score
            step_size = 5.0
            max_layer_nm = problem.get("max_layer_nm", 400)
            max_total_nm = problem.get("max_total_nm", 1000)
            
            for _ in range(100): # Max 100 polish iterations
                if time.time() - start_time > deadline:
                    break
                improved = False
                for j in range(len(current_thicks)):
                    for delta in [step_size, -step_size, step_size/2, -step_size/2]:
                        new_thicks = current_thicks.copy()
                        new_thicks[j] += delta
                        if new_thicks[j] < 0 or new_thicks[j] > max_layer_nm:
                            continue
                        if sum(new_thicks) > max_total_nm:
                            continue
                        new_score = _mean_reflectance(best_materials, new_thicks, wls, n_amb, n_sub, mat_dict)
                        if new_score < current_score - 1e-6:
                            current_thicks = new_thicks
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step_size *= 0.7
                    if step_size < 0.1:
                        break
                else:
                    if current_score < best_score:
                        best_score = current_score
                        best_thicknesses = current_thicks.tolist()
                        
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
