<<<<<<< SEARCH
def _try_swap_materials(materials_list, thicknesses, problem, deadline=None):
    """Try swapping material of a layer."""
    if len(materials_list) == 0:
        return materials_list, thicknesses, _evaluate(materials_list, thicknesses, problem)[0]
    
    materials = list(problem["materials"].keys())
    best_mr = _evaluate(materials_list, thicknesses, problem)[0]
    best_mats = list(materials_list)
    best_thick = list(thicknesses)
    
    for pos in range(len(materials_list)):
        current_mat = materials_list[pos]
        for mat in materials:
            if mat == current_mat:
                continue
            new_mats = list(materials_list)
            new_mats[pos] = mat
            m_opt, t_opt, mr_opt = _coordinate_descent(new_mats, thicknesses, problem, max_iter=100, deadline=deadline)
            
            if mr_opt < best_mr - 1e-12:
                best_mr = mr_opt
                best_mats = m_opt
                best_thick = t_opt
    
    return best_mats, best_thick, best_mr
=======
def _try_swap_materials(materials_list, thicknesses, problem, deadline=None):
    """Try swapping material of a layer."""
    if len(materials_list) == 0:
        return materials_list, thicknesses, _evaluate(materials_list, thicknesses, problem)[0]
    
    materials = list(problem["materials"].keys())
    best_mr = _evaluate(materials_list, thicknesses, problem)[0]
    best_mats = list(materials_list)
    best_thick = list(thicknesses)
    
    for pos in range(len(materials_list)):
        if time.time() > deadline:
            return best_mats, best_thick, best_mr
        current_mat = materials_list[pos]
        for mat in materials:
            if mat == current_mat:
                continue
            new_mats = list(materials_list)
            new_mats[pos] = mat
            m_opt, t_opt, mr_opt = _local_optimize(new_mats, thicknesses, problem, deadline)
            
            if mr_opt < best_mr - 1e-6:
                best_mr = mr_opt
                best_mats = m_opt
                best_thick = t_opt
    
    return best_mats, best_thick, best_mr
>>>>>>> REPLACE
