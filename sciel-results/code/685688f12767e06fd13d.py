    # Extract final parameters
    n = N
    inner_block = final_params[:3*n]
    inner_arr = np.array(inner_block).reshape(n, 3)
    inner_positions = inner_arr[:, :2]
    inner_angles = inner_arr[:, 2]
    
    ox = final_params[3*n]
    oy = final_params[3*n+1]
    oside = final_params[3*n+2]
    oangle = final_params[3*n+3]
