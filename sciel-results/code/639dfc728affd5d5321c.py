def design(problem):
    size = problem["pixels"]
    cell = []
    r_out_sq = (0.42 * size) ** 2
    r_in_sq = (0.22 * size) ** 2
    
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            dist_sq = x * x + y * y
            if r_in_sq < dist_sq < r_out_sq:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
