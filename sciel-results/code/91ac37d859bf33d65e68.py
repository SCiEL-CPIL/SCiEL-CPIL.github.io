def design(problem):
    size = problem["pixels"]
    cell = []
    block_size = 4
    for i in range(size):
        row = []
        block_i = i // block_size
        for j in range(size):
            block_j = j // block_size
            if (block_i + block_j) % 2 == 0:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
