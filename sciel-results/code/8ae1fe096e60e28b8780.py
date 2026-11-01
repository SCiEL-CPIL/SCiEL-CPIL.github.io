# Inside construct, before solve_rows:
if k == 5 and n_rows >= 1:
    # Optimal ruler for 6 marks
    first_row = [0, 1, 4, 10, 12, 17]
    # Calculate mask for this row
    mask = 0
    for i in range(len(first_row)):
        for j in range(i+1, len(first_row)):
            d = first_row[j] - first_row[i]
            mask |= (1 << d)
    g_max = first_row[-1]
    rows = [first_row]
    # Start solving from row 1
    solve_rows(1, mask, g_max)
else:
    solve_rows(0, 0, 0)
